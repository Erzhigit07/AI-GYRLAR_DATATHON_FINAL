"""Offline Qwen baseline with isolated per-video workers and fixed development split.

Only aggregate/technical progress is printed. Private responses and evidence stay
in a new output directory. No model training, network downloads, or API publication.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import math
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import av
import psutil
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.forte.evaluation import evaluate_manifest  # noqa: E402
from src.forte.qwen_baseline import (  # noqa: E402
    aggregate_windows,
    parse_window_response,
    to_candidate,
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def emit(data):
    print(json.dumps(data, ensure_ascii=True, allow_nan=False), flush=True)


def guard(event, args):
    if event == "socket.connect" and args[1][0] != "127.0.0.1":
        raise RuntimeError("Only loopback connections permitted")
    if event == "socket.getaddrinfo" and args[0] not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("External DNS is disabled")


def request(port, endpoint, *, body=None, timeout=10):
    if endpoint not in {"/health", "/v1/models", "/v1/chat/completions"}:
        raise ValueError("Unexpected model endpoint")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["FORTE_LOCAL_TOKEN"],
        },
    )
    with opener.open(req, timeout=max(0.1, timeout)) as response:
        return json.load(response)


def sample_video(path, settings, deadline):
    frames, next_time, first_pts = [], 0.0, None
    with av.open(str(path), options={"protocol_whitelist": "file,pipe"}) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        for decoded in container.decode(stream):
            if time.monotonic() >= deadline:
                raise TimeoutError("Decode deadline")
            if decoded.pts is None or decoded.is_corrupt:
                raise ValueError("Corrupt frame or missing PTS")
            absolute = float(decoded.pts * decoded.time_base)
            if first_pts is None:
                first_pts = absolute
            stamp = absolute - first_pts
            if stamp + 1e-6 < next_time:
                continue
            picture = decoded.to_image().convert("RGB")
            picture.thumbnail(
                (settings["image_max_side"], settings["image_max_side"]), Image.Resampling.LANCZOS
            )
            buffer = io.BytesIO()
            picture.save(buffer, format="JPEG", quality=settings["jpeg_quality"])
            raw = buffer.getvalue()
            frames.append(
                {
                    "time_sec": stamp,
                    "jpeg_sha256": hashlib.sha256(raw).hexdigest(),
                    "width": picture.width,
                    "height": picture.height,
                    "base64": base64.b64encode(raw).decode("ascii"),
                }
            )
            if len(frames) > settings["max_sampled_frames"]:
                raise ValueError("Frame limit exceeded")
            step = settings["sample_step_sec"]
            next_time = (math.floor((stamp + 1e-6) / step) + 1) * step
    if not frames:
        raise ValueError("Video has no decodable sampled frames")
    return frames


def windows(count, size, overlap):
    if count < 1 or not 0 <= overlap < size:
        raise ValueError("Invalid window settings")
    result, start = [], 0
    while True:
        end = min(count, start + size)
        result.append(list(range(start, end)))
        if end == count:
            return result
        start = end - overlap


def output_schema(stamps, rules):
    """Constrain verdict, completeness and evidence together before generation."""
    if not isinstance(stamps, (list, tuple)) or not stamps:
        raise ValueError("nonempty timestamp list required")
    for value in stamps:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("timestamps must be finite nonnegative numbers")
    if (
        not isinstance(rules, (list, tuple))
        or not rules
        or not all(
            isinstance(value, str) and value.strip() and len(value) <= 120 for value in rules
        )
    ):
        raise ValueError("nonempty bounded rule identifiers required")
    if len(set(stamps)) != len(stamps) or len(set(rules)) != len(rules):
        raise ValueError("duplicate timestamps or rules")
    event = {
        "type": "object",
        "additionalProperties": False,
        "required": ["time_seconds", "rule_id", "description"],
        "properties": {
            "time_seconds": {"type": "number", "enum": list(stamps)},
            "rule_id": {"type": "string", "enum": list(rules)},
            "description": {"type": "string", "minLength": 1, "maxLength": 200},
        },
    }

    def branch(verdict, complete, events, reason):
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["verdict", "analysis_complete", "events", "uncertainty_reason"],
            "properties": {
                "verdict": {"const": verdict},
                "analysis_complete": {"const": complete},
                "events": events,
                "uncertainty_reason": reason,
            },
        }

    return {
        "oneOf": [
            branch(
                "violation",
                True,
                {"type": "array", "minItems": 1, "maxItems": 3, "items": event},
                {"const": ""},
            ),
            branch(
                "no_violation",
                True,
                {"type": "array", "minItems": 0, "maxItems": 0},
                {"const": ""},
            ),
            branch(
                "uncertain",
                False,
                {"type": "array", "minItems": 0, "maxItems": 0},
                {"type": "string", "minLength": 1, "maxLength": 200},
            ),
        ]
    }


def run_worker(args):
    settings = json.loads(args.worker_settings.read_text(encoding="utf-8"))
    deadline = time.monotonic() + args.worker_budget
    result_dir = args.worker_result.parent
    started = time.monotonic()
    frames, groups, parsed_windows, technical = [], [], [], []
    active_request = None
    try:
        frames = sample_video(args.worker_input, settings, deadline)
        groups = windows(len(frames), settings["frames_per_window"], settings["overlap_frames"])
        overview_size = min(settings["overview_frames"], len(frames))
        overview = (
            [0]
            if overview_size == 1
            else [
                round(index * (len(frames) - 1) / (overview_size - 1))
                for index in range(overview_size)
            ]
        )
        groups = [overview, *groups]
        for index, selected in enumerate(groups):
            if time.monotonic() >= deadline:
                raise TimeoutError("Video worker deadline")
            stamps = [round(frames[item]["time_sec"], 3) for item in selected]
            kind = "overview" if index == 0 else "local"
            scope = (
                "OVERVIEW: evenly selected frames across this video's full sampled timeline. "
                "Gaps are intentional; infer no unseen event."
                if index == 0
                else "LOCAL WINDOW: chronological frames at the planned dense sampling interval."
            )
            content = [{"type": "text", "text": settings["prompt"] + "\n\n" + scope}]
            for item, stamp in zip(selected, stamps, strict=True):
                content.extend(
                    [
                        {"type": "text", "text": f"Observation frame at {stamp:.3f} seconds:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + frames[item]["base64"]
                            },
                        },
                    ]
                )
            body = {
                "model": settings["model_id"],
                "messages": [{"role": "user", "content": content}],
                **settings["generation"],
                "stream": False,
                "cache_prompt": False,
                "response_format": {
                    "type": "json_object",
                    "schema": output_schema(stamps, settings["rule_ids"]),
                },
            }
            before = time.monotonic()
            active_request = {
                "index": index,
                "kind": kind,
                "frame_indices": selected,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "request_sha256": hashlib.sha256(
                    json.dumps(body, sort_keys=True).encode()
                ).hexdigest(),
            }
            write_new(result_dir / f"request_{index:03d}.json", active_request)
            try:
                response = request(
                    settings["port"],
                    "/v1/chat/completions",
                    body=body,
                    timeout=min(settings["request_timeout_sec"], deadline - before),
                )
            except (TimeoutError, socket.timeout):
                raise TimeoutError("Model request timeout") from None
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                    raise TimeoutError("Model request timeout") from None
                raise RuntimeError("Local model transport error") from None
            write_new(result_dir / f"window_{index:03d}_private.json", response)
            choice = response["choices"][0]
            message = choice["message"]
            parsed = parse_window_response(
                message.get("content") or "",
                finish_reason=choice.get("finish_reason"),
                supplied_times=stamps,
                allowed_rule_ids=set(settings["rule_ids"]),
                refusal=bool(message.get("refusal")),
            )
            # Preserve exact decoded PTS for the existing evidence contract.
            for event in parsed.get("events", []):
                event["time_seconds"] = next(
                    frames[item]["time_sec"]
                    for item, stamp in zip(selected, stamps, strict=True)
                    if abs(stamp - event["time_seconds"]) <= 0.001
                )
            parsed_windows.append(parsed)
            technical.append(
                {
                    "index": index,
                    "kind": kind,
                    "frame_indices": selected,
                    "status": parsed["status"],
                    "seconds": time.monotonic() - before,
                    "usage": response.get("usage"),
                    "finish_reason": choice.get("finish_reason"),
                    "request_sha256": hashlib.sha256(
                        json.dumps(body, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
            write_new(
                result_dir / f"window_result_{index:03d}_private.json",
                {"parsed": parsed, "technical": technical[-1]},
            )
            active_request = None
        aggregate = aggregate_windows(parsed_windows, expected_windows=len(groups))
        write_new(
            args.worker_result,
            {
                "status": "finished",
                "aggregation": aggregate,
                "candidate": to_candidate(aggregate),
                "parsed_windows": parsed_windows,
                "technical_windows": technical,
                "sampled_frames": [
                    {k: v for k, v in frame.items() if k != "base64"} for frame in frames
                ],
                "wall_sec": time.monotonic() - started,
            },
        )
    except Exception as exc:
        if active_request is not None:
            status = "timeout" if isinstance(exc, TimeoutError) else "error"
            technical.append(
                {**active_request, "status": status, "seconds": time.monotonic() - before}
            )
            parsed = parse_window_response(
                "",
                finish_reason=None,
                supplied_times=stamps,
                allowed_rule_ids=set(settings["rule_ids"]),
                transport_status=status,
            )
            parsed_windows.append(parsed)
            write_new(
                result_dir / f"window_result_{active_request['index']:03d}_private.json",
                {"parsed": parsed, "technical": technical[-1]},
            )
        write_new(
            args.worker_result,
            {
                "status": "timeout" if isinstance(exc, TimeoutError) else "error",
                "error_type": type(exc).__name__,
                "wall_sec": time.monotonic() - started,
                "technical_windows": technical,
                "parsed_windows": parsed_windows,
                "expected_windows": len(groups),
                "sampled_frames": [
                    {k: v for k, v in frame.items() if k != "base64"} for frame in frames
                ],
            },
        )
        return 1
    return 0


class Runtime:
    def __init__(self, command, root, output, environment):
        self.command, self.root, self.output, self.environment = command, root, output, environment
        self.process = None
        self.log = None
        self.generation = 0

    def start(self, port, timeout_sec=180):
        self.generation += 1
        self.log = (self.output / f"model_server_{self.generation:02d}.log").open("xb")
        self.process = subprocess.Popen(
            self.command,
            cwd=self.root,
            env=self.environment,
            stdout=self.log,
            stderr=self.log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        started = time.monotonic()
        while time.monotonic() - started < timeout_sec:
            if self.process.poll() is not None:
                raise RuntimeError("Local model failed during startup")
            try:
                remaining = timeout_sec - (time.monotonic() - started)
                if request(port, "/health", timeout=min(2, remaining)).get("status") == "ok":
                    return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(0.5)
        raise TimeoutError("Local model startup deadline")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log:
            self.log.close()


def run(args):
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    root = (REPO / configuration["data_root"]).resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.stage == "smoke":
        keys = set(configuration["smoke_keys"])
        manifest["videos"] = [
            record
            for record in manifest["videos"]
            if record["split"] == "holdout" or record["key"] in keys
        ]
    selected = [record for record in manifest["videos"] if record["split"] == "development"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prompt_path = (REPO / configuration["prompt_file"]).resolve(strict=True)
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if sha(prompt_path) != configuration["prompt_sha256"]:
        raise ValueError("Frozen prompt changed")
    if sha(REPO / "src/forte/qwen_baseline.py") != configuration["adapter_sha256"]:
        raise ValueError("Frozen adapter changed")
    if sha(Path(__file__)) != configuration["runner_sha256"]:
        raise ValueError("Frozen runner changed")
    for filename, expected in configuration["code_sha256"].items():
        if sha(REPO / filename) != expected:
            raise ValueError("Frozen pipeline component changed")
    if sha(REPO / configuration["rules_file"]) != configuration["rules_sha256"]:
        raise ValueError("Frozen rules changed")
    source_info = json.loads((root / "compact_sources.json").read_text(encoding="utf-8"))
    for source in source_info["files"]:
        if sha(root / "compact" / source["name"]) != source["sha256"]:
            raise ValueError("Local model/runtime archive integrity failed")
    protected = {str(path): sha(path) for path in [manifest_path, prompt_path, args.config]}
    for directory in [root / "dense-review", root / "prompt-recheck"]:
        protected.update({str(path): sha(path) for path in directory.rglob("*") if path.is_file()})
    write_new(output / "protected_before.json", protected)
    write_new(output / "configuration.json", configuration)
    model_command = [
        str((REPO / value).resolve()) if value.startswith("outputs/") else value
        for value in configuration["server_command"]
    ]
    if sha(model_command[0]) != configuration["runtime_executable_sha256"]:
        raise ValueError("Runtime executable changed")
    runtime = None
    started = time.monotonic()
    samples, stops, observer = [], threading.Event(), None
    aggregate_status = "STARTING"
    evaluation = None
    token = secrets.token_urlsafe(32)
    os.environ["FORTE_LOCAL_TOKEN"] = token
    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith(("LLAMA_", "MTMD_")) or name.lower().endswith("_proxy"):
            del environment[name]
    environment.update(
        LLAMA_API_KEY=token,
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    outcomes_metadata = []
    try:
        if psutil.virtual_memory().available / 2**30 < configuration["minimum_start_ram_gib"]:
            raise MemoryError("Insufficient free RAM for guarded model startup")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", configuration["port"])) == 0:
                raise RuntimeError("Model port is occupied; existing process preserved")
        runtime = Runtime(model_command, root, output, environment)

        def monitor():
            low = 0
            while not stops.wait(2):
                available = psutil.virtual_memory().available / 2**30
                item = {"elapsed_sec": time.monotonic() - started, "available_ram_gib": available}
                try:
                    if runtime.process and runtime.process.poll() is None:
                        item["server_rss_gib"] = (
                            psutil.Process(runtime.process.pid).memory_info().rss / 2**30
                        )
                except psutil.Error:
                    pass
                samples.append(item)
                low = low + 1 if available < configuration["stop_ram_gib"] else 0
                if low >= 3:
                    stops.set()
                    runtime.stop()
                    return

        observer = threading.Thread(target=monitor, daemon=True)
        observer.start()
        runtime.start(configuration["port"])
        worker_settings = {
            **configuration["sampling"],
            "generation": configuration["generation"],
            "port": configuration["port"],
            "prompt": prompt,
            "model_id": request(configuration["port"], "/v1/models")["data"][0]["id"],
            "rule_ids": configuration["rule_ids"],
            "request_timeout_sec": configuration["request_timeout_sec"],
        }
        settings_path = output / "worker_settings.json"
        write_new(settings_path, worker_settings)
        emit({"stage": args.stage, "model_ready": True, "selected": len(selected)})
        total_deadline = time.monotonic() + configuration["total_budget_sec"]

        def analyzer(neutral_video, *, deadline):
            index = len(outcomes_metadata)
            folder = output / "responses_private" / f"attempt_{index:03d}"
            folder.mkdir(parents=True, exist_ok=False)
            result_path = folder / "worker_result_private.json"
            item = {"attempt_index": index, "status": "starting", "worker_launched": False}
            outcomes_metadata.append(item)
            if stops.is_set() or time.monotonic() >= total_deadline:
                item["status"] = "budget_or_memory_stop"
                raise TimeoutError("Global run budget or memory guard reached")
            if runtime.process.poll() is not None:
                restart_budget = min(deadline, total_deadline) - time.monotonic() - 30
                if (
                    runtime.generation > configuration["maximum_server_restarts"]
                    or restart_budget <= 0
                ):
                    item["status"] = "model_unavailable"
                    raise RuntimeError("Local model is unavailable")
                try:
                    runtime.start(configuration["port"], timeout_sec=min(60, restart_budget))
                except Exception:
                    item["status"] = "restart_failed"
                    runtime.stop()
                    raise
            budget = min(
                deadline - time.monotonic() - configuration["evidence_reserve_sec"],
                total_deadline - time.monotonic(),
            )
            if budget <= 0:
                item["status"] = "timeout"
                raise TimeoutError("No analyzer budget remaining")
            before = time.monotonic()
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker-input",
                neutral_video,
                "--worker-settings",
                str(settings_path),
                "--worker-result",
                str(result_path),
                "--worker-budget",
                str(budget),
            ]
            with (folder / "worker.stderr.log").open("xb") as error_log:
                worker = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=environment,
                    stdout=subprocess.DEVNULL,
                    stderr=error_log,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                item["worker_launched"] = True
                try:
                    worker.wait(timeout=budget)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
                    item.update(status="timeout", seconds=time.monotonic() - before)
                    runtime.stop()
                    raise TimeoutError("Isolated analyzer process deadline") from None
            item["seconds"] = time.monotonic() - before
            if not result_path.is_file():
                item["status"] = "worker_error"
                raise RuntimeError("Worker exited without a complete result")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            item["status"] = result["status"]
            if result["status"] == "timeout":
                runtime.stop()
                raise TimeoutError("Model request deadline")
            if result["status"] != "finished":
                raise RuntimeError("Analyzer worker failed")
            item["aggregation_status"] = result["aggregation"]["status"]
            item["window_status_counts"] = result["aggregation"]["window_status_counts"]
            emit(
                {
                    "stage": args.stage,
                    "attempt_complete": index + 1,
                    "total": len(selected),
                    "status": item["aggregation_status"],
                    "seconds": round(item["seconds"], 2),
                    "window_status_counts": item["window_status_counts"],
                }
            )
            return result["candidate"]

        versions = {
            name: importlib.metadata.version(name) for name in ["av", "Pillow", "psutil", "numpy"]
        }
        evaluation = evaluate_manifest(
            manifest,
            analyzer,
            output / "evaluation",
            base_dir=root,
            run_metadata={
                "analyzer_version": "qwen-local-binary-baseline-v1",
                "code_revision": configuration["code_revision"],
                "settings": configuration,
                "package_versions": versions,
                "stage": args.stage,
            },
            split="development",
            allow_holdout=False,
            timeout_sec=configuration["video_timeout_sec"],
        )
        aggregate_status = "FINISHED"
    except Exception as exc:
        aggregate_status = "BLOCKED_" + type(exc).__name__
        emit({"status": aggregate_status})
    finally:
        stops.set()
        if observer:
            observer.join(timeout=5)
        if runtime:
            runtime.stop()
        write_new(output / "attempt_metadata.json", outcomes_metadata)
        write_new(output / "memory_samples.json", samples)
        unchanged = all(
            Path(path).is_file() and sha(path) == value for path, value in protected.items()
        )
        summary = {
            "status": aggregate_status,
            "stage": args.stage,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_sec": time.monotonic() - started,
            "protected_files_unchanged": unchanged,
            "protected_file_count": len(protected),
            "server_stopped": runtime is None
            or runtime.process is None
            or runtime.process.poll() is not None,
            "min_available_ram_gib": min((s["available_ram_gib"] for s in samples), default=None),
            "peak_server_rss_gib": max((s.get("server_rss_gib", 0) for s in samples), default=None),
            "attempt_status_counts": dict(Counter(item["status"] for item in outcomes_metadata)),
            "workers_launched": sum(item["worker_launched"] for item in outcomes_metadata),
            "model_window_requests_attempted": len(
                list((output / "responses_private").glob("attempt_*/request_*.json"))
            ),
            "metrics": evaluation["metrics"] if evaluation else None,
            "model_weights_trained": False,
            "independent_evaluation": False,
        }
        write_new(output / "run_summary.json", summary)
        emit(summary)
    return 0 if aggregate_status == "FINISHED" and unchanged else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", choices=["smoke", "full"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--worker-settings", type=Path)
    parser.add_argument("--worker-result", type=Path)
    parser.add_argument("--worker-budget", type=float)
    args = parser.parse_args()
    sys.addaudithook(guard)
    if args.worker_input:
        return run_worker(args)
    if args.config is None or args.stage is None or args.output is None:
        parser.error("--config, --stage and --output are required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

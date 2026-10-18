"""Evaluate a frozen manual-type run sequentially through the loopback API.

Annotations are only used after predictions. No labels, filenames or true event
times are included in inference requests. All private artifacts stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forte.type_evaluation import evaluate_types  # noqa: E402


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def remux(source, target):
    import av

    with av.open(str(source)) as origin, av.open(str(target), "w", format="mov") as destination:
        stream = origin.streams.video[0]
        output = destination.add_stream_from_template(stream)
        for packet in origin.demux(stream):
            if packet.dts is not None:
                packet.stream = output
                destination.mux(packet)


def run(run_dir, *, video_ids=None, output_dir=None):
    plan = read(run_dir / "evaluation_plan.json")
    for filename, field in (
        ("config.json", "config_sha256"),
        ("typed_manual_labels.json", "typed_labels_sha256"),
        ("rules.json", "rules_sha256"),
        ("prompt.txt", "prompt_sha256"),
    ):
        if sha(run_dir / filename) != plan[field]:
            raise ValueError("Frozen run integrity mismatch")
    dataset = read(run_dir / "typed_manual_labels.json")
    if video_ids:
        if len(set(video_ids)) != len(video_ids):
            raise ValueError("Duplicate requested video IDs")
        by_id = {row["video_id"]: row for row in dataset["videos"]}
        if set(video_ids) - by_id.keys():
            raise ValueError("Requested video ID is not in the frozen dataset")
        if output_dir is None or output_dir == run_dir:
            raise ValueError("Subset demonstrations require a separate output directory")
        dataset = {**dataset, "videos": [by_id[video_id] for video_id in video_ids]}
    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selection.json"
    selection = {
        "experiment_id": plan["experiment_id"],
        "config_sha256": plan["config_sha256"],
        "video_ids": [row["video_id"] for row in dataset["videos"]],
        "scope": "selected_demonstrations" if video_ids else "full_manual_dataset",
        "limitation": "Selected demonstrations are not an independent quality estimate.",
    }
    if selection_path.exists() and read(selection_path) != selection:
        raise ValueError("Output directory belongs to a different selection")
    write(selection_path, selection)
    service = read(run_dir / "service_config.json")
    if service["host"] != "127.0.0.1":
        raise ValueError("Evaluation must remain on loopback")
    base = "http://127.0.0.1:" + str(service["api_port"])
    # Credentials are never printed, copied to reports or sent to external hosts.
    key = (ROOT / service["api_key_file"]).read_text(encoding="utf-8").strip()
    auth = {"X-API-Key": key}

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def get(path, *, authenticated=False):
        request = urllib.request.Request(base + path, headers=auth if authenticated else {})
        with opener.open(request, timeout=15) as response:
            return json.load(response)

    def check_runtime():
        active = read(ROOT / service["runtime_root"] / "active_service.json")
        metadata = read(Path(active["runtime_dir"]) / "service_metadata.json")
        if metadata["experiment_config_sha256"] != plan["config_sha256"]:
            raise ValueError("Running service is not the frozen evaluated configuration")
        public = get("/api/v1/rules")
        if [rule["rule_id"] for rule in public["rules"]] != dataset["rule_ids"]:
            raise ValueError("Public rule set differs from the evaluation plan")
        until = time.monotonic() + 90
        while get("/healthz").get("model_ready") is not True:
            if time.monotonic() >= until:
                raise RuntimeError("Model unavailable")
            time.sleep(2)

    results_dir = output_dir / "api_results"
    results_dir.mkdir(exist_ok=True)
    outcomes_path = output_dir / "outcomes.json"
    outcomes = read(outcomes_path) if outcomes_path.exists() else []
    completed_ids = {item["video_id"] for item in outcomes}
    if len(completed_ids) != len(outcomes):
        raise ValueError("Duplicate outcomes")
    started = time.monotonic()
    for row in dataset["videos"]:
        if row["video_id"] in completed_ids:
            continue
        check_runtime()
        source = ROOT / row["media"]
        if sha(source) != row["media_sha256"]:
            raise ValueError("Evaluation video changed")
        staged = output_dir / "neutral_upload.mov"
        remux(source, staged)
        boundary = "ForteManualType" + uuid.uuid4().hex
        body = (
            (
                "--" + boundary + '\r\nContent-Disposition: form-data; name="file"; '
                'filename="upload.mov"\r\nContent-Type: video/quicktime\r\n\r\n'
            ).encode()
            + staged.read_bytes()
            + ("\r\n--" + boundary + "--\r\n").encode()
        )
        staged.unlink()
        before = time.monotonic()
        request = urllib.request.Request(
            base + "/api/v1/analyze",
            data=body,
            headers={**auth, "Content-Type": "multipart/form-data; boundary=" + boundary},
        )
        with opener.open(request, timeout=30) as response:
            job_id = json.load(response)["job_id"]
        del body, request
        while time.monotonic() - before < 320:
            response = get("/api/v1/jobs/" + job_id, authenticated=True)
            if response["status"] != "processing":
                break
            time.sleep(1)
        else:
            raise RuntimeError("API did not return a terminal status within 320 seconds")
        prefix = results_dir / row["project_key"]
        write(prefix.with_suffix(".result.json"), response)
        outcome = {
            "video_id": row["video_id"],
            "project_key": row["project_key"],
            "job_id": job_id,
            "status": response["status"],
            "elapsed_sec": round(time.monotonic() - before, 3),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if response["status"] == "done":
            outcome.update(
                verdict=response["verdict"],
                violations=[
                    {k: event[k] for k in ("type", "timecode_sec", "description")}
                    for event in response["violations"]
                ],
            )
            explained = get("/api/v1/jobs/" + job_id + "/explanation", authenticated=True)
            write(prefix.with_suffix(".explanation.json"), explained)
            outcome["xai_status"] = (explained.get("explanation") or {}).get("status")
        else:
            outcome["error"] = response.get("error")
        if sha(source) != row["media_sha256"]:
            raise ValueError("Source video changed during evaluation")
        outcomes.append(outcome)
        write(outcomes_path, outcomes)
        metrics = evaluate_types(dataset, outcomes)
        write(output_dir / "metrics.json", metrics)
        print(
            json.dumps(
                {
                    "completed": len(outcomes),
                    "total": len(dataset["videos"]),
                    "video_id": row["video_id"],
                    "status": outcome["status"],
                    "predicted_types": sorted({e["type"] for e in outcome.get("violations", [])}),
                    "elapsed_sec": outcome["elapsed_sec"],
                }
            ),
            flush=True,
        )
    check_runtime()
    final = {
        "status": "FINISHED",
        "experiment_id": plan["experiment_id"],
        "scope": selection["scope"],
        "video_ids": selection["video_ids"],
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_inference_sec": sum(item["elapsed_sec"] for item in outcomes),
        "this_invocation_sec": round(time.monotonic() - started, 3),
        "metrics": evaluate_types(dataset, outcomes),
    }
    write(output_dir / "run_summary.json", final)
    print(json.dumps({k: v for k, v in final.items() if k != "metrics"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--videos", nargs="+", help="Explicit subset, e.g. X-20 X-33")
    parser.add_argument("--output-dir", type=Path, help="Separate directory for subset results")
    arguments = parser.parse_args()
    run(
        arguments.run.resolve(),
        video_ids=arguments.videos,
        output_dir=arguments.output_dir.resolve() if arguments.output_dir else None,
    )

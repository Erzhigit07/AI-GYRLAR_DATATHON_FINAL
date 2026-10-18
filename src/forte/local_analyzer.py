"""Synchronous Qwen adapter for the API's already-isolated analysis process.

The service launcher owns the persistent loopback model server. This adapter
never creates a subprocess and never returns model-supplied evidence images.
Private worker checkpoints live only inside the API job's temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from src.forte.contracts import validate_candidate
from src.forte.explanations import attach_trace

MAX_SETTINGS_BYTES = 256 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_WORKER_RESULT_BYTES = 16 * 1024 * 1024
_GENERATION_RANGES = {
    "max_tokens": (1, 8192, True),
    "temperature": (0, 2, False),
    "seed": (-1, 2**32 - 1, True),
    "top_p": (0.000001, 1, False),
    "top_k": (0, 1000, True),
    "min_p": (0, 1, False),
    "repeat_penalty": (0.000001, 4, False),
    "presence_penalty": (-2, 2, False),
    "frequency_penalty": (-2, 2, False),
}
_TRANSPORT_ERRORS = {
    "TimeoutError",
    "URLError",
    "HTTPError",
    "ConnectionError",
    "ConnectionResetError",
    "BrokenPipeError",
    "RuntimeError",
    "RemoteDisconnected",
    "IncompleteRead",
}


def _number(value: object, field: str, low: float, high: float, integer: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid numeric setting: {field}")
    if integer and not isinstance(value, int):
        raise ValueError(f"Invalid integer setting: {field}")
    try:
        valid = math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"Setting outside allowed range: {field}")
    return value


def _text(value: object, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"Invalid text setting: {field}")
    return value


def _absolute_path(value: object, field: str, *, directory: bool = False) -> Path:
    candidate = Path(_text(value, field, 4096))
    if not candidate.is_absolute():
        raise ValueError(f"An absolute path is required: {field}")
    path = candidate.resolve(strict=True)
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Unexpected path type: {field}")
    return path


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant")


def _read_json(path: Path, limit: int):
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("JSON file exceeds the size limit")
    return json.loads(
        payload.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant
    )


def load_settings(path: str | Path) -> dict[str, Any]:
    """Validate operator-owned runtime settings, without reading competition data."""
    settings_path = _absolute_path(str(path), "settings")
    source = _read_json(settings_path, MAX_SETTINGS_BYTES)
    required = {
        "sample_step_sec",
        "frames_per_window",
        "overlap_frames",
        "overview_frames",
        "image_max_side",
        "jpeg_quality",
        "max_sampled_frames",
        "generation",
        "port",
        "prompt",
        "model_id",
        "rule_ids",
        "request_timeout_sec",
        "evidence_reserve_sec",
        "runner_path",
        "runner_sha256",
        "parser_path",
        "parser_sha256",
    }
    if not isinstance(source, dict) or not required <= set(source):
        raise ValueError("Required local analyzer settings are missing")
    if set(source) - required - {"control_dir"}:
        raise ValueError("Unknown local analyzer settings")
    ranges = {
        "sample_step_sec": (0.1, 60, False),
        "frames_per_window": (1, 64, True),
        "overlap_frames": (0, 63, True),
        "overview_frames": (1, 64, True),
        "image_max_side": (32, 1536, True),
        "jpeg_quality": (1, 100, True),
        "max_sampled_frames": (1, 4096, True),
        "port": (1, 65535, True),
        "request_timeout_sec": (0.1, 300, False),
        "evidence_reserve_sec": (5, 120, False),
    }
    for field, bounds in ranges.items():
        _number(source[field], field, *bounds)
    if source["overlap_frames"] >= source["frames_per_window"]:
        raise ValueError("overlap_frames must be smaller than frames_per_window")
    _text(source["prompt"], "prompt", 65536)
    _text(source["model_id"], "model_id", 512)
    rules = source["rule_ids"]
    if not isinstance(rules, list) or not 1 <= len(rules) <= 128:
        raise ValueError("Expected a bounded list of rule identifiers")
    for rule in rules:
        _text(rule, "rule_id", 120)
    if len(set(rules)) != len(rules):
        raise ValueError("Duplicate rule identifiers")
    generation = source["generation"]
    if not isinstance(generation, dict) or "max_tokens" not in generation:
        raise ValueError("Bounded generation max_tokens is required")
    if set(generation) - set(_GENERATION_RANGES):
        raise ValueError("Unsupported generation settings")
    for field, value in generation.items():
        _number(value, field, *_GENERATION_RANGES[field])
    for prefix in ("runner", "parser"):
        source[f"{prefix}_path"] = str(_absolute_path(source[f"{prefix}_path"], f"{prefix}_path"))
        digest = source[f"{prefix}_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("Expected a lowercase SHA-256 digest")
    if "control_dir" in source:
        source["control_dir"] = str(
            _absolute_path(source["control_dir"], "control_dir", directory=True)
        )
    return source


def _verified_source(path: str, digest: str) -> bytes:
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Frozen analyzer source integrity check failed")
    return data


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _local_request(port, endpoint, *, body=None, timeout=10):
    """Keep frames and the model token on loopback, including HTTP redirects."""
    _number(port, "port", 1, 65535, True)
    _number(timeout, "request timeout", 0.000001, 300)
    if endpoint not in {"/health", "/v1/models", "/v1/chat/completions"}:
        raise ValueError("Unexpected local model endpoint")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        data=None if body is None else json.dumps(body, allow_nan=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["FORTE_LOCAL_TOKEN"],
        },
    )
    with opener.open(request, timeout=timeout) as response:
        payload = response.read(MAX_WORKER_RESULT_BYTES + 1)
    if len(payload) > MAX_WORKER_RESULT_BYTES:
        raise ValueError("Local model response exceeds the size limit")
    return json.loads(
        payload.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant
    )


def _load_runner(settings: dict[str, Any]) -> ModuleType:
    """Execute verified bytes; avoid stale bytecode and a mutable parser dependency."""
    runner_bytes = _verified_source(settings["runner_path"], settings["runner_sha256"])
    parser_bytes = _verified_source(settings["parser_path"], settings["parser_sha256"])
    parser = ModuleType("_forte_service_frozen_parser")
    parser.__file__ = settings["parser_path"]
    exec(compile(parser_bytes, parser.__file__, "exec"), parser.__dict__)
    runner = ModuleType("_forte_service_frozen_runner")
    runner.__file__ = settings["runner_path"]
    original_path = sys.path[:]
    canonical_parser = "src.forte.qwen_baseline"
    missing = object()
    original_parser = sys.modules.get(canonical_parser, missing)
    try:
        # The runner imports this canonical name while its source executes.
        # Use verified parser bytes even if another chat is editing the module.
        sys.modules[canonical_parser] = parser
        exec(compile(runner_bytes, runner.__file__, "exec"), runner.__dict__)
    finally:
        if original_parser is missing:
            sys.modules.pop(canonical_parser, None)
        else:
            sys.modules[canonical_parser] = original_parser
        # The historical runner inserts its own repo path; snapshots need no
        # permanent import-path changes in this API process.
        sys.path[:] = original_path
    for name in ("aggregate_windows", "parse_window_response", "to_candidate"):
        setattr(runner, name, getattr(parser, name))
    runner.request = _local_request
    return runner


def _write_marker(path: Path, deadline: float, *, restart_required: bool) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(
            json.dumps(
                {"pid": os.getpid(), "deadline": deadline, "restart_required": restart_required}
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_success(control: Path) -> None:
    destination = control / "last_success.json"
    temporary = destination.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps({"completed_at": time.monotonic()}), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def analyze(video_path: str | Path, *, deadline: float) -> dict[str, Any]:
    """Analyze one private upload inside the API child; reserve time for evidence.

    The deadline is absolute time.monotonic(), including upload and queue time.
    Errors propagate to the API's existing safe, machine-readable error mapping.
    """
    _number(deadline, "deadline", 0, float("inf"))
    if deadline <= time.monotonic():
        raise TimeoutError("Analysis deadline has expired")
    token = os.environ.get("FORTE_LOCAL_TOKEN", "")
    if not token or any(character.isspace() for character in token):
        raise ValueError("Configure the local model authentication token")
    settings_path = os.environ.get("FORTE_SERVICE_SETTINGS", "")
    if not settings_path:
        raise ValueError("Configure the local analyzer settings path")
    settings = load_settings(settings_path)
    video = Path(video_path).resolve(strict=True)
    if not video.is_file():
        raise ValueError("A private video upload is required")
    analysis_deadline = deadline - settings["evidence_reserve_sec"]
    if analysis_deadline <= time.monotonic():
        raise TimeoutError("No model budget remains before evidence extraction")
    runner = _load_runner(settings)
    marker = None
    keep_marker = False
    if "control_dir" in settings:
        control = Path(settings["control_dir"])
        while not (control / "ready").is_file():
            remaining = analysis_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Local model did not become ready in time")
            time.sleep(min(0.05, remaining))
        marker = control / f"active-{uuid.uuid4().hex}.json"
    with tempfile.TemporaryDirectory(prefix="qwen-", dir=video.parent) as folder:
        temporary = Path(folder)
        worker_settings = temporary / "settings.json"
        result_path = temporary / "result.json"
        private_keys = {
            "runner_path",
            "runner_sha256",
            "parser_path",
            "parser_sha256",
            "evidence_reserve_sec",
            "control_dir",
        }
        worker_settings.write_text(
            json.dumps(
                {key: value for key, value in settings.items() if key not in private_keys},
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        worker_entered = worker_returned = False
        try:
            if marker is not None:
                _write_marker(marker, analysis_deadline, restart_required=False)
            budget = analysis_deadline - time.monotonic()
            if budget <= 0:
                raise TimeoutError("No model budget remains")
            worker_entered = True
            code = runner.run_worker(
                SimpleNamespace(
                    worker_input=str(video),
                    worker_settings=worker_settings,
                    worker_result=result_path,
                    worker_budget=budget,
                )
            )
            worker_returned = True
            result = _read_json(result_path, MAX_WORKER_RESULT_BYTES)
            if not isinstance(result, dict):
                raise RuntimeError("Invalid analyzer worker result")
            status = result.get("status")
            if status == "timeout" or time.monotonic() >= analysis_deadline:
                keep_marker = True
                if marker is not None:
                    _write_marker(marker, analysis_deadline, restart_required=True)
                raise TimeoutError("Model analysis deadline exceeded")
            if code != 0 or status != "finished":
                keep_marker = result.get("error_type") in _TRANSPORT_ERRORS
                if keep_marker and marker is not None:
                    _write_marker(marker, analysis_deadline, restart_required=True)
                raise RuntimeError("Local model analysis failed")
            candidate = result.get("candidate")
            validated = validate_candidate(candidate)
            if marker is not None:
                _write_success(marker.parent)
            return attach_trace(
                {**validated, "analysis_complete": True, "uncertain": False},
                result,
                settings["rule_ids"],
            )
        except BaseException:
            if worker_entered and not worker_returned:
                keep_marker = True
                if marker is not None:
                    _write_marker(marker, analysis_deadline, restart_required=True)
            # A hard-killed child cannot execute finally. Its marker expires
            # independently and tells the launcher to reset its model server.
            raise
        finally:
            if marker is not None and not keep_marker:
                marker.unlink(missing_ok=True)

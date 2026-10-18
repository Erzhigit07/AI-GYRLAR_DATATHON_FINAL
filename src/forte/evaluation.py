"""Private offline evaluation using an existing split; never infer labels from prose.

The analyzer receives only a neutral local filename and a monotonic deadline.
The callable must cooperate with that deadline (or wrap an isolated worker): this
in-process evaluator cannot interrupt a hung callback. Late results are rejected.
All reports stay local, and an existing output directory is never overwritten.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from src.forte.contracts import ContractError
from src.forte.evidence import EvidenceError, build_result


class InputChangedError(RuntimeError):
    """Input bytes changed during a run; saved scores must not be used."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _prepare_records(manifest: Mapping[str, Any], base_dir: Path):
    records = manifest.get("videos")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest.videos must be a nonempty list using the existing split")
    root = base_dir.resolve(strict=True)
    prepared = []
    keys = set()
    groups: dict[tuple[str, str], tuple[str, int]] = {}
    related_count = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Each manifest record must be a mapping")
        key, split = record.get("key"), record.get("split")
        if not isinstance(key, str) or not key.strip() or key in keys:
            raise ValueError("Manifest keys must be nonempty and unique")
        keys.add(key)
        if split not in {"development", "holdout"}:
            raise ValueError("Every record must retain its development/holdout split")
        annotations = record.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            raise ValueError(f"Missing binary annotations for {key}")
        labels = [item.get("class") if isinstance(item, Mapping) else None for item in annotations]
        if any(type(label) is not int or label not in (0, 1) for label in labels):
            raise ValueError(f"Annotations must have integer class 0 or 1 for {key}")
        if len(set(labels)) != 1:
            raise ValueError(f"Conflicting labels among aliases for {key}")
        label = labels[0]
        filename = record.get("file")
        if not isinstance(filename, str) or Path(filename).is_absolute():
            raise ValueError("Manifest file paths must be relative to base_dir")
        path = (root / filename).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Manifest input must be a regular file inside base_dir")
        actual_sha = _sha256(path)
        source_sha = record.get("source_sha256")
        if (
            not isinstance(source_sha, str)
            or len(source_sha) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in source_sha)
        ):
            raise ValueError(f"A source SHA256 is required for {key}")
        source_sha = source_sha.lower()
        expected_file_sha = record.get("file_sha256")
        if expected_file_sha is not None and expected_file_sha != actual_sha:
            raise ValueError(f"Local file SHA256 differs from manifest for {key}")
        identities = [("source_sha256", source_sha), ("file_sha256", actual_sha)]
        related = record.get("related_group")
        if related is not None:
            if not isinstance(related, str) or not related.strip():
                raise ValueError("related_group must be a nonempty string when supplied")
            related_count += 1
            identities.append(("related_group", related))
        for identity in identities:
            previous = groups.get(identity)
            if previous is not None:
                if previous[0] != split:
                    raise ValueError(f"Cross-split leakage detected by {identity[0]}")
                if identity[0] != "related_group" and previous[1] != label:
                    raise ValueError("Exact duplicate records have conflicting binary labels")
            groups[identity] = (split, label)
        prepared.append(
            {"key": key, "split": split, "path": path, "label": label, "sha256": actual_sha}
        )
    return prepared, {
        "exact_duplicate_split_check": "PASS",
        "related_group_split_check": "PASS" if related_count == len(prepared) else "NOT VERIFIED",
        "related_group_records": related_count,
        "total_records": len(prepared),
        "limitation": "Supplied group assignments checked; semantic relatedness is not inferred.",
    }


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "evaluation_sha256": _sha256(Path(__file__))}
    for package in ("av", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT INSTALLED"
    return versions


def _metrics(outcomes: list[dict[str, Any]], selected_count: int) -> dict[str, Any]:
    counts = Counter({"TP": 0, "TN": 0, "FP": 0, "FN": 0})
    completed = [item for item in outcomes if item["status"] == "complete"]
    for item in completed:
        positive = item["prediction"] == 1
        actual_positive = item["label"] == 1
        counts[("T" if positive == actual_positive else "F") + ("P" if positive else "N")] += 1
    tp, tn, fp, fn = (counts[name] for name in ("TP", "TN", "FP", "FN"))

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "selected": selected_count,
        "attempted": len(outcomes),
        "not_attempted": selected_count - len(outcomes),
        "complete": len(completed),
        "status_counts": dict(Counter(item["status"] for item in outcomes)),
        "attempted_by_class": dict(Counter(str(item["label"]) for item in outcomes)),
        "complete_by_class": dict(Counter(str(item["label"]) for item in completed)),
        "coverage_per_attempted": ratio(len(completed), len(outcomes)),
        "coverage_per_selected": ratio(len(completed), selected_count),
        "confusion_complete_only": dict(counts),
        "accuracy_complete_only": ratio(tp + tn, len(completed)),
        "precision_complete_only": ratio(tp, tp + fp),
        "recall_complete_only": ratio(tp, tp + fn),
        "f1_complete_only": ratio(2 * tp, 2 * tp + fp + fn),
        "correct_per_attempted": ratio(tp + tn, len(outcomes)),
        "metric_scope": "Internal binary diagnostics; no official temporal tolerance assumed.",
    }


def evaluate_manifest(
    manifest: Mapping[str, Any],
    analyzer: Callable[..., Mapping[str, Any]],
    output_dir: str | Path,
    *,
    base_dir: str | Path,
    run_metadata: Mapping[str, Any],
    split: str = "development",
    allow_holdout: bool = False,
    timeout_sec: float = 300.0,
) -> dict[str, Any]:
    """Run an existing split and save private results in a new local directory.

    ``analyzer(str(neutral_video), deadline=monotonic_deadline)`` returns the
    candidate contract consumed by ``build_result``. It must not use labels or
    hidden state from the caller. Holdout needs an explicit opt-in and must not
    be used to choose prompts, parameters, or models.

    ``run_metadata`` requires analyzer_version, code_revision and settings. It
    should identify the exact model/prompt configuration, including dirty code
    when relevant. Archive-member SHA and remuxed-file SHA are deliberately distinct.
    """
    run_started = monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    if not isinstance(allow_holdout, bool):
        raise ValueError("allow_holdout must be an explicit boolean")
    if split not in {"development", "holdout"} or (split == "holdout" and not allow_holdout):
        raise ValueError("Holdout requires split='holdout' and allow_holdout=True")
    if (
        isinstance(timeout_sec, bool)
        or not isinstance(timeout_sec, (int, float))
        or not math.isfinite(timeout_sec)
        or timeout_sec <= 0
    ):
        raise ValueError("timeout_sec must be positive and finite")
    if not isinstance(run_metadata, Mapping) or any(
        not isinstance(run_metadata.get(name), str) or not run_metadata[name].strip()
        for name in ("analyzer_version", "code_revision")
    ):
        raise ValueError("run_metadata must identify analyzer_version and code_revision")
    if not isinstance(run_metadata.get("settings"), Mapping):
        raise ValueError("run_metadata must include explicit settings")
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    metadata_snapshot = json.loads(_json_bytes(run_metadata))
    manifest_bytes = _json_bytes(manifest)
    manifest_snapshot = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    runtime_versions = _runtime_versions()
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing evaluation directory")
    prepared, audit = _prepare_records(manifest_snapshot, Path(base_dir))
    selected = [record for record in prepared if record["split"] == split]
    if not selected:
        raise ValueError("The requested existing split has no records")
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "manifest_snapshot.json", manifest_snapshot)
    outcomes = []
    with (destination / "outcomes.jsonl").open("x", encoding="utf-8") as report_stream:
        for record in selected:
            started = monotonic()
            deadline = started + timeout_sec
            outcome = {"key": record["key"], "label": record["label"], "status": "error"}
            try:
                with tempfile.TemporaryDirectory(prefix="private-eval-", dir=destination) as work:
                    neutral = Path(work) / "video.mov"
                    shutil.copyfile(record["path"], neutral)
                    if _sha256(neutral) != record["sha256"]:
                        raise InputChangedError("Input changed after preflight hashing")
                    if monotonic() >= deadline:
                        raise TimeoutError("Deadline reached while preparing neutral input")
                    candidate = analyzer(str(neutral), deadline=deadline)
                    if monotonic() >= deadline:
                        raise TimeoutError("Analyzer exceeded the end-to-end deadline")
                    if isinstance(candidate, Mapping) and (
                        candidate.get("analysis_complete") is False
                        or candidate.get("uncertain") is True
                        or candidate.get("verdict") in ("unknown", "uncertain")
                    ):
                        outcome["status"] = "incomplete"
                    else:
                        result = build_result(str(neutral), candidate, deadline=deadline)
                        outcome.update(
                            status="complete",
                            prediction=int(result["verdict"] == "violation"),
                            result=result,
                        )
                    if _sha256(neutral) != record["sha256"]:
                        raise EvidenceError("Analyzer modified its neutral video copy")
                    if monotonic() >= deadline:
                        raise TimeoutError("Validation exceeded the end-to-end deadline")
            except TimeoutError as error:
                outcome.update(status="timeout", error_type=type(error).__name__, error=str(error))
            except (ContractError, EvidenceError) as error:
                outcome.update(
                    status="invalid_result", error_type=type(error).__name__, error=str(error)
                )
            except Exception as error:
                outcome.update(status="error", error_type=type(error).__name__, error=str(error))
            try:
                unchanged = _sha256(record["path"]) == record["sha256"]
            except OSError:
                unchanged = False
            if not unchanged:
                outcome.update(status="input_changed", error="Source changed during evaluation")
            finished = monotonic()
            if (
                unchanged
                and outcome["status"] in {"complete", "incomplete"}
                and finished >= deadline
            ):
                outcome.update(status="timeout", error="End-to-end deadline exceeded")
            if outcome["status"] != "complete":
                outcome.pop("prediction", None)
                outcome.pop("result", None)
            outcome["elapsed_sec"] = finished - started
            report_stream.write(_json_bytes(outcome).decode("utf-8") + "\n")
            report_stream.flush()
            outcomes.append({key: value for key, value in outcome.items() if key != "result"})
            if not unchanged:
                break
    integrity = []
    for record in prepared:
        try:
            after = _sha256(record["path"])
        except OSError:
            after = None
        integrity.append(
            {
                "key": record["key"],
                "before": record["sha256"],
                "after": after,
                "unchanged": record["sha256"] == after,
            }
        )
    unchanged = all(item["unchanged"] for item in integrity)
    summary = {
        "status": "FINISHED" if unchanged else "INVALID_INPUT_MUTATION",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "holdout_explicitly_allowed": allow_holdout,
        "manifest_sha256": manifest_sha,
        "run_metadata": metadata_snapshot,
        "runtime": runtime_versions,
        "split_audit": audit,
        "input_integrity": integrity,
        "metrics": _metrics(outcomes, len(selected)) if unchanged else None,
        "timeout_sec": timeout_sec,
        "timeout_enforcement": (
            "Cooperative; late results rejected, hung callback not interrupted."
        ),
        "timing_scope": "Neutral copy, analyzer, evidence validation, input checks and cleanup.",
        "elapsed_sec": sum(item["elapsed_sec"] for item in outcomes),
        "run_wall_sec": monotonic() - run_started,
        "output_privacy": (
            "Local only. Outcomes may contain private descriptions and JPEG evidence."
        ),
    }
    _write_json(destination / "summary.json", summary)
    if not unchanged:
        raise InputChangedError("Input integrity failed; saved evaluation metrics are invalid")
    return summary

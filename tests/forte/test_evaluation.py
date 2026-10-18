"""Only synthetic fixtures; no competition videos, labels, or model calls."""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("av")
pytest.importorskip("PIL")

from src.forte import evaluation  # noqa: E402
from src.forte.contracts import ContractError  # noqa: E402

METADATA = {
    "analyzer_version": "synthetic-test-adapter-v1",
    "code_revision": "synthetic-test-revision",
    "settings": {"synthetic_fixture": True},
}


def make_manifest(tmp_path, labels=(1, 0, 0, 1), splits=None):
    """Byte fixtures are deliberately not videos; unit tests replace the decoder."""
    records = []
    for index, label in enumerate(labels):
        content = f"SYNTHETIC UNIT FIXTURE {index}".encode()
        name = f"original_label_{label}_{index}.mov"
        (tmp_path / name).write_bytes(content)
        records.append(
            {
                "key": f"synthetic_{index}",
                "file": name,
                "source_sha256": hashlib.sha256(content).hexdigest(),
                "split": splits[index] if splits else "development",
                "annotations": [{"class": label}],
            }
        )
    return {"videos": records}


def candidate(positive=False):
    return {
        "verdict": "violation" if positive else "no_violation",
        "violations": (
            [{"type": "synthetic_event", "timecode_sec": 0.0, "description": "Synthetic fixture"}]
            if positive
            else []
        ),
        "analysis_complete": True,
        "uncertain": False,
    }


@pytest.fixture
def fake_builder(monkeypatch):
    def build(path, result, *, deadline):
        if not isinstance(result, dict):
            raise ContractError("Synthetic invalid result")
        return {"verdict": result["verdict"], "violations": result["violations"]}

    monkeypatch.setattr(evaluation, "build_result", build)


def run(tmp_path, manifest, analyzer, *, output="run", **kwargs):
    return evaluation.evaluate_manifest(
        manifest,
        analyzer,
        tmp_path / output,
        base_dir=tmp_path,
        run_metadata=METADATA,
        **kwargs,
    )


def test_existing_development_split_neutral_input_and_confusion(tmp_path, fake_builder):
    manifest = make_manifest(
        tmp_path, labels=(1, 0, 0, 1, 1), splits=["development"] * 4 + ["holdout"]
    )
    predictions = iter((True, False, True, False))
    seen_paths = []

    def analyze(path, *, deadline):
        neutral = Path(path)
        assert neutral.name == "video.mov"
        assert "original_label" not in path
        assert neutral.exists()
        assert deadline > evaluation.monotonic()
        seen_paths.append(neutral)
        return candidate(next(predictions))

    report = run(tmp_path, manifest, analyze)
    metrics = report["metrics"]
    assert metrics["attempted"] == metrics["selected"] == 4
    assert metrics["confusion_complete_only"] == {"TP": 1, "TN": 1, "FP": 1, "FN": 1}
    assert metrics["accuracy_complete_only"] == metrics["f1_complete_only"] == 0.5
    assert metrics["coverage_per_attempted"] == 1
    assert report["split_audit"]["related_group_split_check"] == "NOT VERIFIED"
    assert all(item["unchanged"] for item in report["input_integrity"])
    assert all(not path.exists() for path in seen_paths)
    assert report["run_metadata"] == METADATA
    assert (tmp_path / "run" / "summary.json").is_file()
    assert len((tmp_path / "run" / "outcomes.jsonl").read_text().splitlines()) == 4
    assert report["manifest_sha256"] == hashlib.sha256(evaluation._json_bytes(manifest)).hexdigest()


def test_errors_incomplete_timeout_and_invalid_are_not_negative_predictions(tmp_path, fake_builder):
    manifest = make_manifest(tmp_path, labels=(0, 0, 0, 0, 0))
    attempts = iter(("error", "incomplete", "timeout", "invalid", "complete"))

    def analyze(path, *, deadline):
        kind = next(attempts)
        if kind == "error":
            raise RuntimeError("Synthetic analyzer failure")
        if kind == "incomplete":
            return {**candidate(), "analysis_complete": False}
        if kind == "timeout":
            raise TimeoutError("Synthetic timeout")
        if kind == "invalid":
            return None
        return candidate()

    report = run(tmp_path, manifest, analyze)
    metrics = report["metrics"]
    assert metrics["attempted"] == 5
    assert metrics["complete"] == 1
    assert metrics["coverage_per_attempted"] == 0.2
    assert metrics["accuracy_complete_only"] == 1
    assert metrics["correct_per_attempted"] == 0.2
    assert metrics["status_counts"] == dict.fromkeys(
        ("error", "incomplete", "timeout", "invalid_result", "complete"), 1
    )
    assert metrics["confusion_complete_only"] == {"TP": 0, "TN": 1, "FP": 0, "FN": 0}
    assert metrics["recall_complete_only"] is None


def test_holdout_needs_explicit_opt_in(tmp_path, fake_builder):
    manifest = make_manifest(tmp_path, labels=(0,), splits=["holdout"])
    with pytest.raises(ValueError, match="Holdout requires"):
        run(tmp_path, manifest, lambda path, **kwargs: candidate(), split="holdout")
    with pytest.raises(ValueError, match="explicit boolean"):
        run(
            tmp_path,
            manifest,
            lambda path, **kwargs: candidate(),
            split="holdout",
            allow_holdout="false",
        )
    assert not (tmp_path / "run").exists()
    result = run(
        tmp_path, manifest, lambda path, **kwargs: candidate(), split="holdout", allow_holdout=True
    )
    assert result["metrics"]["attempted"] == 1
    assert result["holdout_explicitly_allowed"] is True


@pytest.mark.parametrize("group", ["source_sha256", "actual_file", "related_group"])
def test_cross_split_duplicate_groups_fail_closed(tmp_path, group):
    manifest = make_manifest(tmp_path, labels=(0, 0), splits=["development", "holdout"])
    first, second = manifest["videos"]
    if group == "actual_file":
        (tmp_path / second["file"]).write_bytes((tmp_path / first["file"]).read_bytes())
    elif group == "related_group":
        first[group] = second[group] = "explicit-synthetic-scene"
    else:
        second[group] = first[group]
    with pytest.raises(ValueError, match="Cross-split leakage"):
        run(tmp_path, manifest, lambda path, **kwargs: candidate())
    assert not (tmp_path / "run").exists()


def test_conflicting_alias_labels_rejected_before_call(tmp_path):
    manifest = make_manifest(tmp_path, labels=(0,))
    manifest["videos"][0]["annotations"].append({"class": 1})
    with pytest.raises(ValueError, match="Conflicting labels"):
        run(tmp_path, manifest, lambda path, **kwargs: candidate())


def test_conflicting_labels_on_exact_duplicates_rejected(tmp_path):
    manifest = make_manifest(tmp_path, labels=(0, 1))
    manifest["videos"][1]["source_sha256"] = manifest["videos"][0]["source_sha256"]
    with pytest.raises(ValueError, match="conflicting binary labels"):
        run(tmp_path, manifest, lambda path, **kwargs: candidate())


@pytest.mark.parametrize("mapping_complete", [False, True])
def test_related_group_coverage_reported_honestly(tmp_path, fake_builder, mapping_complete):
    manifest = make_manifest(tmp_path, labels=(0, 0))
    manifest["videos"][0]["related_group"] = "synthetic-scene-0"
    if mapping_complete:
        manifest["videos"][1]["related_group"] = "synthetic-scene-1"
    result = run(tmp_path, manifest, lambda path, **kwargs: candidate())
    assert result["split_audit"]["related_group_split_check"] == (
        "PASS" if mapping_complete else "NOT VERIFIED"
    )


def test_existing_output_never_overwritten(tmp_path, fake_builder):
    manifest = make_manifest(tmp_path, labels=(0,))
    run(tmp_path, manifest, lambda path, **kwargs: candidate())
    old = (tmp_path / "run" / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(tmp_path, manifest, lambda path, **kwargs: candidate())
    assert (tmp_path / "run" / "summary.json").read_bytes() == old


def test_source_mutation_invalidates_run_and_preserves_evidence(tmp_path, fake_builder):
    manifest = make_manifest(tmp_path, labels=(0, 0))
    original = tmp_path / manifest["videos"][0]["file"]

    def deliberately_bad_analyzer(path, *, deadline):
        original.write_bytes(b"SYNTHETIC deliberate mutation test")
        return candidate()

    with pytest.raises(evaluation.InputChangedError):
        run(tmp_path, manifest, deliberately_bad_analyzer)
    report = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert report["status"] == "INVALID_INPUT_MUTATION"
    assert report["metrics"] is None
    assert report["input_integrity"][0]["unchanged"] is False
    outcome_lines = (tmp_path / "run" / "outcomes.jsonl").read_text().splitlines()
    assert len(outcome_lines) == 1
    assert json.loads(outcome_lines[0])["status"] == "input_changed"


def test_mutating_neutral_copy_is_rejected_without_source_changes(tmp_path, fake_builder):
    manifest = make_manifest(tmp_path, labels=(0,))

    def analyze(path, *, deadline):
        Path(path).write_bytes(b"SYNTHETIC tampered copy")
        return candidate()

    report = run(tmp_path, manifest, analyze)
    assert report["metrics"]["status_counts"] == {"invalid_result": 1}
    assert report["metrics"]["accuracy_complete_only"] is None
    assert all(item["unchanged"] for item in report["input_integrity"])


def test_late_result_rejected_and_time_includes_copy(tmp_path, fake_builder, monkeypatch):
    manifest = make_manifest(tmp_path, labels=(0,))
    clock = [10.0]
    monkeypatch.setattr(evaluation, "monotonic", lambda: clock[0])
    real_copy = evaluation.shutil.copyfile

    def copy(*args, **kwargs):
        result = real_copy(*args, **kwargs)
        clock[0] += 2.0
        return result

    def analyze(path, *, deadline):
        assert deadline == 15.0
        clock[0] += 4.0
        return candidate()

    monkeypatch.setattr(evaluation.shutil, "copyfile", copy)
    report = run(tmp_path, manifest, analyze, timeout_sec=5)
    assert report["metrics"]["status_counts"] == {"timeout": 1}
    assert report["elapsed_sec"] == 6.0
    assert "hung callback not interrupted" in report["timeout_enforcement"]


def test_file_sha_mismatch_and_external_path_rejected(tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    manifest = make_manifest(base, labels=(0,))
    manifest["videos"][0]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 differs"):
        run(base, manifest, lambda path, **kwargs: candidate())
    manifest["videos"][0].pop("file_sha256")
    (tmp_path / "outside.mov").write_bytes(b"SYNTHETIC external fixture")
    manifest["videos"][0]["file"] = "../outside.mov"
    with pytest.raises(ValueError, match="inside base_dir"):
        run(base, manifest, lambda path, **kwargs: candidate())


def test_deadline_includes_final_integrity_check(tmp_path, fake_builder, monkeypatch):
    manifest = make_manifest(tmp_path, labels=(0,))
    original = tmp_path / manifest["videos"][0]["file"]
    clock, original_reads = [10.0], [0]
    real_hash = evaluation._sha256
    monkeypatch.setattr(evaluation, "monotonic", lambda: clock[0])

    def slow_verification(path):
        if path == original:
            original_reads[0] += 1
            if original_reads[0] == 2:
                clock[0] += 6.0
        return real_hash(path)

    monkeypatch.setattr(evaluation, "_sha256", slow_verification)
    report = run(tmp_path, manifest, lambda path, **kwargs: candidate(), timeout_sec=5)
    assert report["metrics"]["status_counts"] == {"timeout": 1}
    assert report["elapsed_sec"] == 6


def test_real_builder_with_tiny_synthetic_movie(tmp_path):
    import av
    from PIL import Image

    path = tmp_path / "synthetic.mov"
    with av.open(str(path), "w", format="mov") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for index in range(4):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), (index * 50, 0, 0)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    manifest = {
        "videos": [
            {
                "key": "synthetic-video",
                "file": path.name,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "split": "development",
                "annotations": [{"class": 1}],
            }
        ]
    }
    result = run(tmp_path, manifest, lambda path, **kwargs: candidate(True))
    assert result["metrics"]["confusion_complete_only"]["TP"] == 1
    assert all(item["unchanged"] for item in result["input_integrity"])
    outcome = json.loads((tmp_path / "run" / "outcomes.jsonl").read_text())
    evidence = outcome["result"]["violations"][0]["evidence_frames"]
    assert 1 <= len(evidence) <= 3

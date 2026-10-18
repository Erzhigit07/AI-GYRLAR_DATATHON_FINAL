"""Local adapter checks use synthetic video/source fixtures; no model is started."""

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("PIL")

from src.forte import local_analyzer as adapter  # noqa: E402
from src.forte.contracts import ContractError  # noqa: E402

NEGATIVE = {
    "verdict": "no_violation",
    "violations": [],
    "analysis_complete": True,
    "uncertain": False,
}


@pytest.fixture
def configured(tmp_path, monkeypatch):
    parser = tmp_path / "parser.py"
    parser.write_text(
        "def aggregate_windows(*args, **kwargs): return 'frozen-parser'\n"
        "def parse_window_response(*args, **kwargs): return 'frozen-parser'\n"
        "def to_candidate(*args, **kwargs): return 'frozen-parser'\n",
        encoding="utf-8",
    )
    runner = tmp_path / "runner.py"
    runner.write_text("import sys\nsys.path.insert(0, 'synthetic-import-path')\n", encoding="utf-8")
    control = tmp_path / "control"
    control.mkdir()
    (control / "ready").touch()
    settings = {
        "sample_step_sec": 1.0,
        "frames_per_window": 16,
        "overlap_frames": 2,
        "overview_frames": 16,
        "image_max_side": 448,
        "jpeg_quality": 80,
        "max_sampled_frames": 300,
        "generation": {
            "max_tokens": 768,
            "temperature": 0,
            "seed": 2026,
            "top_p": 1.0,
            "top_k": 1,
            "repeat_penalty": 1.0,
        },
        "port": 19081,
        "prompt": "Synthetic instruction",
        "model_id": "synthetic-model",
        "rule_ids": ["synthetic-rule"],
        "request_timeout_sec": 60,
        "evidence_reserve_sec": 20,
        "control_dir": str(control),
        "runner_path": str(runner),
        "parser_path": str(parser),
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "parser_sha256": hashlib.sha256(parser.read_bytes()).hexdigest(),
    }
    settings_path = tmp_path / "runtime-settings.json"
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    video = tmp_path / "input.mov"
    video.write_bytes(b"Synthetic input; decoding is mocked in unit tests")
    monkeypatch.setenv("FORTE_SERVICE_SETTINGS", str(settings_path))
    monkeypatch.setenv("FORTE_LOCAL_TOKEN", "synthetic-local-token-only")
    return SimpleNamespace(
        settings=settings,
        path=settings_path,
        video=video,
        root=tmp_path,
        control=control,
    )


def save_settings(configured):
    configured.path.write_text(json.dumps(configured.settings), encoding="utf-8")


def fake_runner(monkeypatch, payload, *, code=0, callback=None):
    def run_worker(args):
        if callback:
            callback(args)
        args.worker_result.write_text(json.dumps(payload), encoding="utf-8")
        return code

    monkeypatch.setattr(
        adapter,
        "_load_runner",
        lambda settings: SimpleNamespace(
            run_worker=run_worker,
        ),
    )


def test_settings_validate_generation_and_absolute_snapshot_paths(configured):
    settings = adapter.load_settings(configured.path)
    assert settings == configured.settings


@pytest.mark.parametrize(
    "field,value",
    [
        ("port", True),
        ("port", 0),
        ("port", "19081"),
        ("port", 65536),
        ("sample_step_sec", 0),
        ("max_sampled_frames", 999999),
        ("overlap_frames", 16),
        ("evidence_reserve_sec", 0),
        ("rule_ids", ["duplicate", "duplicate"]),
        ("runner_path", "relative.py"),
        ("runner_sha256", "0" * 63),
        ("generation", {"max_tokens": 0}),
        ("generation", {"max_tokens": 768, "messages": []}),
        ("generation", {"max_tokens": 768, "stream": True}),
        ("generation", {"max_tokens": 768, "seed": True}),
    ],
)
def test_invalid_configuration_is_rejected(configured, field, value):
    configured.settings[field] = value
    save_settings(configured)
    with pytest.raises(ValueError):
        adapter.load_settings(configured.path)


def test_duplicate_or_nonfinite_json_is_rejected(configured):
    configured.path.write_text('{"port": 1, "port": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        adapter.load_settings(configured.path)
    configured.path.write_text('{"port": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Nonfinite"):
        adapter.load_settings(configured.path)


def test_verified_loader_uses_frozen_parser_and_restores_import_path(configured):
    original = sys.path[:]
    runner = adapter._load_runner(configured.settings)
    assert runner.aggregate_windows() == "frozen-parser"
    assert runner.parse_window_response() == "frozen-parser"
    assert runner.to_candidate() == "frozen-parser"
    assert runner.request is adapter._local_request
    assert sys.path == original
    Path(configured.settings["parser_path"]).write_text("raise RuntimeError('changed')")
    with pytest.raises(ValueError, match="integrity"):
        adapter._load_runner(configured.settings)


@pytest.mark.parametrize("previous", ["module", "absent", "none"])
@pytest.mark.parametrize("runner_fails", [False, True])
def test_frozen_parser_import_restores_registry_even_on_runner_failure(
    configured,
    monkeypatch,
    previous,
    runner_fails,
):
    canonical = "src.forte.qwen_baseline"
    original = ModuleType(canonical) if previous == "module" else None
    if previous == "absent":
        monkeypatch.delitem(sys.modules, canonical, raising=False)
    else:
        # This intentionally has none of the parser functions. Importing it
        # directly would fail instead of running the verified snapshot.
        monkeypatch.setitem(sys.modules, canonical, original)
    code = (
        "import sys\nsys.path.insert(0, 'synthetic-runner-path')\n"
        "from src.forte.qwen_baseline import parse_window_response\n"
        "value_during_import = parse_window_response()\n"
    )
    if runner_fails:
        code += "raise RuntimeError('synthetic runner failure')\n"
    runner_path = Path(configured.settings["runner_path"])
    runner_path.write_text(code, encoding="utf-8")
    configured.settings["runner_sha256"] = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    original_path = sys.path[:]
    if runner_fails:
        with pytest.raises(RuntimeError, match="synthetic runner failure"):
            adapter._load_runner(configured.settings)
    else:
        loaded = adapter._load_runner(configured.settings)
        assert loaded.value_during_import == "frozen-parser"
    assert sys.path == original_path
    if previous == "absent":
        assert canonical not in sys.modules
    else:
        assert sys.modules[canonical] is original


def test_frozen_runner_does_not_import_broken_mutable_parser(configured, monkeypatch):
    import src.forte

    mutable = configured.root / "mutable"
    mutable.mkdir()
    (mutable / "qwen_baseline.py").write_text("def broken(", encoding="utf-8")
    monkeypatch.setattr(src.forte, "__path__", [str(mutable)])
    monkeypatch.delitem(sys.modules, "src.forte.qwen_baseline", raising=False)
    runner_path = Path(configured.settings["runner_path"])
    runner_path.write_text(
        "from src.forte.qwen_baseline import to_candidate\nvalue_during_import = to_candidate()\n",
        encoding="utf-8",
    )
    configured.settings["runner_sha256"] = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    runner = adapter._load_runner(configured.settings)
    assert runner.value_during_import == "frozen-parser"
    assert "src.forte.qwen_baseline" not in sys.modules


def test_analyzer_executes_in_same_process_reserves_evidence_and_cleans(configured, monkeypatch):
    started = time.monotonic()
    deadline = time.monotonic() + 60

    def inspect(args):
        assert 0 < args.worker_budget <= 40
        assert args.worker_input == str(configured.video)
        assert args.worker_result.parent.parent == configured.root
        settings = json.loads(args.worker_settings.read_text(encoding="utf-8"))
        assert "control_dir" not in settings and "runner_path" not in settings
        assert "synthetic-local-token-only" not in args.worker_settings.read_text()
        marker = json.loads(next(configured.control.glob("active-*.json")).read_text())
        assert marker["deadline"] == deadline - 20
        assert marker["restart_required"] is False

    fake_runner(monkeypatch, {"status": "finished", "candidate": NEGATIVE}, callback=inspect)
    assert adapter.analyze(configured.video, deadline=deadline) == NEGATIVE
    assert not list(configured.root.glob("qwen-*"))
    assert not list(configured.control.glob("active-*"))
    success = json.loads((configured.control / "last_success.json").read_text())
    assert set(success) == {"completed_at"}
    assert started <= success["completed_at"] <= time.monotonic()


def test_success_record_atomically_replaces_previous_success(configured, monkeypatch):
    destination = configured.control / "last_success.json"
    previous = json.dumps({"completed_at": 0.0})
    destination.write_text(previous, encoding="utf-8")
    replace = adapter.os.replace
    replacements = []

    def inspect_replace(source, target):
        if target == destination:
            assert destination.read_text(encoding="utf-8") == previous
            assert json.loads(source.read_text(encoding="utf-8"))["completed_at"] > 0
            replacements.append(target)
        return replace(source, target)

    monkeypatch.setattr(adapter.os, "replace", inspect_replace)
    fake_runner(monkeypatch, {"status": "finished", "candidate": NEGATIVE})
    adapter.analyze(configured.video, deadline=time.monotonic() + 60)
    assert replacements == [destination]
    assert json.loads(destination.read_text())["completed_at"] > 0
    assert not list(configured.control.glob("*.tmp"))


@pytest.mark.parametrize(
    "status,error_type,expected,retained",
    [
        ("timeout", "TimeoutError", TimeoutError, True),
        ("error", "RuntimeError", RuntimeError, True),
        ("error", "URLError", RuntimeError, True),
        ("error", "ValueError", RuntimeError, False),
    ],
)
def test_worker_failures_signal_model_reset_only_when_needed(
    configured,
    monkeypatch,
    status,
    error_type,
    expected,
    retained,
):
    fake_runner(monkeypatch, {"status": status, "error_type": error_type}, code=1)
    with pytest.raises(expected):
        adapter.analyze(configured.video, deadline=time.monotonic() + 60)
    markers = list(configured.control.glob("active-*.json"))
    assert bool(markers) is retained
    if retained:
        assert json.loads(markers[0].read_text())["restart_required"] is True
    assert not list(configured.root.glob("qwen-*"))
    assert not (configured.control / "last_success.json").exists()


def test_incomplete_candidate_is_never_converted_to_negative(configured, monkeypatch):
    success_path = configured.control / "last_success.json"
    previous = json.dumps({"completed_at": 1.0})
    success_path.write_text(previous, encoding="utf-8")
    fake_runner(
        monkeypatch,
        {
            "status": "finished",
            "candidate": {
                **NEGATIVE,
                "analysis_complete": False,
                "uncertain": True,
            },
        },
    )
    with pytest.raises(ContractError):
        adapter.analyze(configured.video, deadline=time.monotonic() + 60)
    assert not list(configured.control.glob("active-*"))
    assert success_path.read_text(encoding="utf-8") == previous


def test_unexpected_worker_exception_preserves_restart_marker(configured, monkeypatch):
    def failure(args):
        raise TimeoutError("Synthetic worker transport timeout")

    monkeypatch.setattr(
        adapter,
        "_load_runner",
        lambda settings: SimpleNamespace(
            run_worker=failure,
        ),
    )
    with pytest.raises(TimeoutError):
        adapter.analyze(configured.video, deadline=time.monotonic() + 60)
    marker = next(configured.control.glob("active-*.json"))
    assert json.loads(marker.read_text())["restart_required"] is True
    assert not list(configured.root.glob("qwen-*"))


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf"), "invalid"])
def test_invalid_deadline_is_rejected(configured, deadline):
    with pytest.raises(ValueError):
        adapter.analyze(configured.video, deadline=deadline)


def test_expired_budget_does_not_start_worker(configured, monkeypatch):
    load = Mock()
    monkeypatch.setattr(adapter, "_load_runner", load)
    with pytest.raises(TimeoutError):
        adapter.analyze(configured.video, deadline=time.monotonic() + 10)
    load.assert_not_called()


def test_wait_for_model_ready_is_bounded(configured, monkeypatch):
    (configured.control / "ready").unlink()
    fake_runner(monkeypatch, {"status": "finished", "candidate": NEGATIVE})
    before = time.monotonic()
    with pytest.raises(TimeoutError, match="ready"):
        adapter.analyze(configured.video, deadline=before + 20.1)
    assert time.monotonic() - before < 1
    assert not list(configured.control.glob("active-*"))


def test_missing_token_fails_before_loading_configuration(configured, monkeypatch):
    monkeypatch.delenv("FORTE_LOCAL_TOKEN")
    with pytest.raises(ValueError, match="token"):
        adapter.analyze(configured.video, deadline=time.monotonic() + 60)


def test_redirect_handler_refuses_remote_redirects():
    assert (
        adapter._NoRedirect().redirect_request(
            None,
            None,
            302,
            "Found",
            {},
            "https://example.invalid/collect",
        )
        is None
    )


@pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("av", "psutil")),
    reason="Optional local video runtime unavailable",
)
def test_real_frozen_worker_on_synthetic_video_without_model(configured, monkeypatch):
    import av
    from PIL import Image

    repo = Path(__file__).resolve().parents[2]
    for prefix, source in (
        ("runner", repo / "experiments/forte_qwen_baseline.py"),
        ("parser", repo / "src/forte/qwen_baseline.py"),
    ):
        data = source.read_bytes()
        Path(configured.settings[f"{prefix}_path"]).write_bytes(data)
        configured.settings[f"{prefix}_sha256"] = hashlib.sha256(data).hexdigest()
    save_settings(configured)
    with av.open(str(configured.video), "w", format="mov") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(4):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 48), (index * 60, 0, 0)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    calls = []

    def response(port, endpoint, *, body, timeout):
        calls.append(body)
        assert endpoint == "/v1/chat/completions"
        assert "input.mov" not in json.dumps(body)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "verdict": "no_violation",
                                "analysis_complete": True,
                                "events": [],
                                "uncertainty_reason": "",
                            }
                        )
                    },
                }
            ]
        }

    monkeypatch.setattr(adapter, "_local_request", response)
    # An accidental nested worker/server launch makes the test fail immediately.
    monkeypatch.setattr("subprocess.Popen", Mock(side_effect=AssertionError("No subprocess")))
    result = adapter.analyze(configured.video, deadline=time.monotonic() + 60)
    assert result == NEGATIVE
    assert result.explanation_trace["aggregation"]["completed_windows"] == 2
    assert result.explanation_trace["aggregation"]["positive_windows"] == 0
    assert [window["kind"] for window in result.explanation_trace["windows"]] == [
        "overview",
        "local",
    ]
    assert len(calls) == 2  # overview and local window
    assert not list(configured.root.glob("qwen-*"))

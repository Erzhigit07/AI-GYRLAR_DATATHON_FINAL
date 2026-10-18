"""Synthetic decision traces and generated videos; no real model or contest data."""

import copy
import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path

import pytest

av = pytest.importorskip("av")
Image = pytest.importorskip("PIL.Image")
TestClient = pytest.importorskip("fastapi.testclient").TestClient
pytest.importorskip("httpx")

from src.forte.api import create_app  # noqa: E402
from src.forte.contracts import ContractError  # noqa: E402
from src.forte.evidence import build_result, build_result_with_context  # noqa: E402
from src.forte.explanations import (  # noqa: E402
    attach_trace,
    build_explanation,
    window_trace,
)
from src.forte.qwen_baseline import aggregate_windows, to_candidate  # noqa: E402

# Public test-only credentials. These are never the operator's keys.
KEYS = {"alice": "synthetic-xai-alice-key", "bob": "synthetic-xai-bob-key"}
AUTH = {"X-API-Key": KEYS["alice"]}


def fixture_worker(*, positive=True):
    event = {"rule_id": "synthetic-rule", "time_seconds": 1.5, "description": "Synthetic event"}
    windows = [
        {
            "status": "complete",
            "analysis_complete": True,
            "uncertainty_reason": "",
            "verdict": "violation" if positive and i != 1 else "no_violation",
            "events": [dict(event)] if positive and i != 1 else [],
        }
        for i in range(3)
    ]
    aggregate = aggregate_windows(windows, expected_windows=3)
    worker = {
        "parsed_windows": windows,
        "aggregation": aggregate,
        "technical_windows": [
            {
                "index": i,
                "kind": "overview" if i == 0 else "local",
                "status": "complete",
                "frame_indices": group,
                "private_metadata": "must-not-leave-worker",
            }
            for i, group in enumerate(([0, 2, 3], [0, 1], [1, 2, 3]))
        ],
        "sampled_frames": [{"time_sec": t} for t in (0.0, 0.75, 1.5, 3.75)],
        "private_response": "must-not-leave-worker",
    }
    return worker, to_candidate(aggregate)


def synthetic_explained(video_path, *, deadline):
    assert Path(video_path).name == "input.mov"
    worker, candidate = fixture_worker()
    return attach_trace(candidate, worker, ["synthetic-rule"])


def synthetic_negative(video_path, *, deadline):
    worker, candidate = fixture_worker(positive=False)
    return attach_trace(candidate, worker, ["synthetic-rule"])


def synthetic_partial(video_path, *, deadline):
    _, candidate = fixture_worker()
    return attach_trace(candidate, {}, ["synthetic-rule"])


def synthetic_failed(video_path, *, deadline):
    raise RuntimeError("private-error-do-not-expose")


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "synthetic.mov"
    with av.open(str(path), "w", format="mov") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 80, 40, "yuv420p"
        for i, pts in enumerate((0, 1, 3, 6, 10, 15)):
            frame = av.VideoFrame.from_image(Image.new("RGB", (80, 40), (i * 40, 30, 120)))
            frame.pts, frame.time_base = pts, Fraction(1, 4)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_window_mapping_deduplication_and_whitelisted_metadata():
    worker, candidate = fixture_worker()
    original = copy.deepcopy((worker, candidate))
    trace = window_trace(worker, candidate, ["synthetic-rule"])
    assert trace["triggered_window_ids"] == ["window-0000", "window-0002"]
    assert trace["aggregation"]["positive_windows"] == 2
    assert trace["aggregation"]["negative_windows"] == 1
    assert trace["aggregation"]["distinct_events"] == 1
    assert trace["windows"][0]["kind"] == "overview"
    assert trace["windows"][0]["frame_timestamps_sec"] == [0.0, 1.5, 3.75]
    assert trace["windows"][2]["events"][0]["result_event_index"] == 0
    assert "must-not-leave-worker" not in json.dumps(trace)
    assert (worker, candidate) == original
    # Attribute metadata does not change the existing strict mapping/serialization.
    explained = attach_trace(candidate, worker, ["synthetic-rule"])
    assert explained == candidate
    assert json.dumps(explained) == json.dumps(candidate)


@pytest.mark.parametrize("fault", ["missing", "mismatch", "time", "indices", "incomplete"])
def test_bad_trace_is_unavailable_without_reclassifying_candidate(fault):
    worker, candidate = fixture_worker()
    if fault == "missing":
        del worker["parsed_windows"]
    elif fault == "mismatch":
        worker["aggregation"]["event_count"] = 9
    elif fault == "time":
        worker["parsed_windows"][0]["events"][0]["time_seconds"] = 0.3
    elif fault == "indices":
        worker["technical_windows"][0]["frame_indices"] = [0, 999]
    else:
        worker["parsed_windows"][0]["analysis_complete"] = False
    explained = attach_trace(candidate, worker, ["synthetic-rule"])
    assert explained == candidate
    assert explained.explanation_trace is None


def test_event_limit_keeps_all_window_references_and_reports_omissions():
    worker, _ = fixture_worker()
    frames = [{"time_sec": float(i)} for i in range(34)]
    windows = []
    metadata = []
    for index, selected in enumerate((range(32), range(32, 34))):
        windows.append(
            {
                "status": "complete",
                "analysis_complete": True,
                "uncertainty_reason": "",
                "verdict": "violation",
                "events": [
                    {"rule_id": "synthetic-rule", "time_seconds": float(i), "description": "Event"}
                    for i in selected
                ],
            }
        )
        metadata.append(
            {"index": index, "kind": "local", "status": "complete", "frame_indices": list(selected)}
        )
    worker.update(
        parsed_windows=windows,
        technical_windows=metadata,
        sampled_frames=frames,
        aggregation=aggregate_windows(windows, expected_windows=2),
    )
    trace = window_trace(worker, to_candidate(worker["aggregation"]), ["synthetic-rule"])
    assert trace["aggregation"]["omitted_events"] == 2
    assert trace["aggregation"]["returned_events"] == 32
    assert trace["windows"][1]["events"][0]["result_event_index"] is None
    assert trace["triggered_window_ids"] == ["window-0000", "window-0001"]


def test_context_real_pts_one_decode_and_unchanged_result(video, monkeypatch):
    import src.forte.evidence as evidence

    worker, candidate = fixture_worker()
    legacy = build_result(video, candidate)
    original_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    calls = []
    original = evidence.av.open

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(evidence.av, "open", counted)
    result, contexts = build_result_with_context(video, candidate)
    assert len(calls) == 1
    assert result == legacy
    assert [frame["timestamp_sec"] for frame in contexts[0].values()] == [0.25, 1.5, 2.5]
    explanation = build_explanation(
        attach_trace(candidate, worker, ["synthetic-rule"]), result, contexts
    )
    assert explanation["events"][0]["window_ids"] == ["window-0000", "window-0002"]
    assert explanation["events"][0]["missing_context_roles"] == []
    assert explanation["confidence"] is None
    assert hashlib.sha256(video.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize("timestamp,missing", [(0.0, "before"), (3.75, "after")])
def test_context_at_boundary_never_invents_a_before_or_after_frame(video, timestamp, missing):
    _, candidate = fixture_worker()
    candidate["violations"][0]["timecode_sec"] = timestamp
    result, contexts = build_result_with_context(video, candidate)
    assert contexts[0][missing] is None
    assert contexts[0]["event"]["timestamp_sec"] == timestamp
    explanation = build_explanation(candidate, result, contexts)
    assert explanation["status"] == "partial"
    assert explanation["aggregation"] is None
    assert explanation["events"][0]["missing_context_roles"] == [missing]


def test_invalid_event_time_cannot_be_clamped_as_context(video):
    _, candidate = fixture_worker()
    candidate["violations"][0]["timecode_sec"] = 4.0
    with pytest.raises(ContractError):
        build_result_with_context(video, candidate)


@pytest.mark.parametrize(
    "adapter,expected",
    [
        ("synthetic_explained", "available"),
        ("synthetic_negative", "available"),
        ("synthetic_partial", "partial"),
        ("synthetic_failed", "failed"),
    ],
)
def test_api_explanation_is_authenticated_and_keeps_original_result_schema(
    tmp_path, video, adapter, expected
):
    app = create_app(api_keys=KEYS, adapter=f"{__name__}:{adapter}", directory=tmp_path / "uploads")
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            headers=AUTH,
            files={"file": ("private-name.mov", video.read_bytes())},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        path = f"/api/v1/jobs/{job_id}"
        xai_path = path + "/explanation"
        assert client.get(xai_path).status_code == 401
        assert client.get(xai_path, headers={"X-API-Key": KEYS["bob"]}).status_code == 404
        assert client.get("/api/v1/jobs/unknown/explanation", headers=AUTH).status_code == 404
        started = time.monotonic()
        while time.monotonic() - started < 20:
            response = client.get(xai_path, headers=AUTH)
            assert response.headers["cache-control"] == "no-store"
            payload = response.json()
            if payload["status"] != "processing":
                break
            assert set(payload) == {"job_id", "status"}
            time.sleep(0.025)
        else:
            pytest.fail("Synthetic job did not finish")
        ordinary = client.get(path, headers=AUTH).json()
        if expected == "failed":
            assert payload == ordinary
            assert payload["status"] == "failed"
            assert "explanation" not in payload
        else:
            assert set(ordinary) == {"job_id", "status", "verdict", "violations"}
            assert payload["explanation"]["status"] == expected
            assert payload["explanation"]["verdict"] == ordinary["verdict"]
            if adapter == "synthetic_negative":
                assert payload["explanation"]["aggregation"]["positive_windows"] == 0
                assert payload["explanation"]["events"] == []
            else:
                event = payload["explanation"]["events"][0]
                assert [f["role"] for f in event["context_frames"]] == ["before", "event", "after"]
                assert (
                    event["context_frames"][1]["image_base64"]
                    == (ordinary["violations"][0]["evidence_frames"][0]["image_base64"])
                )
        encoded = json.dumps(payload)
        for private in (
            "private-name.mov",
            "must-not-leave-worker",
            "private-error",
            *KEYS.values(),
        ):
            assert private not in encoded
        assert not list((tmp_path / "uploads").iterdir())
    assert app.state.jobs.get(job_id, "alice", explanation=True) is None

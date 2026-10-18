"""Contract tests use only explicitly synthetic in-memory image fixtures."""

import base64
import copy
import io

import pytest
from PIL import Image

from src.forte.contracts import ContractError, validate_candidate, validate_result


def synthetic_frame(timestamp=0.2, image_format="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), color=(20, 120, 240)).save(buffer, format=image_format)
    return {
        "timestamp_sec": timestamp,
        "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def synthetic_result():
    frame = synthetic_frame()
    return {
        "verdict": "violation",
        "violations": [
            {
                "type": "synthetic_test_event",
                "timecode_sec": 0.2,
                "description": "Synthetic contract fixture, not a detected real event.",
                "evidence_frames": [frame],
            }
        ],
    }


def candidate():
    return {
        "verdict": "no_violation",
        "violations": [],
        "analysis_complete": True,
        "uncertain": False,
    }


def test_complete_result_and_input_are_preserved():
    result = synthetic_result()
    before = copy.deepcopy(result)
    trusted = [copy.deepcopy(result["violations"][0]["evidence_frames"])]
    assert validate_result(result, 1.0, trusted_evidence=trusted) == result
    assert result == before
    assert validate_candidate(candidate()) == {"verdict": "no_violation", "violations": []}
    assert validate_result({"verdict": "no_violation", "violations": []}, 1.0)


@pytest.mark.parametrize(
    "patch",
    [
        {"analysis_complete": False},
        {"analysis_complete": 1},
        {"uncertain": True},
        {"uncertain": 0},
        {"verdict": "unknown"},
        {"verdict": "failed"},
        {"verdict": "violation"},
        {"error": "decoder failed"},
    ],
)
def test_incomplete_error_or_unknown_is_never_negative(patch):
    source = candidate() | patch
    with pytest.raises(ContractError):
        validate_candidate(source)


def test_candidate_rejects_model_supplied_evidence_and_missing_completeness():
    source = candidate()
    del source["analysis_complete"]
    with pytest.raises(ContractError):
        validate_candidate(source)
    source = candidate() | {"verdict": "violation", "violations": synthetic_result()["violations"]}
    with pytest.raises(ContractError):
        validate_candidate(source)


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), "0.2", -0.1, 1.0])
def test_invalid_event_timestamps(value):
    result = synthetic_result()
    result["violations"][0]["timecode_sec"] = value
    with pytest.raises(ContractError):
        validate_result(result, 1.0, trusted_evidence=[[synthetic_frame()]])


@pytest.mark.parametrize("value", [True, float("nan"), -1, 1.0])
def test_invalid_evidence_timestamps(value):
    result = synthetic_result()
    result["violations"][0]["evidence_frames"][0]["timestamp_sec"] = value
    trusted = [copy.deepcopy(result["violations"][0]["evidence_frames"])]
    with pytest.raises(ContractError):
        validate_result(result, 1.0, trusted_evidence=trusted)


def test_trusted_provenance_and_exact_pts_required():
    result = synthetic_result()
    with pytest.raises(ContractError, match="trusted evidence"):
        validate_result(result, 1.0)
    with pytest.raises(ContractError, match="does not match"):
        validate_result(result, 1.0, trusted_evidence=[[synthetic_frame(0.3)]])
    result["violations"][0]["timecode_sec"] = 0.21
    with pytest.raises(ContractError, match="actual evidence frame"):
        validate_result(result, 1.0, trusted_evidence=[[synthetic_frame()]])


@pytest.mark.parametrize("frames", [[], [synthetic_frame()] * 4])
def test_evidence_count_is_bounded(frames):
    result = synthetic_result()
    result["violations"][0]["evidence_frames"] = frames
    with pytest.raises(ContractError, match="1 to 3"):
        validate_result(result, 1.0, trusted_evidence=[frames])


@pytest.mark.parametrize(
    "encoded",
    [
        "not base64%%%",
        base64.b64encode(b"not a JPEG").decode("ascii"),
        synthetic_frame(image_format="PNG")["image_base64"],
        base64.b64encode(base64.b64decode(synthetic_frame()["image_base64"])[:40]).decode("ascii"),
    ],
)
def test_rejects_invalid_nonjpeg_and_truncated_images(encoded):
    result = synthetic_result()
    result["violations"][0]["evidence_frames"][0]["image_base64"] = encoded
    trusted = [copy.deepcopy(result["violations"][0]["evidence_frames"])]
    with pytest.raises(ContractError):
        validate_result(result, 1.0, trusted_evidence=trusted)


@pytest.mark.parametrize("field", ["type", "description"])
def test_empty_event_strings(field):
    result = synthetic_result()
    result["violations"][0][field] = "  "
    with pytest.raises(ContractError):
        validate_result(result, 1.0, trusted_evidence=[[synthetic_frame()]])


def test_negative_result_cannot_hide_events_or_errors():
    result = synthetic_result() | {"verdict": "no_violation"}
    with pytest.raises(ContractError, match="inconsistent"):
        validate_result(result, 1.0)
    with pytest.raises(ContractError):
        validate_result({"verdict": "no_violation", "violations": [], "error": "failed"}, 1.0)


@pytest.mark.parametrize("duration", [0, -1, True, float("inf"), float("nan")])
def test_invalid_duration(duration):
    with pytest.raises(ContractError):
        validate_result({"verdict": "no_violation", "violations": []}, duration)

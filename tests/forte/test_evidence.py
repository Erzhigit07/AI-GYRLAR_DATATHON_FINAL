"""Evidence tests generate small synthetic MOVs; no contest data is used."""

import base64
import copy
import hashlib
import io
import time
from fractions import Fraction

import pytest
from PIL import Image

from src.forte.contracts import ContractError

av = pytest.importorskip("av", reason="ForteBank video tests require requirements-forte.txt")

# The module needs the optional PyAV dependency checked above.
from src.forte.evidence import (  # noqa: E402
    EvidenceError,
    build_result,
    extract_evidence,
    inspect_video,
)


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "synthetic_fixture.mov"
    with av.open(str(path), "w", format="mov") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 80, 40, "yuv420p"
        for index, pts in enumerate([0, 1, 4, 9]):
            image = Image.new("RGB", (80, 40), (index * 60, 50, 200 - index * 40))
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def synthetic_candidate(*timestamps):
    return {
        "verdict": "violation" if timestamps else "no_violation",
        "analysis_complete": True,
        "uncertain": False,
        "violations": [
            {
                "type": "synthetic_test_event",
                "description": "Synthetic test only, not a real finding.",
                "timecode_sec": timestamp,
            }
            for timestamp in timestamps
        ],
    }


def test_variable_pts_nearest_jpeg_aspect_and_no_source_change(synthetic_video):
    before = hashlib.sha256(synthetic_video.read_bytes()).hexdigest()
    info = inspect_video(synthetic_video)
    assert info.decoded_frames == 4
    assert info.duration_sec == pytest.approx(1.0)
    frames = extract_evidence(synthetic_video, [0.38, 0.11, 0.85], max_side=40)
    assert [item["timestamp_sec"] for item in frames] == pytest.approx([0.4, 0.1, 0.9])
    for frame in frames:
        with Image.open(io.BytesIO(base64.b64decode(frame["image_base64"]))) as image:
            assert image.format == "JPEG"
            assert image.size == (40, 20)
            image.load()
    assert hashlib.sha256(synthetic_video.read_bytes()).hexdigest() == before


def test_build_result_uses_one_decode_and_real_pts(synthetic_video, monkeypatch):
    import src.forte.evidence as module

    count = 0
    original_open = module.av.open

    def count_open(*args, **kwargs):
        nonlocal count
        count += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(module.av, "open", count_open)
    candidate = synthetic_candidate(0.38, 0.85)
    before = copy.deepcopy(candidate)
    result = build_result(synthetic_video, candidate)
    assert count == 1
    assert candidate == before
    assert result["verdict"] == "violation"
    assert [item["timecode_sec"] for item in result["violations"]] == pytest.approx([0.4, 0.9])
    assert all(len(item["evidence_frames"]) == 1 for item in result["violations"])


def test_negative_requires_readable_video(synthetic_video, tmp_path):
    assert build_result(synthetic_video, synthetic_candidate()) == {
        "verdict": "no_violation",
        "violations": [],
    }
    corrupt = tmp_path / "synthetic_invalid.mov"
    corrupt.write_bytes(b"this is a synthetic invalid video fixture")
    with pytest.raises(EvidenceError):
        build_result(corrupt, synthetic_candidate())
    with pytest.raises(EvidenceError):
        build_result(tmp_path / "missing.mov", synthetic_candidate())


def test_deadline_expiration_and_incomplete_analysis(synthetic_video):
    with pytest.raises(TimeoutError):
        build_result(synthetic_video, synthetic_candidate(0.1), deadline=time.monotonic() - 1)
    source = synthetic_candidate() | {"analysis_complete": False}
    with pytest.raises(ContractError, match="complete"):
        build_result(synthetic_video, source)


@pytest.mark.parametrize("timestamp", [-1, 1.0, 999, True, float("nan"), float("inf")])
def test_outside_and_invalid_timestamps(synthetic_video, timestamp):
    with pytest.raises(ContractError):
        build_result(synthetic_video, synthetic_candidate(timestamp))


def test_extraction_limits_and_gap(synthetic_video):
    with pytest.raises(EvidenceError, match="1 to 3"):
        extract_evidence(synthetic_video, [0.1] * 4)
    with pytest.raises(EvidenceError, match="max_side"):
        extract_evidence(synthetic_video, [0.1], max_side=20_000)
    with pytest.raises(EvidenceError, match="sufficiently close"):
        extract_evidence(synthetic_video, [0.2], max_distance_sec=0.01)

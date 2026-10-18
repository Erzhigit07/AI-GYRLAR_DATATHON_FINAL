"""Read-only, bounded local decoding and server-generated JPEG evidence.

Timestamps are relative to the first decoded video PTS. Audio is not decoded.
No fixed-frame-rate indexing, external services or model-provided images are used.
"""

import base64
import io
import math
import time
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image

from src.forte.contracts import (
    MAX_JPEG_BYTES,
    MAX_VIOLATIONS,
    ContractError,
    finite_number,
    validate_candidate,
    validate_result,
)

MAX_DECODED_FRAMES = 200_000
MAX_VIDEO_DURATION_SEC = 3600
MAX_DECODED_PIXELS = 16_777_216


class EvidenceError(ValueError):
    """Video or requested evidence cannot be decoded and verified safely."""


@dataclass(frozen=True)
class VideoInfo:
    duration_sec: float
    decoded_frames: int
    start_time_sec: float


def _deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("video processing deadline exceeded")


def _jpeg(frame: av.VideoFrame, timestamp: float, max_side: int) -> dict:
    image = frame.to_image().convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    payload = buffer.getvalue()
    if len(payload) > MAX_JPEG_BYTES:
        raise EvidenceError("encoded evidence JPEG exceeds size limit")
    return {"timestamp_sec": timestamp, "image_base64": base64.b64encode(payload).decode("ascii")}


def _scan(
    video_path: str | Path,
    requested_seconds: list[float],
    *,
    max_side: int,
    max_distance_sec: float,
    deadline: float | None,
    context_indices: frozenset[int] = frozenset(),
) -> tuple[VideoInfo, list[dict]]:
    """One full decode verifies the timeline and selects all requested frames."""
    path = Path(video_path)
    if not path.is_file():
        raise EvidenceError("video file does not exist")
    _deadline(deadline)
    targets = sorted(enumerate(requested_seconds), key=lambda item: item[1])
    selected = {}
    cursor = 0
    previous = None
    previous_time = 0.0
    first_pts = None
    frame_count = 0
    frame_duration = 0.0
    previous_delta = 0.0

    def select(index: int, requested: float, frame: av.VideoFrame, timestamp: float) -> None:
        if index not in context_indices and abs(requested - timestamp) > max_distance_sec:
            raise EvidenceError("no decoded frame sufficiently close to requested time")
        _deadline(deadline)
        selected[index] = _jpeg(frame, timestamp, max_side)

    try:
        # A supplied file must not turn decoding into an HTTP/network request.
        with av.open(str(path), mode="r", options={"protocol_whitelist": "file,pipe"}) as container:
            if len(container.streams.video) != 1:
                raise EvidenceError("exactly one video stream is required")
            stream = container.streams.video[0]
            if stream.codec_context.width * stream.codec_context.height > MAX_DECODED_PIXELS:
                raise EvidenceError("declared video frame exceeds pixel limit")
            average_rate = float(stream.average_rate) if stream.average_rate else 0.0
            for frame in container.decode(stream):
                _deadline(deadline)
                frame_count += 1
                if frame_count > MAX_DECODED_FRAMES:
                    raise EvidenceError("video frame count limit exceeded")
                if frame.is_corrupt or frame.pts is None or frame.time_base is None:
                    raise EvidenceError("video contains corrupt frames or missing PTS")
                if frame.width * frame.height > MAX_DECODED_PIXELS:
                    raise EvidenceError("decoded frame exceeds pixel limit")
                absolute = float(frame.pts * frame.time_base)
                if not math.isfinite(absolute):
                    raise EvidenceError("video has invalid frame PTS")
                if first_pts is None:
                    first_pts = absolute
                timestamp = absolute - first_pts
                if previous is not None and timestamp <= previous_time:
                    raise EvidenceError("video frame PTS must be strictly increasing")
                if timestamp >= MAX_VIDEO_DURATION_SEC:
                    raise EvidenceError("video duration limit exceeded")
                while cursor < len(targets) and targets[cursor][1] <= timestamp:
                    index, requested = targets[cursor]
                    if previous is not None and requested - previous_time <= timestamp - requested:
                        select(index, requested, previous, previous_time)
                    else:
                        select(index, requested, frame, timestamp)
                    cursor += 1
                previous_delta = timestamp - previous_time
                previous_time, previous = timestamp, frame
                frame_duration = float(frame.duration * frame.time_base) if frame.duration else 0.0
            if previous is None or first_pts is None:
                raise EvidenceError("video has no decodable frames")
            if stream.frames and frame_count != stream.frames:
                raise EvidenceError("decoded frame count does not match declared video frame count")
            if not math.isfinite(frame_duration) or frame_duration <= 0:
                frame_duration = previous_delta or (1.0 / average_rate if average_rate > 0 else 0.0)
            if frame_duration <= 0 or not math.isfinite(frame_duration):
                raise EvidenceError("final frame duration cannot be determined")
            duration = previous_time + frame_duration
            if duration > MAX_VIDEO_DURATION_SEC:
                raise EvidenceError("video duration limit exceeded")
            for index, requested in enumerate(requested_seconds):
                if index not in context_indices and not 0 <= requested < duration:
                    raise ContractError("event.timecode_sec must be within [0, duration_sec)")
            while cursor < len(targets):
                index, requested = targets[cursor]
                select(index, requested, previous, previous_time)
                cursor += 1
    except (av.FFmpegError, OSError) as exc:
        raise EvidenceError("video could not be decoded") from exc
    _deadline(deadline)
    return VideoInfo(duration, frame_count, first_pts), [selected[i] for i in range(len(targets))]


def inspect_video(video_path: str | Path, *, deadline: float | None = None) -> VideoInfo:
    """Validate every video frame and report duration derived from decoded PTS."""
    return _scan(video_path, [], max_side=960, max_distance_sec=1.0, deadline=deadline)[0]


def extract_evidence(
    video_path: str | Path,
    requested_seconds: list[float],
    *,
    max_side: int = 960,
    max_distance_sec: float = 1.0,
    deadline: float | None = None,
) -> list[dict]:
    """Extract 1–3 nearest decoded frames, preserving aspect ratio.

    ``max_distance_sec`` is an internal frame-extraction bound, not an official
    event-localization tolerance. Returned timestamps are actual decoded PTS.
    """
    if not isinstance(requested_seconds, list) or not 1 <= len(requested_seconds) <= 3:
        raise EvidenceError("request 1 to 3 evidence frames")
    if isinstance(max_side, bool) or not isinstance(max_side, int) or not 32 <= max_side <= 2048:
        raise EvidenceError("max_side must be an integer in [32, 2048]")
    distance = finite_number(max_distance_sec, "max_distance_sec")
    if distance < 0:
        raise EvidenceError("max_distance_sec must not be negative")
    requested = [finite_number(value, "requested timestamp") for value in requested_seconds]
    if any(value < 0 for value in requested):
        raise ContractError("requested timestamps must not be negative")
    return _scan(
        video_path, requested, max_side=max_side, max_distance_sec=distance, deadline=deadline
    )[1]


def build_result(
    video_path: str | Path, candidate: object, *, deadline: float | None = None
) -> dict:
    """Convert a complete analyzer candidate into a verified public result.

    All events are processed in one video decode. One real frame is attached per
    event; the final event timecode is aligned to that frame's actual PTS. An empty
    event list still requires complete analysis and a fully decodable video.
    """
    result = validate_candidate(candidate)
    requested = [event["timecode_sec"] for event in result["violations"]]
    if len(requested) > MAX_VIOLATIONS:
        raise ContractError("too many events")
    info, frames = _scan(
        video_path, requested, max_side=960, max_distance_sec=1.0, deadline=deadline
    )
    trusted_evidence = []
    for event, frame in zip(result["violations"], frames, strict=True):
        event["timecode_sec"] = frame["timestamp_sec"]
        event["evidence_frames"] = [dict(frame)]
        trusted_evidence.append([frame])
    _deadline(deadline)
    validated = validate_result(result, info.duration_sec, trusted_evidence=trusted_evidence)
    _deadline(deadline)
    return validated


def build_result_with_context(
    video_path: str | Path, candidate: object, *, deadline: float | None = None
) -> tuple[dict, list[dict]]:
    """Build the unchanged public result plus nearest frames at event ±1 second.

    One full decode checks the source and extracts all images. Only contextual
    targets may clamp to video boundaries. The event itself retains the strict
    bounds and distance checks. Missing before/after context is explicit; a
    boundary frame is never relabelled as occurring before/after itself.
    """
    result = validate_candidate(candidate)
    requested = []
    context_indices = set()
    for event in result["violations"]:
        timestamp = event["timecode_sec"]
        context_indices.update((len(requested), len(requested) + 2))
        requested.extend((timestamp - 1.0, timestamp, timestamp + 1.0))
    info, frames = _scan(
        video_path,
        requested,
        max_side=960,
        max_distance_sec=1.0,
        deadline=deadline,
        context_indices=frozenset(context_indices),
    )
    trusted = []
    contexts = []
    for index, event in enumerate(result["violations"]):
        before, center, after = frames[index * 3 : index * 3 + 3]
        event["timecode_sec"] = center["timestamp_sec"]
        event["evidence_frames"] = [dict(center)]
        trusted.append([center])
        contexts.append(
            {
                "before": before if before["timestamp_sec"] < center["timestamp_sec"] else None,
                "event": center,
                "after": after if after["timestamp_sec"] > center["timestamp_sec"] else None,
            }
        )
    validated = validate_result(result, info.duration_sec, trusted_evidence=trusted)
    # Apply the same JPEG/provenance checks to every contextual image as well.
    contextual_result = {
        "verdict": result["verdict"],
        "violations": [
            {**event, "evidence_frames": [f for f in context.values() if f is not None]}
            for event, context in zip(result["violations"], contexts, strict=True)
        ],
    }
    validate_result(
        contextual_result,
        info.duration_sec,
        trusted_evidence=[event["evidence_frames"] for event in contextual_result["violations"]],
    )
    _deadline(deadline)
    return validated, contexts

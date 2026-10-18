"""Strict public result validation; model-supplied images are never evidence."""

import base64
import binascii
import io
import math
from collections.abc import Mapping, Sequence

from PIL import Image, UnidentifiedImageError

MAX_VIOLATIONS = 32
MAX_JPEG_BYTES = 2 * 1024 * 1024
MAX_IMAGE_PIXELS = 4 * 1024 * 1024


class ContractError(ValueError):
    """Analysis is incomplete, inconsistent or not supported by trusted evidence."""


def _fields(value: object, expected: set[str], label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContractError(f"{label}: expected fields {sorted(expected)}")
    return value


def _text(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ContractError(f"{label}: expected a nonempty string of at most {limit} characters")
    return value


def finite_number(value: object, label: str) -> float:
    """Reject booleans, coercion from strings and non-finite JSON numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label}: expected a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ContractError(f"{label}: expected a finite number") from exc
    if not math.isfinite(number):
        raise ContractError(f"{label}: expected a finite number")
    return number


def _timestamp(value: object, duration_sec: float, label: str) -> float:
    number = finite_number(value, label)
    if not 0 <= number < duration_sec:
        raise ContractError(f"{label}: timestamp must be within [0, duration_sec)")
    return number


def _events(result: Mapping) -> list:
    verdict = result["verdict"]
    violations = result["violations"]
    if verdict not in ("violation", "no_violation"):
        raise ContractError("verdict: expected violation or no_violation")
    if not isinstance(violations, list) or len(violations) > MAX_VIOLATIONS:
        raise ContractError(f"violations: expected a list of at most {MAX_VIOLATIONS} events")
    if (verdict == "violation") != bool(violations):
        raise ContractError("verdict and violations are inconsistent")
    return violations


def validate_candidate(candidate: object) -> dict:
    """Accept only complete local analysis, before the server extracts evidence.

    Completeness is an explicit adapter responsibility, not a semantic correctness
    guarantee. An adapter must mark refusals, truncation and uncertainty as failures.
    No model-generated or caller-provided evidence is accepted here.
    """
    source = _fields(
        candidate,
        {"verdict", "violations", "analysis_complete", "uncertain"},
        "candidate",
    )
    if source["analysis_complete"] is not True or source["uncertain"] is not False:
        raise ContractError("analysis must be complete and must not be uncertain")
    normalized = []
    for event in _events(source):
        item = _fields(event, {"type", "timecode_sec", "description"}, "event")
        timestamp = finite_number(item["timecode_sec"], "event.timecode_sec")
        if timestamp < 0:
            raise ContractError("event.timecode_sec must not be negative")
        normalized.append(
            {
                "type": _text(item["type"], "event.type", 120),
                "timecode_sec": timestamp,
                "description": _text(item["description"], "event.description", 4000),
            }
        )
    return {"verdict": source["verdict"], "violations": normalized}


def _validate_jpeg(encoded: object) -> None:
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_JPEG_BYTES + 2) // 3):
        raise ContractError("evidence image must be bounded base64 JPEG")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ContractError("evidence image is not valid base64") from exc
    if not payload or len(payload) > MAX_JPEG_BYTES:
        raise ContractError("evidence image is empty or too large")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "JPEG":
                raise ContractError("evidence image must actually be JPEG")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ContractError("evidence image has too many pixels")
            image.verify()
        # verify() checks the file structure; load() also checks actual decoding.
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ContractError("evidence JPEG cannot be decoded") from exc


def validate_result(
    result: object,
    duration_sec: float,
    *,
    trusted_evidence: Sequence[Sequence[Mapping]] | None = None,
) -> dict:
    """Validate a public result against frames extracted by the server.

    ``trusted_evidence`` is supplied by ``build_result``, never by an analyzer or
    HTTP client. Exact byte/timestamp equality proves correspondence to those
    frames. A syntactically valid JPEG alone does not prove video provenance.
    """
    duration = finite_number(duration_sec, "duration_sec")
    if duration <= 0:
        raise ContractError("duration_sec must be positive")
    source = _fields(result, {"verdict", "violations"}, "result")
    events = _events(source)
    if events and (trusted_evidence is None or len(trusted_evidence) != len(events)):
        raise ContractError("positive results require server-extracted trusted evidence")
    if not events and trusted_evidence not in (None, [], ()):
        raise ContractError("unexpected evidence for no_violation")
    validated = []
    for index, event in enumerate(events):
        item = _fields(event, {"type", "timecode_sec", "description", "evidence_frames"}, "event")
        timestamp = _timestamp(item["timecode_sec"], duration, "event.timecode_sec")
        frames = item["evidence_frames"]
        if not isinstance(frames, list) or not 1 <= len(frames) <= 3:
            raise ContractError("each event must contain 1 to 3 evidence frames")
        trusted = list(trusted_evidence[index])
        if frames != trusted:
            raise ContractError("evidence does not match the server-extracted video frames")
        checked_frames = []
        for frame in frames:
            frame = _fields(frame, {"timestamp_sec", "image_base64"}, "evidence frame")
            frame_timestamp = _timestamp(frame["timestamp_sec"], duration, "evidence.timestamp_sec")
            _validate_jpeg(frame["image_base64"])
            checked_frames.append(
                {"timestamp_sec": frame_timestamp, "image_base64": frame["image_base64"]}
            )
        if timestamp not in [frame["timestamp_sec"] for frame in checked_frames]:
            raise ContractError("event timecode must match an actual evidence frame PTS")
        validated.append(
            {
                "type": _text(item["type"], "event.type", 120),
                "timecode_sec": timestamp,
                "description": _text(item["description"], "event.description", 4000),
                "evidence_frames": checked_frames,
            }
        )
    return {"verdict": source["verdict"], "violations": validated}

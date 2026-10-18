"""Strict, local-only result adapters for a sampled-frame binary baseline.

This module does not run inference or read videos, annotations or labels. A
complete result means that every planned window returned a well-formed definite
answer, not that every instant of a video was observed or the answer is correct.
Uncertainty and technical failures never become negative predictions.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

MAX_RESPONSE_CHARACTERS = 65536
MAX_EVENTS = 32
MAX_DESCRIPTION_CHARACTERS = 4000
MAX_UNCERTAINTY_CHARACTERS = 1000
TIMESTAMP_TOLERANCE_SEC = 0.001
WINDOW_STATUSES = frozenset(
    {"complete", "uncertain", "truncated", "invalid_json", "refusal", "timeout", "error"}
)
_RESPONSE_FIELDS = {"verdict", "analysis_complete", "events", "uncertainty_reason"}
_EVENT_FIELDS = {"time_seconds", "rule_id", "description"}


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("expected a finite number") from error
    if not math.isfinite(number):
        raise ValueError("expected a finite number")
    return number


def _string(value: object, limit: int, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError("invalid bounded string")
    return value.strip()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant")


def _window(status: str, reason_code: str, **changes: Any) -> dict[str, Any]:
    return {
        "status": status,
        "verdict": None,
        "analysis_complete": False,
        "events": [],
        "uncertainty_reason": "",
        "reason_code": reason_code,
        **changes,
    }


def parse_window_response(
    text: str | None,
    *,
    finish_reason: str | None,
    supplied_times: Sequence[float],
    allowed_rule_ids: Iterable[str],
    refusal: bool = False,
    transport_status: str | None = None,
) -> dict[str, Any]:
    """Validate strict JSON and attach events to actual supplied frame PTS.

    Timestamp rounding up to 1 ms is tolerated and replaced with the actual PTS.
    Fenced JSON, extra prose/fields, duplicate keys and NaN are rejected. A
    caller-reported transport failure/refusal takes precedence over any text.
    ``reason_code`` never includes untrusted/private model output.
    """
    if not isinstance(refusal, bool):
        raise ValueError("refusal must be a boolean")
    if transport_status not in (None, "timeout", "error"):
        raise ValueError("transport_status must be timeout, error or None")
    if not isinstance(supplied_times, Sequence) or isinstance(supplied_times, (str, bytes)):
        raise ValueError("supplied_times must be a sequence")
    times = [_finite_number(value) for value in supplied_times]
    if not times or any(value < 0 for value in times):
        raise ValueError("supplied_times must be nonempty and nonnegative")
    if isinstance(allowed_rule_ids, (str, bytes)):
        raise ValueError("allowed_rule_ids must be an iterable of rule identifiers")
    rules = {_string(value, 120) for value in allowed_rule_ids}
    if not rules:
        raise ValueError("at least one allowed rule identifier is required")
    if transport_status is not None:
        return _window(transport_status, "transport_" + transport_status)
    if refusal or finish_reason in ("content_filter", "refusal"):
        return _window("refusal", "model_refusal")
    if finish_reason in ("length", "max_tokens"):
        return _window("truncated", "generation_limit")
    if finish_reason != "stop":
        return _window("error", "unexpected_finish_reason")
    if not isinstance(text, str) or not 0 < len(text) <= MAX_RESPONSE_CHARACTERS:
        return _window("invalid_json", "response_size_or_type")
    try:
        response = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, RecursionError):
        return _window("invalid_json", "strict_json_failed")
    try:
        if not isinstance(response, dict) or set(response) != _RESPONSE_FIELDS:
            raise ValueError("response fields")
        verdict = response["verdict"]
        if verdict not in ("violation", "no_violation", "uncertain"):
            raise ValueError("verdict")
        complete = response["analysis_complete"]
        if not isinstance(complete, bool):
            raise ValueError("analysis_complete")
        reason = _string(response["uncertainty_reason"], MAX_UNCERTAINTY_CHARACTERS, empty=True)
        if not isinstance(response["events"], list) or len(response["events"]) > MAX_EVENTS:
            raise ValueError("events")
        events = []
        for event in response["events"]:
            if not isinstance(event, dict) or set(event) != _EVENT_FIELDS:
                raise ValueError("event fields")
            timestamp = _finite_number(event["time_seconds"])
            if timestamp < 0:
                raise ValueError("event timestamp")
            actual_time = min(times, key=lambda value: abs(value - timestamp))
            if abs(actual_time - timestamp) > TIMESTAMP_TOLERANCE_SEC + 1e-12:
                raise ValueError("event timestamp was not supplied")
            rule = _string(event["rule_id"], 120)
            if rule not in rules:
                raise ValueError("unknown rule")
            events.append(
                {
                    "time_seconds": actual_time,
                    "rule_id": rule,
                    "description": _string(event["description"], MAX_DESCRIPTION_CHARACTERS),
                }
            )
        if verdict == "violation" and not events:
            raise ValueError("positive verdict requires an event")
        if verdict == "no_violation" and events:
            raise ValueError("negative verdict cannot contain events")
    except (ValueError, TypeError):
        return _window("invalid_json", "schema_validation_failed")
    if not complete or verdict == "uncertain" or reason:
        return _window(
            "uncertain",
            "model_uncertain_or_incomplete",
            events=events,
            uncertainty_reason=reason,
        )
    return _window("complete", "validated", verdict=verdict, analysis_complete=True, events=events)


def aggregate_windows(
    windows: Sequence[Mapping[str, Any]], *, expected_windows: int
) -> dict[str, Any]:
    """Conservatively classify only if every planned window is complete.

    These are observations at sampled frames, not proof of continuous coverage.
    Positive events from an incomplete video are retained separately for local
    inspection and never supplied as a completed binary candidate.
    For the existing 32-event candidate limit, the earliest events are retained
    deterministically; all events and the omitted count remain in the aggregation.
    """
    if type(expected_windows) is not int or expected_windows <= 0:
        raise ValueError("expected_windows must be a positive integer")
    if len(windows) > expected_windows:
        raise ValueError("more windows than planned")
    counts: Counter[str] = Counter()
    events = []
    seen_events = set()
    for window in windows:
        status = window.get("status")
        if status not in WINDOW_STATUSES:
            raise ValueError("unknown window status")
        counts[status] += 1
        if status == "complete":
            if (
                window.get("analysis_complete") is not True
                or window.get("verdict") not in ("violation", "no_violation")
                or window.get("uncertainty_reason") != ""
                or not isinstance(window.get("events"), list)
                or (window["verdict"] == "violation") != bool(window["events"])
            ):
                raise ValueError("inconsistent complete window")
            for event in window["events"]:
                if not isinstance(event, Mapping) or set(event) != _EVENT_FIELDS:
                    raise ValueError("invalid parsed event")
                timestamp = _finite_number(event["time_seconds"])
                if timestamp < 0:
                    raise ValueError("negative parsed timestamp")
                normalized = {
                    "type": _string(event["rule_id"], 120),
                    "timecode_sec": timestamp,
                    "description": _string(event["description"], MAX_DESCRIPTION_CHARACTERS),
                }
                key = tuple(normalized.values())
                if key not in seen_events:
                    seen_events.add(key)
                    events.append(normalized)
    missing = expected_windows - len(windows)
    events.sort(key=lambda item: (item["timecode_sec"], item["type"], item["description"]))
    complete = not missing and counts["complete"] == expected_windows
    if missing:
        reason = "missing_windows"
    elif counts["complete"] != expected_windows:
        reason = "noncomplete_windows"
    else:
        reason = "validated"
    verdict = ("violation" if events else "no_violation") if complete else "uncertain"
    return {
        "status": "complete" if complete else "incomplete",
        "prediction": int(verdict == "violation") if complete else None,
        "verdict": verdict,
        "analysis_complete": complete,
        "uncertain": not complete,
        "violations": events[:MAX_EVENTS] if complete else [],
        "observed_events": events,
        "event_count": len(events),
        "events_omitted": max(0, len(events) - MAX_EVENTS) if complete else 0,
        "window_status_counts": dict(counts),
        "expected_windows": expected_windows,
        "attempted_windows": len(windows),
        "missing_windows": missing,
        "reason_code": reason,
    }


def to_candidate(aggregation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert to the existing evidence/evaluation candidate contract.

    Frame extraction and evidence provenance remain the responsibility of
    ``src.forte.evidence.build_result``. This function adds no evidence images.
    """
    if aggregation.get("status") != "complete":
        return {
            "verdict": "uncertain",
            "violations": [],
            "analysis_complete": False,
            "uncertain": True,
        }
    from src.forte.contracts import validate_candidate

    candidate = {
        "verdict": aggregation.get("verdict"),
        "violations": aggregation.get("violations"),
        "analysis_complete": aggregation.get("analysis_complete"),
        "uncertain": aggregation.get("uncertain"),
    }
    return {**validate_candidate(candidate), "analysis_complete": True, "uncertain": False}


def summarize_outcomes(
    outcomes: Sequence[Mapping[str, Any]], *, selected_count: int | None = None
) -> dict[str, Any]:
    """Binary diagnostics with explicit abstentions and null undefined ratios.

    Labels are supplied only after inference: positive integer 1 is violation.
    Accuracy/precision/recall/F1 use completed predictions exclusively. A zero
    denominator returns None, not an invented score. correct_per_attempted
    separately penalizes every attempted noncompleted prediction.
    """
    if selected_count is None:
        selected_count = len(outcomes)
    if type(selected_count) is not int or selected_count < len(outcomes):
        raise ValueError("selected_count must be an integer >= attempted count")
    counts = {name: 0 for name in ("TP", "TN", "FP", "FN")}
    statuses: Counter[str] = Counter()
    attempted_classes: Counter[str] = Counter()
    complete_classes: Counter[str] = Counter()
    for outcome in outcomes:
        label, status = outcome.get("label"), outcome.get("status")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("labels must be integer 0 or 1")
        status = _string(status, 80)
        statuses[status] += 1
        attempted_classes[str(label)] += 1
        prediction = outcome.get("prediction")
        if status != "complete":
            if prediction is not None:
                raise ValueError("noncompleted outcomes must not have binary predictions")
            continue
        if type(prediction) is not int or prediction not in (0, 1):
            raise ValueError("complete outcomes need integer 0 or 1 predictions")
        complete_classes[str(label)] += 1
        counts[("T" if prediction == label else "F") + ("P" if prediction else "N")] += 1
    tp, tn, fp, fn = (counts[name] for name in ("TP", "TN", "FP", "FN"))
    complete = sum(counts.values())

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "selected": selected_count,
        "attempted": len(outcomes),
        "not_attempted": selected_count - len(outcomes),
        "complete": complete,
        "status_counts": dict(statuses),
        "attempted_by_class": dict(attempted_classes),
        "complete_by_class": dict(complete_classes),
        "coverage_per_attempted": ratio(complete, len(outcomes)),
        "coverage_per_selected": ratio(complete, selected_count),
        "confusion_complete_only": counts,
        "accuracy_complete_only": ratio(tp + tn, complete),
        "precision_complete_only": ratio(tp, tp + fp),
        "recall_complete_only": ratio(tp, tp + fn),
        "f1_complete_only": ratio(2 * tp, 2 * tp + fp + fn),
        "correct_per_attempted": ratio(tp + tn, len(outcomes)),
        "metric_scope": "Completed binary predictions only; positive=violation; no temporal score.",
        "zero_denominator_policy": "null (undefined)",
    }

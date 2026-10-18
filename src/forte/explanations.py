"""Evidence-backed decision traces; no new inference or model explanations.

Only whitelisted, validated worker observations leave the private worker. The
trace describes the pipeline's aggregation, not the model's internal reasoning.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from src.forte.contracts import finite_number, validate_candidate

MAX_TRACE_BYTES = 4 * 1024 * 1024
MAX_WINDOWS = 4097


class ExplainedCandidate(dict):
    """Keep the strict candidate mapping unchanged, with optional internal metadata.

    Existing adapters may still return plain dictionaries. This attribute is
    consumed only inside the isolated job process and never enters model input.
    """

    def __init__(self, candidate: dict, trace: dict | None = None):
        super().__init__(candidate)
        self.explanation_trace = trace


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("Inconsistent window explanation trace")


def _event_key(event: dict) -> tuple:
    return event["type"], event["timecode_sec"], event["description"]


def window_trace(worker: dict, candidate: dict, rule_ids: list[str]) -> dict:
    """Cross-check a frozen worker's trace against its already accepted candidate.

    No mutable parser is used to reclassify outputs. Missing, malformed or
    inconsistent audit data must never produce invented counts or fragments.
    """
    result = validate_candidate(candidate)
    parsed = worker["parsed_windows"]
    technical = worker["technical_windows"]
    sampled = worker["sampled_frames"]
    aggregate = worker["aggregation"]
    _require(isinstance(parsed, list) and 1 <= len(parsed) <= MAX_WINDOWS)
    _require(isinstance(technical, list) and len(technical) == len(parsed))
    _require(isinstance(sampled, list) and 1 <= len(sampled) <= 4096)
    stamps = [finite_number(frame["time_sec"], "sampled PTS") for frame in sampled]
    _require(stamps[0] >= 0 and all(a < b for a, b in zip(stamps, stamps[1:], strict=False)))
    _require(isinstance(aggregate, Mapping))
    _require(aggregate["status"] == "complete" and aggregate["analysis_complete"] is True)
    _require(aggregate["uncertain"] is False and aggregate["missing_windows"] == 0)
    _require(aggregate["expected_windows"] == aggregate["attempted_windows"] == len(parsed))
    event_indices = {_event_key(event): i for i, event in enumerate(result["violations"])}
    observed = {}
    windows = []
    for index, (answer, metadata) in enumerate(zip(parsed, technical, strict=True)):
        _require(metadata["index"] == index and metadata["kind"] in {"overview", "local"})
        _require(metadata["status"] == answer["status"] == "complete")
        _require(answer["analysis_complete"] is True and answer["uncertainty_reason"] == "")
        indices = metadata["frame_indices"]
        _require(isinstance(indices, list) and 1 <= len(indices) <= 64)
        _require(all(type(i) is int and 0 <= i < len(stamps) for i in indices))
        _require(all(a < b for a, b in zip(indices, indices[1:], strict=False)))
        times = [stamps[i] for i in indices]
        events = answer["events"]
        _require(isinstance(events, list) and len(events) <= 32)
        _require(answer["verdict"] in {"violation", "no_violation"})
        _require((answer["verdict"] == "violation") == bool(events))
        public_events = []
        for event in events:
            timestamp = finite_number(event["time_seconds"], "event PTS")
            _require(timestamp in times and event["rule_id"] in rule_ids)
            description = event["description"]
            _require(isinstance(description, str) and 0 < len(description.strip()) <= 4000)
            normalized = {
                "type": event["rule_id"],
                "timecode_sec": timestamp,
                "description": description.strip(),
            }
            key = _event_key(normalized)
            observed[key] = normalized
            public_events.append({**normalized, "result_event_index": event_indices.get(key)})
        windows.append(
            {
                "window_id": f"window-{index:04d}",
                "index": index,
                "kind": metadata["kind"],
                "start_sec": times[0],
                "end_sec": times[-1],
                "frame_timestamps_sec": times,
                "verdict": answer["verdict"],
                "triggered": bool(events),
                "events": public_events,
            }
        )
    ordered = sorted(
        observed.values(), key=lambda e: (e["timecode_sec"], e["type"], e["description"])
    )
    _require(ordered[:32] == result["violations"] == aggregate["violations"])
    _require(aggregate["verdict"] == result["verdict"])
    _require(aggregate["event_count"] == len(ordered))
    _require(aggregate["events_omitted"] == max(0, len(ordered) - 32))
    positive = sum(window["triggered"] for window in windows)
    reason = (
        f"Все {len(windows)} окон обработаны. В {positive} окнах модель сообщила о нарушении. "
        "По действующему правилу одного сработавшего окна достаточно для итога «есть нарушение»."
        if positive
        else f"Все {len(windows)} окон обработаны. Ни в одном модель не сообщила о нарушении "
        "из проверяемого набора правил. Итог: «нарушение не обнаружено»."
    )
    trace = {
        "windows": windows,
        "triggered_window_ids": [w["window_id"] for w in windows if w["triggered"]],
        "rule_ids": list(rule_ids),
        "aggregation": {
            "method": "any_event_after_all_windows_complete",
            "verdict": result["verdict"],
            "expected_windows": len(windows),
            "completed_windows": len(windows),
            "positive_windows": positive,
            "negative_windows": len(windows) - positive,
            "distinct_events": len(ordered),
            "returned_events": len(result["violations"]),
            "omitted_events": max(0, len(ordered) - 32),
            "reason_code": "window_triggered" if positive else "no_windows_triggered",
            "explanation": reason,
        },
    }
    _require(
        len(json.dumps(trace, ensure_ascii=False, allow_nan=False).encode()) <= MAX_TRACE_BYTES
    )
    return trace


def attach_trace(candidate: dict, worker: dict, rule_ids: list[str]) -> ExplainedCandidate:
    """Audit metadata failure cannot alter a valid model decision."""
    try:
        trace = window_trace(worker, candidate, rule_ids)
    except (KeyError, TypeError, ValueError):
        trace = None
    return ExplainedCandidate(candidate, trace)


def build_explanation(candidate: dict, result: dict, contexts: list[dict]) -> dict:
    """Join trusted extracted frames with window references; never invent a cause."""
    trace = candidate.explanation_trace if isinstance(candidate, ExplainedCandidate) else None
    windows = trace["windows"] if trace is not None else []
    events = []
    for index, (event, context) in enumerate(zip(result["violations"], contexts, strict=True)):
        frames = []
        for role in ("before", "event", "after"):
            frame = context[role]
            if frame is not None:
                frames.append({"role": role, **frame})
        events.append(
            {
                "event_index": index,
                "type": event["type"],
                "timecode_sec": event["timecode_sec"],
                "description": event["description"],
                "description_source": "model_output",
                "window_ids": [
                    window["window_id"]
                    for window in windows
                    if any(item["result_event_index"] == index for item in window["events"])
                ],
                "context_frames": frames,
                "missing_context_roles": [
                    role for role in ("before", "after") if context[role] is None
                ],
            }
        )
    return {
        "schema_version": 1,
        "status": "available" if trace is not None else "partial",
        "verdict": result["verdict"],
        "kind": "evidence_and_pipeline_trace",
        "window_trace_available": trace is not None,
        "unavailable_reason": None if trace is not None else "window_trace_unavailable",
        "aggregation": trace["aggregation"] if trace is not None else None,
        "rule_ids": trace["rule_ids"] if trace is not None else None,
        "triggered_window_ids": trace["triggered_window_ids"] if trace is not None else [],
        "windows": windows,
        "events": events,
        "context_offset_sec": 1.0,
        "confidence": None,
        "limitations": [
            "Это объяснение результатов и правила их объединения, а не внутренних мыслей модели.",
            "Описание события создано моделью и может ошибаться; кадры требуют проверки человеком.",
            "Анализируются отдельные кадры. Границы окна не означают непрерывное наблюдение "
            "или длительность нарушения; обзорное окно содержит разреженные кадры.",
            "Кадры до и после извлечены сервером для проверки и могли не входить в запрос модели.",
            "Отрицательный итог относится только к проверяемым правилам и выбранным кадрам.",
        ],
    }

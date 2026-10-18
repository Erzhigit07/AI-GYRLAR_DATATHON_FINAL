"""Per-type diagnostics for partial human annotation; unknown is not negative."""

from __future__ import annotations


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _scores(tp, tn, fp, fn):
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision_on_scored_pairs": _ratio(tp, tp + fp),
        "recall_on_completed_positives": _ratio(tp, tp + fn),
        "f1_on_scored_pairs": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def evaluate_types(dataset: dict, outcomes: list[dict]) -> dict:
    if (
        dataset.get("label_authority") != "user_manual"
        or dataset.get("original_labels_used") is not False
    ):
        raise ValueError("Human-authoritative typed labels are required")
    rows = dataset["videos"]
    ids = [row["video_id"] for row in rows]
    by_id = {item["video_id"]: item for item in outcomes}
    if len(set(ids)) != len(ids) or len(by_id) != len(outcomes) or set(by_id) - set(ids):
        raise ValueError("Duplicate or unknown video identifier")
    rule_ids = dataset["rule_ids"]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("Duplicate rule identifier")
    for item in outcomes:
        if item["status"] not in {"done", "failed"}:
            raise ValueError("Expected terminal outcomes")
        if item["status"] == "done":
            predicted = {event["type"] for event in item["violations"]}
            if predicted - set(rule_ids) or (item["verdict"] == "violation") != bool(predicted):
                raise ValueError("Unknown type or inconsistent model verdict")
    per_type = []
    for rule in rule_ids:
        true_positive, true_negative, false_positive, false_negative = [], [], [], []
        failed_positive, failed_negative, unknown_predictions = [], [], []
        positive_total = negative_total = unknown_total = 0
        errors = []
        for row in rows:
            truth = row["rule_labels"][rule]
            if truth not in (0, 1, None):
                raise ValueError("Type truth must be 0, 1 or unknown")
            item = by_id.get(row["video_id"])
            done = item is not None and item["status"] == "done"
            predicted = done and any(e["type"] == rule for e in item["violations"])
            if truth is None:
                unknown_total += 1
                if predicted:
                    unknown_predictions.append(row["video_id"])
                continue
            positive_total += truth == 1
            negative_total += truth == 0
            if not done:
                (failed_positive if truth else failed_negative).append(row["video_id"])
            elif truth:
                (true_positive if predicted else false_negative).append(row["video_id"])
                if predicted:
                    times = [e["timecode_sec"] for e in item["violations"] if e["type"] == rule]
                    for anchor in row["event_anchors"]:
                        if anchor["type"] == rule:
                            errors.append(
                                {
                                    "video_id": row["video_id"],
                                    "human_time_sec": anchor["timecode_sec"],
                                    "nearest_abs_error_sec": min(
                                        abs(t - anchor["timecode_sec"]) for t in times
                                    ),
                                }
                            )
            else:
                (false_positive if predicted else true_negative).append(row["video_id"])
        per_type.append(
            {
                "rule_id": rule,
                "annotated_positive_videos": positive_total,
                "annotated_negative_videos": negative_total,
                "unknown_pairs": unknown_total,
                **_scores(
                    len(true_positive), len(true_negative), len(false_positive), len(false_negative)
                ),
                "detected_over_all_annotated_positives": _ratio(len(true_positive), positive_total),
                "failed_positive_count": len(failed_positive),
                "failed_negative_count": len(failed_negative),
                "detected_positive_ids": true_positive,
                "missed_positive_ids": false_negative,
                "false_positive_negative_ids": false_positive,
                "failed_positive_ids": failed_positive,
                "failed_negative_ids": failed_negative,
                "unscored_predictions_on_other_positive_videos": unknown_predictions,
                "nearest_anchor_errors": errors,
            }
        )
    completed = [row for row in rows if by_id.get(row["video_id"], {}).get("status") == "done"]
    tp = tn = fp = fn = 0
    for row in completed:
        positive = by_id[row["video_id"]]["verdict"] == "violation"
        tp += row["label"] == 1 and positive
        fn += row["label"] == 1 and not positive
        fp += row["label"] == 0 and positive
        tn += row["label"] == 0 and not positive
    return {
        "videos": len(rows),
        "attempted": len(outcomes),
        "completed": len(completed),
        "failed_or_missing": len(rows) - len(completed),
        "coverage": _ratio(len(completed), len(rows)),
        "binary_complete_only": {
            **_scores(tp, tn, fp, fn),
            "accuracy": _ratio(tp + tn, len(completed)),
        },
        "per_type": per_type,
        "evaluation_kind": "diagnostic_on_previously_exposed_manual_data",
        "official_temporal_metric": "NOT_EVALUATED",
        "limitations": [
            "Types were mapped from human comments before inference; not independent test data.",
            "Unmentioned types in human-positive videos are UNKNOWN and excluded from TP/TN/FP/FN.",
            "Failed/missing jobs are reported separately and never converted "
            "to negative predictions.",
            "Nearest anchor distances are descriptive, not an official event matching score.",
        ],
    }

"""Synthetic catalogs and partial labels; no private data or model execution."""

import copy
import json
from pathlib import Path

import pytest

from src.forte.rule_catalog import catalog_rules, manual_prompt
from src.forte.type_evaluation import evaluate_types


def catalog():
    return {
        "detection_rules": [
            {
                "rule_id": "synthetic-rule",
                "title": "Synthetic criterion",
                "enabled": True,
                "source": "Human annotation",
                "source_kind": "user_manual_annotation",
                "clause": "",
                "observable_predicate": "Visible synthetic action",
                "exclude": "Ordinary motion",
                "private_note": "PRIVATE_DO_NOT_PUBLISH",
            }
        ]
    }


def test_user_rules_publish_honest_source_and_no_private_fields():
    result = catalog_rules(catalog(), ["synthetic-rule"])
    assert result[0]["source_kind"] == "user_manual_annotation"
    assert result[0]["source"] == "Human annotation"
    assert "private_note" not in result[0]
    assert "PRIVATE_DO_NOT_PUBLISH" not in manual_prompt(catalog())


@pytest.mark.parametrize("damage", ["duplicate", "disabled", "source", "both", "unknown"])
def test_invalid_rule_catalog_is_rejected(damage):
    data = catalog()
    rule = data["detection_rules"][0]
    selected = ["synthetic-rule"]
    if damage == "duplicate":
        data["detection_rules"].append(dict(rule))
    elif damage == "disabled":
        rule["enabled"] = False
    elif damage == "source":
        rule["source"] = ""
    elif damage == "both":
        data["verified_detection_rules"] = data["detection_rules"]
    else:
        selected = ["missing-rule"]
    with pytest.raises(ValueError):
        catalog_rules(data, selected)


def test_manual_catalog_has_all_eleven_types_and_no_video_answers():
    root = Path(__file__).resolve().parents[2]
    data = json.loads((root / "configs/forte_rules.manual_v1.json").read_text(encoding="utf-8"))
    ids = [rule["rule_id"] for rule in data["detection_rules"]]
    assert len(ids) == len(set(ids)) == 11
    assert "other_activity_with_unsecured_cash" in ids
    assert len(catalog_rules(data, ids)) == 11
    prompt = manual_prompt(data)
    assert all(rule_id in prompt for rule_id in ids)
    assert "X-46" not in prompt and "video_40" not in prompt
    assert "13 сек" not in prompt


def fixture_dataset():
    return {
        "label_authority": "user_manual",
        "original_labels_used": False,
        "rule_ids": ["rule-a", "rule-b"],
        "videos": [
            {
                "video_id": "positive",
                "label": 1,
                "rule_labels": {"rule-a": 1, "rule-b": None},
                "event_anchors": [{"type": "rule-a", "timecode_sec": 4.0}],
            },
            {
                "video_id": "other-positive",
                "label": 1,
                "rule_labels": {"rule-a": None, "rule-b": 1},
                "event_anchors": [],
            },
            {
                "video_id": "negative",
                "label": 0,
                "rule_labels": {"rule-a": 0, "rule-b": 0},
                "event_anchors": [],
            },
        ],
    }


def outcome(video, *types):
    return {
        "video_id": video,
        "status": "done",
        "verdict": "violation" if types else "no_violation",
        "violations": [{"type": t, "timecode_sec": 5.0, "description": "Synthetic"} for t in types],
    }


def test_unknown_types_are_not_false_positives_and_failures_not_negatives():
    results = [
        outcome("positive", "rule-a"),
        outcome("other-positive", "rule-a"),
        {"video_id": "negative", "status": "failed"},
    ]
    metrics = evaluate_types(fixture_dataset(), results)
    a, b = metrics["per_type"]
    assert (a["TP"], a["FP"], a["TN"], a["FN"]) == (1, 0, 0, 0)
    assert a["failed_negative_count"] == 1
    assert a["unscored_predictions_on_other_positive_videos"] == ["other-positive"]
    assert a["nearest_anchor_errors"][0]["nearest_abs_error_sec"] == 1.0
    assert b["FN"] == 1 and b["failed_negative_count"] == 1
    assert metrics["coverage"] == pytest.approx(2 / 3)


def test_known_negative_false_alarm_and_missing_positive_are_visible():
    metrics = evaluate_types(fixture_dataset(), [outcome("negative", "rule-a")])
    a = metrics["per_type"][0]
    assert a["FP"] == 1 and a["failed_positive_count"] == 1
    assert a["FN"] == 0
    assert a["detected_over_all_annotated_positives"] == 0
    assert a["recall_on_completed_positives"] is None


def test_original_authority_and_unknown_prediction_rejected():
    data = fixture_dataset()
    wrong = copy.deepcopy(data)
    wrong["label_authority"] = "original"
    with pytest.raises(ValueError):
        evaluate_types(wrong, [])
    with pytest.raises(ValueError):
        evaluate_types(data, [outcome("positive", "invented-rule")])
    with pytest.raises(ValueError):
        evaluate_types(data, [outcome("positive"), outcome("positive")])

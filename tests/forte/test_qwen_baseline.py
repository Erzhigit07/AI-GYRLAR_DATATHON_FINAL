"""Synthetic JSON fixtures only: no videos, private annotations or model calls."""

import importlib.util
import json
import unittest

from src.forte.qwen_baseline import (
    MAX_EVENTS,
    MAX_RESPONSE_CHARACTERS,
    aggregate_windows,
    parse_window_response,
    summarize_outcomes,
    to_candidate,
)


def response(verdict="no_violation", *, complete=True, events=None, reason=""):
    return {
        "verdict": verdict,
        "analysis_complete": complete,
        "events": events if events is not None else [],
        "uncertainty_reason": reason,
    }


def event(timestamp=1.23456, rule="synthetic_rule", description="Synthetic fixture"):
    return {"time_seconds": timestamp, "rule_id": rule, "description": description}


def parse(value=None, **kwargs):
    text = json.dumps(response() if value is None else value)
    return parse_window_response(
        text,
        finish_reason=kwargs.pop("finish_reason", "stop"),
        supplied_times=kwargs.pop("supplied_times", [0.0, 1.23456, 2.0]),
        allowed_rule_ids=kwargs.pop("allowed_rule_ids", {"synthetic_rule"}),
        **kwargs,
    )


class ParseWindowTests(unittest.TestCase):
    def test_complete_negative(self):
        result = parse()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["verdict"], "no_violation")
        self.assertTrue(result["analysis_complete"])

    def test_complete_positive_normalizes_rounded_pts(self):
        result = parse(response("violation", events=[event(1.235)]))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["events"][0]["time_seconds"], 1.23456)

    def test_positive_without_event_is_invalid(self):
        self.assertEqual(parse(response("violation"))["status"], "invalid_json")

    def test_negative_with_event_is_invalid(self):
        self.assertEqual(parse(response(events=[event()]))["status"], "invalid_json")

    def test_unknown_rule_is_invalid(self):
        self.assertEqual(
            parse(response("violation", events=[event(rule="unknown_rule")]))["status"],
            "invalid_json",
        )

    def test_unsupplied_timestamp_is_invalid(self):
        for timestamp in (1.0, 1.236, -0.0001, True, "1.23456", float("inf"), float("nan")):
            with self.subTest(timestamp=timestamp):
                result = parse(response("violation", events=[event(timestamp)]))
                self.assertEqual(result["status"], "invalid_json")

    def test_uncertainty_never_is_a_negative_prediction(self):
        variants = [
            response(complete=False),
            response(reason="Synthetic ambiguity"),
            response("uncertain", complete=False, reason="Synthetic ambiguity"),
            response("uncertain", complete=True),
            response("violation", complete=False, events=[event()]),
        ]
        for value in variants:
            with self.subTest(value=value):
                result = parse(value)
                self.assertEqual(result["status"], "uncertain")
                self.assertIsNone(result["verdict"])
                self.assertFalse(result["analysis_complete"])

    def test_technical_outcomes_are_separate(self):
        variants = [
            ({"finish_reason": "length"}, "truncated"),
            ({"finish_reason": "max_tokens"}, "truncated"),
            ({"finish_reason": "content_filter"}, "refusal"),
            ({"finish_reason": "refusal"}, "refusal"),
            ({"refusal": True}, "refusal"),
            ({"transport_status": "timeout"}, "timeout"),
            ({"transport_status": "error"}, "error"),
            ({"finish_reason": None}, "error"),
            ({"finish_reason": "tool_calls"}, "error"),
        ]
        for kwargs, expected in variants:
            with self.subTest(kwargs=kwargs):
                result = parse(**kwargs)
                self.assertEqual(result["status"], expected)
                self.assertIsNone(result["verdict"])
                self.assertFalse(result["analysis_complete"])

    def test_transport_failure_precedes_complete_response(self):
        result = parse(response("violation", events=[event()]), transport_status="timeout")
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["events"], [])

    def test_invalid_json_duplicate_keys_fences_and_constants(self):
        valid = json.dumps(response())
        variants = [
            valid.replace('"analysis_complete": true', '"analysis_complete": NaN'),
            valid.replace('"analysis_complete": true', '"analysis_complete": Infinity'),
            valid.replace('"events": []', '"events": [], "events": []'),
            json.dumps(response("violation", events=[event()])).replace(
                '"rule_id": "synthetic_rule"',
                '"rule_id": "synthetic_rule", "rule_id": "synthetic_rule"',
            ),
            "```json\n" + valid + "\n```",
            valid + " trailing explanation",
            "[]",
            "null",
            "",
            None,
            "{" * 2000,
        ]
        for text in variants:
            with self.subTest(text_type=type(text).__name__):
                result = parse_window_response(
                    text,
                    finish_reason="stop",
                    supplied_times=[1.23456],
                    allowed_rule_ids=["synthetic_rule"],
                )
                self.assertEqual(result["status"], "invalid_json")
                self.assertNotIn("Synthetic", result["reason_code"])

    def test_extra_fields_and_boolean_coercion_rejected(self):
        variants = [
            {**response(), "extra": "Synthetic"},
            {**response(), "analysis_complete": 1},
            {**response(), "analysis_complete": "true"},
            {**response(), "verdict": ["no_violation"]},
            {**response(), "events": {}},
            {**response(), "uncertainty_reason": None},
            response("violation", events=[{**event(), "image_base64": "not_evidence"}]),
        ]
        for value in variants:
            with self.subTest(value=value):
                self.assertEqual(parse(value)["status"], "invalid_json")

    def test_string_and_event_count_limits(self):
        variants = [
            response("violation", events=[event(description="")]),
            response("violation", events=[event(description=" " * 4)]),
            response("violation", events=[event(description="x" * 4001)]),
            response("violation", events=[event()] * (MAX_EVENTS + 1)),
            response(reason="x" * 1001),
        ]
        for value in variants:
            with self.subTest(value_type=type(value).__name__):
                self.assertEqual(parse(value)["status"], "invalid_json")
        too_long = " " * (MAX_RESPONSE_CHARACTERS + 1)
        self.assertEqual(
            parse_window_response(
                too_long, finish_reason="stop", supplied_times=[0], allowed_rule_ids=["test"]
            )["status"],
            "invalid_json",
        )

    def test_invalid_configuration_raises(self):
        variants = [
            {"supplied_times": []},
            {"supplied_times": [float("nan")]},
            {"supplied_times": [-1]},
            {"supplied_times": [True]},
            {"supplied_times": "1.0"},
            {"allowed_rule_ids": []},
            {"allowed_rule_ids": "synthetic_rule"},
            {"refusal": "false"},
            {"transport_status": "complete"},
        ]
        for kwargs in variants:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                parse(**kwargs)


class AggregateTests(unittest.TestCase):
    def test_all_negative_windows_are_complete(self):
        result = aggregate_windows([parse(), parse()], expected_windows=2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["prediction"], 0)
        self.assertEqual(result["violations"], [])

    def test_positive_window_and_all_complete_is_positive(self):
        positive = parse(response("violation", events=[event()]))
        result = aggregate_windows([parse(), positive], expected_windows=2)
        self.assertEqual(result["prediction"], 1)
        self.assertEqual(result["violations"][0]["timecode_sec"], 1.23456)
        self.assertEqual(result["violations"][0]["type"], "synthetic_rule")

    def test_any_failed_window_abstains_even_with_positive(self):
        positive = parse(response("violation", events=[event()]))
        failures = [
            parse(response("uncertain", complete=False)),
            parse(finish_reason="length"),
            parse(response("violation")),
            parse(refusal=True),
            parse(transport_status="timeout"),
            parse(transport_status="error"),
        ]
        for failure in failures:
            with self.subTest(status=failure["status"]):
                result = aggregate_windows([positive, failure], expected_windows=2)
                self.assertEqual(result["status"], "incomplete")
                self.assertIsNone(result["prediction"])
                self.assertTrue(result["uncertain"])
                self.assertEqual(result["violations"], [])
                self.assertEqual(len(result["observed_events"]), 1)
                self.assertEqual(result["window_status_counts"][failure["status"]], 1)

    def test_missing_windows_do_not_implicitly_become_negative(self):
        for windows in ([], [parse()]):
            with self.subTest(count=len(windows)):
                result = aggregate_windows(windows, expected_windows=2)
                self.assertEqual(result["reason_code"], "missing_windows")
                self.assertIsNone(result["prediction"])
                self.assertEqual(result["missing_windows"], 2 - len(windows))

    def test_duplicate_overlap_events_are_deduplicated(self):
        positive = parse(response("violation", events=[event()]))
        result = aggregate_windows([positive, positive], expected_windows=2)
        self.assertEqual(len(result["violations"]), 1)

    def test_event_limit_retains_positive_with_explicit_omitted_count(self):
        early_events = [event(timestamp=index) for index in range(MAX_EVENTS)]
        positives = [
            parse(
                response("violation", events=[event(timestamp=MAX_EVENTS)]),
                supplied_times=[MAX_EVENTS],
            ),
            parse(
                response("violation", events=early_events), supplied_times=list(range(MAX_EVENTS))
            ),
        ]
        result = aggregate_windows(positives, expected_windows=len(positives))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["prediction"], 1)
        self.assertEqual(result["events_omitted"], 1)
        self.assertEqual(len(result["violations"]), MAX_EVENTS)
        self.assertEqual(len(result["observed_events"]), MAX_EVENTS + 1)
        self.assertEqual(result["violations"][0]["timecode_sec"], 0)
        self.assertEqual(result["violations"][-1]["timecode_sec"], MAX_EVENTS - 1)

    def test_invalid_aggregation_is_rejected(self):
        variants = [
            ([], 0),
            ([], True),
            ([parse(), parse()], 1),
            ([{"status": "invented_status"}], 1),
            ([{**parse(), "verdict": "violation"}], 1),
            ([{**parse(), "uncertainty_reason": "ambiguous"}], 1),
        ]
        for windows, count in variants:
            with self.subTest(count=count), self.assertRaises(ValueError):
                aggregate_windows(windows, expected_windows=count)

    def test_incomplete_adapter_keeps_explicit_uncertainty(self):
        result = to_candidate(aggregate_windows([], expected_windows=1))
        self.assertEqual(result["verdict"], "uncertain")
        self.assertFalse(result["analysis_complete"])
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["violations"], [])

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow absent in this environment")
    def test_complete_adapter_is_accepted_by_existing_contract(self):
        from src.forte.contracts import validate_candidate

        for window in (parse(), parse(response("violation", events=[event()]))):
            with self.subTest(verdict=window["verdict"]):
                result = to_candidate(aggregate_windows([window], expected_windows=1))
                self.assertEqual(validate_candidate(result)["verdict"], window["verdict"])
                self.assertEqual(
                    set(result), {"verdict", "violations", "analysis_complete", "uncertain"}
                )


class MetricTests(unittest.TestCase):
    def test_known_confusion_and_coverage(self):
        outcomes = [
            {"label": 1, "status": "complete", "prediction": 1},
            {"label": 0, "status": "complete", "prediction": 0},
            {"label": 0, "status": "complete", "prediction": 1},
            {"label": 1, "status": "complete", "prediction": 0},
            {"label": 1, "status": "incomplete"},
            {"label": 0, "status": "timeout"},
        ]
        result = summarize_outcomes(outcomes, selected_count=8)
        self.assertEqual(result["confusion_complete_only"], {"TP": 1, "TN": 1, "FP": 1, "FN": 1})
        for metric in ("accuracy", "precision", "recall", "f1"):
            self.assertEqual(result[metric + "_complete_only"], 0.5)
        self.assertAlmostEqual(result["coverage_per_attempted"], 4 / 6)
        self.assertEqual(result["coverage_per_selected"], 0.5)
        self.assertAlmostEqual(result["correct_per_attempted"], 2 / 6)
        self.assertEqual(result["not_attempted"], 2)

    def test_positive_class_uses_violation(self):
        outcomes = [
            {"label": 1, "status": "complete", "prediction": 1},
            {"label": 1, "status": "complete", "prediction": 0},
            {"label": 1, "status": "complete", "prediction": 0},
            {"label": 0, "status": "complete", "prediction": 1},
        ]
        result = summarize_outcomes(outcomes)
        self.assertEqual(result["confusion_complete_only"], {"TP": 1, "TN": 0, "FP": 1, "FN": 2})
        self.assertEqual(result["precision_complete_only"], 0.5)
        self.assertAlmostEqual(result["recall_complete_only"], 1 / 3)
        self.assertEqual(result["f1_complete_only"], 0.4)

    def test_undefined_metrics_are_null(self):
        result = summarize_outcomes([])
        for key in (
            "accuracy_complete_only",
            "precision_complete_only",
            "recall_complete_only",
            "f1_complete_only",
            "coverage_per_attempted",
            "coverage_per_selected",
            "correct_per_attempted",
        ):
            self.assertIsNone(result[key])
        result = summarize_outcomes([{"label": 0, "status": "complete", "prediction": 0}])
        self.assertEqual(result["accuracy_complete_only"], 1.0)
        self.assertIsNone(result["precision_complete_only"])
        self.assertIsNone(result["recall_complete_only"])
        self.assertIsNone(result["f1_complete_only"])

    def test_only_failed_outcomes_report_zero_coverage_and_null_accuracy(self):
        result = summarize_outcomes([{"label": 0, "status": "timeout"}])
        self.assertEqual(result["coverage_per_attempted"], 0.0)
        self.assertEqual(result["correct_per_attempted"], 0.0)
        self.assertIsNone(result["accuracy_complete_only"])

    def test_incomplete_negative_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_outcomes([{"label": 0, "status": "incomplete", "prediction": 0}])

    def test_bad_labels_predictions_and_selected_count_are_rejected(self):
        variants = [
            ([{"label": True, "status": "complete", "prediction": 1}], None),
            ([{"label": 0, "status": "complete", "prediction": False}], None),
            ([{"label": 2, "status": "complete", "prediction": 1}], None),
            ([{"label": 0, "status": "complete"}], None),
            ([{"label": 0, "status": "error"}], 0),
            ([], -1),
            ([], True),
        ]
        for outcomes, count in variants:
            with self.subTest(count=count), self.assertRaises(ValueError):
                summarize_outcomes(outcomes, selected_count=count)


if __name__ == "__main__":
    unittest.main()

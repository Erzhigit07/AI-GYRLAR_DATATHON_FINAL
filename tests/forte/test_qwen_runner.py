"""Synthetic runner checks: mocked decoding/HTTP, no files, models or private data."""

import argparse
import copy
import importlib.util
import json
import time
import unittest
import urllib.error
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("av", "PIL", "psutil"))
runner = None
if HAS_RUNTIME:
    runner_path = Path(__file__).resolve().parents[2] / "experiments/forte_qwen_baseline.py"
    spec = importlib.util.spec_from_file_location("synthetic_qwen_runner_tests", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)


def model_response(verdict="no_violation", *, finish="stop", complete=True, timestamp=None):
    events = []
    if timestamp is not None:
        events = [
            {"time_seconds": timestamp, "rule_id": "synthetic_rule", "description": "Synthetic"}
        ]
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "verdict": verdict,
                            "analysis_complete": complete,
                            "events": events,
                            "uncertainty_reason": "",
                        }
                    )
                },
                "finish_reason": finish,
            }
        ]
    }


def schema_accepts(value, schema):
    """Independent stdlib validator for the bounded keywords used by this schema."""
    if "oneOf" in schema:
        return sum(schema_accepts(value, item) for item in schema["oneOf"]) == 1
    if "const" in schema:
        return type(value) is type(schema["const"]) and value == schema["const"]
    if "enum" in schema and not any(
        type(value) is type(item)
        and value == item
        or type(value) in (int, float)
        and type(item) in (int, float)
        and value == item
        for item in schema["enum"]
    ):
        return False
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return False
        properties = schema["properties"]
        return (
            set(schema.get("required", [])).issubset(value)
            and (
                schema.get("additionalProperties") is not False or not set(value) - set(properties)
            )
            and all(schema_accepts(item, properties[key]) for key, item in value.items())
        )
    if kind == "array":
        return (
            isinstance(value, list)
            and schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value))
            and (
                "items" not in schema
                or all(schema_accepts(item, schema["items"]) for item in value)
            )
        )
    if kind == "string":
        return isinstance(value, str) and schema.get("minLength", 0) <= len(value) <= schema.get(
            "maxLength", len(value)
        )
    return kind == "number" and type(value) in (int, float)


def synthetic_worker(timestamps, *, replies=None, size=16, overlap=2, overview=16):
    settings = {
        "frames_per_window": size,
        "overlap_frames": overlap,
        "overview_frames": overview,
        "prompt": "SYNTHETIC RULE INSTRUCTION",
        "model_id": "synthetic-model",
        "generation": {"temperature": 0, "seed": 42},
        "port": 1,
        "rule_ids": ["synthetic_rule"],
        "request_timeout_sec": 10,
    }
    args = argparse.Namespace(
        worker_settings=Mock(read_text=Mock(return_value=json.dumps(settings))),
        worker_budget=30,
        worker_result=Path("synthetic-results/result.json"),
        worker_input="SYNTHETIC_SOURCE_LABEL_AND_FILENAME.mov",
    )
    frames = [
        {
            "time_sec": timestamp,
            "jpeg_sha256": "synthetic-sha",
            "width": 16,
            "height": 16,
            "base64": "SYNTHETIC_NOT_AN_IMAGE",
        }
        for timestamp in timestamps
    ]
    saved, requests = {}, []
    reply_iterator = iter(replies) if replies is not None else None

    def fake_request(*args, **kwargs):
        requests.append(kwargs["body"])
        reply = next(reply_iterator) if reply_iterator is not None else model_response()
        if isinstance(reply, Exception):
            raise reply
        return reply

    def fake_write(path, data):
        name = Path(path).name
        if name in saved:
            raise FileExistsError("Synthetic exclusive-output check")
        saved[name] = json.loads(json.dumps(data, allow_nan=False))

    with (
        patch.object(runner, "sample_video", return_value=frames),
        patch.object(runner, "request", side_effect=fake_request),
        patch.object(runner, "write_new", side_effect=fake_write),
    ):
        code = runner.run_worker(args)
    return code, saved, requests


@unittest.skipUnless(HAS_RUNTIME, "Optional video runtime (av/Pillow/psutil) unavailable")
class RunnerTests(unittest.TestCase):
    def test_schema_verdict_branches_are_mutually_exclusive_and_consistent(self):
        schema = runner.output_schema([0.0, 1.25], ["synthetic_rule"])
        positive = {
            "verdict": "violation",
            "analysis_complete": True,
            "events": [
                {"time_seconds": 1.25, "rule_id": "synthetic_rule", "description": "Synthetic"}
            ],
            "uncertainty_reason": "",
        }
        negative = {**positive, "verdict": "no_violation", "events": []}
        uncertain = {
            **negative,
            "verdict": "uncertain",
            "analysis_complete": False,
            "uncertainty_reason": "Synthetic ambiguity",
        }
        for value in (positive, negative, uncertain):
            with self.subTest(verdict=value["verdict"]):
                self.assertTrue(schema_accepts(value, schema))
                self.assertEqual(
                    sum(schema_accepts(value, branch) for branch in schema["oneOf"]), 1
                )
                self.assertFalse(
                    schema_accepts(
                        {**value, "analysis_complete": not value["analysis_complete"]}, schema
                    )
                )
        invalid = [
            {**negative, "events": positive["events"]},
            {**positive, "events": []},
            {**positive, "uncertainty_reason": "Synthetic ambiguity"},
            {**negative, "uncertainty_reason": "Synthetic ambiguity"},
            {**uncertain, "uncertainty_reason": ""},
            {**uncertain, "events": positive["events"]},
        ]
        for value in invalid:
            self.assertFalse(schema_accepts(value, schema))

    def test_schema_rejects_missing_extra_unbounded_and_unsupplied_fields(self):
        schema = runner.output_schema([0.0, 1.25], ["synthetic_rule"])
        positive = {
            "verdict": "violation",
            "analysis_complete": True,
            "events": [
                {"time_seconds": 1.25, "rule_id": "synthetic_rule", "description": "Synthetic"}
            ],
            "uncertainty_reason": "",
        }
        for field in positive:
            value = copy.deepcopy(positive)
            del value[field]
            self.assertFalse(schema_accepts(value, schema))
        self.assertFalse(schema_accepts({**positive, "extra": "Synthetic"}, schema))
        for field, replacement in (
            ("time_seconds", 2.0),
            ("time_seconds", True),
            ("time_seconds", "1.25"),
            ("rule_id", "unknown_synthetic_rule"),
            ("description", ""),
            ("description", "x" * 201),
        ):
            value = copy.deepcopy(positive)
            value["events"][0][field] = replacement
            self.assertFalse(schema_accepts(value, schema))
        value = copy.deepcopy(positive)
        value["events"] *= 4
        self.assertFalse(schema_accepts(value, schema))

    def test_sampling_grid_skips_gap_and_uses_first_pts_origin(self):
        class SyntheticContainer:
            streams = SimpleNamespace(
                video=[SimpleNamespace(codec_context=SimpleNamespace(thread_count=0))]
            )

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def decode(self, stream):
                for pts in (50, 150, 151, 152, 160):
                    yield SimpleNamespace(
                        pts=pts,
                        time_base=Fraction(1, 10),
                        is_corrupt=False,
                        to_image=lambda: runner.Image.new("RGB", (32, 16)),
                    )

        settings = {
            "sample_step_sec": 1.0,
            "image_max_side": 16,
            "jpeg_quality": 85,
            "max_sampled_frames": 20,
        }
        with patch.object(runner.av, "open", return_value=SyntheticContainer()) as decode:
            frames = runner.sample_video("SYNTHETIC", settings, time.monotonic() + 30)
        self.assertEqual([frame["time_sec"] for frame in frames], [0.0, 10.0, 11.0])
        self.assertTrue(all((frame["width"], frame["height"]) == (16, 8) for frame in frames))
        self.assertEqual(decode.call_args.kwargs["options"], {"protocol_whitelist": "file,pipe"})

    def test_windows_cover_every_index_and_network_guard_is_local(self):
        for count in (1, 15, 16, 17, 33, 286):
            with self.subTest(count=count):
                groups = runner.windows(count, 16, 2)
                self.assertEqual(set().union(*map(set, groups)), set(range(count)))
                self.assertEqual(groups[-1][-1], count - 1)
                for previous, current in zip(groups, groups[1:], strict=False):
                    self.assertEqual(previous[-2:], current[:2])
        with self.assertRaises(ValueError):
            runner.windows(4, 2, 2)
        with self.assertRaises(RuntimeError):
            runner.guard("socket.connect", (None, ("8.8.8.8", 443)))
        with self.assertRaises(RuntimeError):
            runner.guard("socket.getaddrinfo", ("synthetic.example", 443))
        runner.guard("socket.connect", (None, ("127.0.0.1", 8767)))

    def test_overview_is_evenly_spaced_and_payload_omits_source_identity(self):
        code, saved, requests = synthetic_worker([index * 0.5 for index in range(35)])
        result = saved["result.json"]
        self.assertEqual(code, 0)
        self.assertEqual(result["candidate"]["verdict"], "no_violation")
        overview = saved["request_000.json"]
        self.assertEqual(overview["kind"], "overview")
        self.assertEqual(overview["frame_indices"], [round(index * 34 / 15) for index in range(16)])
        self.assertTrue(all(item["kind"] == "local" for item in result["technical_windows"][1:]))
        for body in requests:
            encoded = json.dumps(body)
            self.assertNotIn("SYNTHETIC_SOURCE_LABEL_AND_FILENAME", encoded)
            self.assertNotIn("worker_input", encoded)
            self.assertNotIn("annotations", encoded)
            self.assertNotIn("sampled_frames", encoded)
        self.assertIn("OVERVIEW:", requests[0]["messages"][0]["content"][0]["text"])
        self.assertIn("LOCAL WINDOW:", requests[1]["messages"][0]["content"][0]["text"])

    def test_positive_candidate_preserves_actual_pts_after_rounding(self):
        code, saved, _ = synthetic_worker(
            [0.0, 1.23456],
            replies=[model_response("violation", timestamp=1.235), model_response()],
            size=2,
            overlap=0,
        )
        result = saved["result.json"]
        self.assertEqual(code, 0)
        self.assertEqual(result["aggregation"]["prediction"], 1)
        self.assertEqual(result["candidate"]["violations"][0]["timecode_sec"], 1.23456)
        self.assertNotIn("base64", result["sampled_frames"][0])

    def test_transport_failure_retains_completed_and_failed_window_checkpoints(self):
        failures = [(TimeoutError("synthetic"), "timeout")]
        failures.append((urllib.error.URLError("synthetic transport failure"), "error"))
        for error, status in failures:
            with self.subTest(status=status):
                code, saved, requests = synthetic_worker(
                    [0.0, 0.5, 1.0],
                    replies=[model_response(), error],
                    size=2,
                    overlap=0,
                    overview=2,
                )
                result = saved["result.json"]
                self.assertEqual(code, 1)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["expected_windows"], 3)
                self.assertEqual(len(requests), 2)
                self.assertEqual(
                    [item["status"] for item in result["parsed_windows"]], ["complete", status]
                )
                self.assertEqual(len(result["technical_windows"]), 2)
                self.assertIn("window_result_000_private.json", saved)
                self.assertIn("window_result_001_private.json", saved)
                self.assertNotIn("request_002.json", saved)
                self.assertNotIn("candidate", result)

    def test_uncertain_or_truncated_window_never_becomes_negative(self):
        replies = [
            (model_response("uncertain", complete=False), "uncertain"),
            (model_response(finish="length"), "truncated"),
        ]
        for failed_reply, status in replies:
            with self.subTest(status=status):
                code, saved, _ = synthetic_worker(
                    [0.0, 1.23456],
                    replies=[model_response("violation", timestamp=1.235), failed_reply],
                    size=2,
                    overlap=0,
                )
                result = saved["result.json"]
                self.assertEqual(code, 0)
                self.assertEqual(result["aggregation"]["status"], "incomplete")
                self.assertIsNone(result["aggregation"]["prediction"])
                self.assertEqual(result["aggregation"]["window_status_counts"][status], 1)
                self.assertEqual(result["candidate"]["verdict"], "uncertain")
                self.assertFalse(result["candidate"]["analysis_complete"])


if __name__ == "__main__":
    unittest.main()

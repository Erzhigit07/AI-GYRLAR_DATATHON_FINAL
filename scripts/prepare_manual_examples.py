"""Build four honestly labelled demonstration examples from the user's annotations."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forte.evidence import build_result_with_context  # noqa: E402
from src.forte.explanations import build_explanation  # noqa: E402

SELECTED = ("X-20", "X-33", "X-40", "X-46")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def remux(source, destination):
    import av

    with av.open(str(source)) as origin, av.open(str(destination), "w", format="mov") as target:
        stream = origin.streams.video[0]
        output = target.add_stream_from_template(stream)
        for packet in origin.demux(stream):
            if packet.dts is not None:
                packet.stream = output
                target.mux(packet)


def prepare(dataset_path, catalog_path, output):
    dataset, catalog = read(dataset_path), read(catalog_path)
    if (
        dataset.get("label_authority") != "user_manual"
        or dataset.get("original_labels_used") is not False
    ):
        raise ValueError("Only the user's authoritative annotations may be used")
    rows = {row["video_id"]: row for row in dataset["videos"]}
    rules = {row["rule_id"]: row for row in catalog["detection_rules"]}
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "examples": []}
    report = []
    for key in SELECTED:
        row = rows[key]
        source = ROOT / row["media"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["media_sha256"]:
            raise ValueError("Source media hash mismatch")
        if row["label"] != 1 or len(row["event_anchors"]) != 1:
            raise ValueError("Selected demo needs exactly one confirmed event anchor")
        anchor = row["event_anchors"][0]
        rule = rules[anchor["type"]]
        # The user's clarification governs X-46: coffee alone is not the criterion.
        description = (
            "Занялась посторонним делом, когда деньги ещё не убраны в сейф."
            if key == "X-46"
            else row["human_comment"]
        )
        candidate = {
            "verdict": "violation",
            "analysis_complete": True,
            "uncertain": False,
            "violations": [{**anchor, "description": description}],
        }
        video = output / (key + ".mov")
        remux(source, video)
        result, contexts = build_result_with_context(video, candidate)
        explanation = build_explanation(candidate, result, contexts)
        reason = (
            "Нарушение и момент указаны пользователем при ручной разметке. "
            "Кадры извлечены из видео на этой секунде и рядом с ней. "
            "Это размеченный пример: модель не принимала это решение."
        )
        explanation.update(
            status="available",
            kind="human_annotation_with_video_evidence",
            result_source="human_annotation",
            model_generated=False,
            window_trace_available=False,
            unavailable_reason="manual_annotation_not_model_inference",
            rule_ids=[anchor["type"]],
            windows=[],
            triggered_window_ids=[],
            confidence=None,
            aggregation={
                "method": "manual_annotation",
                "verdict": "violation",
                "expected_windows": None,
                "completed_windows": None,
                "positive_windows": None,
                "negative_windows": None,
                "distinct_events": 1,
                "returned_events": 1,
                "omitted_events": 0,
                "reason_code": "human_confirmed_event",
                "explanation": reason,
            },
            limitations=[
                "Это ручная разметка, а не самостоятельное обнаружение моделью.",
                "Кадры подтверждают происхождение изображений, но не измеряют точность модели.",
                "Оценка уверенности и внутренние рассуждения модели отсутствуют.",
            ],
        )
        for event in explanation["events"]:
            event["description_source"] = "human_annotation"
            event["rule_title"] = rule["title"]
            event["annotated_timecode_sec"] = anchor["timecode_sec"]
            for frame in event["context_frames"]:
                (output / f"{key}-{frame['role']}.jpg").write_bytes(
                    base64.b64decode(frame["image_base64"], validate=True)
                )
        common = {
            "job_id": "manual-" + key,
            "example_id": key,
            "status": "done",
            "result_source": "human_annotation",
            "model_generated": False,
            "source_label": "Размечено человеком",
        }
        write(output / (key + ".result.json"), {**common, **result})
        write(output / (key + ".explanation.json"), {**common, "explanation": explanation})
        manifest["examples"].append(
            {
                "example_id": key,
                "video_id": key,
                "title": rule["title"],
                "timecode_sec": result["violations"][0]["timecode_sec"],
                "result_file": key + ".result.json",
                "explanation_file": key + ".explanation.json",
                "video_file": key + ".mov",
            }
        )
        report.append(
            {
                "video_id": key,
                "type": anchor["type"],
                "annotated_timecode_sec": anchor["timecode_sec"],
                "actual_frame_time_sec": result["violations"][0]["timecode_sec"],
                "context_timestamps_sec": [
                    f["timestamp_sec"] for f in explanation["events"][0]["context_frames"]
                ],
                "source_media_sha256": row["media_sha256"],
                "source": "human_annotation",
            }
        )
    write(output / "manifest.json", manifest)
    write(
        output / "verification.json",
        {
            "source": "human_annotation",
            "model_generated": False,
            "original_labels_used": False,
            "examples": report,
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        },
    )
    lines = ["ЧЕТЫРЕ ПРИМЕРА ПО РУЧНОЙ РАЗМЕТКЕ", "", "Это не предсказания модели.", ""]
    for example in manifest["examples"]:
        lines.append(
            f"{example['video_id']}: {example['title']} — {example['timecode_sec']:g} сек."
        )
    lines.extend(
        [
            "",
            "MOV — полное видео; JPG before/event/after — кадры до, в момент и после.",
            "JSON result/explanation — тот же материал для API и интерфейса.",
        ]
    )
    (output / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"examples": report, "manifest": str(output / "manifest.json")}, ensure_ascii=True
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=ROOT / "configs/forte_rules.manual_v1.json")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    prepare(arguments.dataset.resolve(), arguments.catalog.resolve(), arguments.output.resolve())

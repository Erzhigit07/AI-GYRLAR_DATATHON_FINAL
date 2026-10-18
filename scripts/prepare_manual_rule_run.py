"""Freeze a user-label-only diagnostic run and an expanded-rule service config."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forte.rule_catalog import catalog_rules, manual_prompt  # noqa: E402


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def prepare(*, labels_path, media_manifest, mapping_path, catalog_path, previous_service, output):
    labels, media, mapping, catalog = map(
        read, (labels_path, media_manifest, mapping_path, catalog_path)
    )
    if (
        labels.get("label_authority") != "user_manual"
        or labels.get("original_labels_for_training") is not False
    ):
        raise ValueError("Only authoritative human labels are allowed")
    source = {row["key"]: row for row in media["videos"]}
    ids = [rule["rule_id"] for rule in catalog["detection_rules"] if rule.get("enabled") is True]
    catalog_rules(catalog, ids)
    positive_ids = {row["video_id"] for row in labels["videos"] if row["label"] == 1}
    if set(mapping["positive_types"]) != positive_ids:
        raise ValueError("Every human-positive video needs an explicit type mapping")
    records, seen_hashes = [], set()
    for row in labels["videos"]:
        if (
            row["label_authority"] != "user_manual"
            or row["status"] != "labeled"
            or row["label"] not in (0, 1)
        ):
            raise ValueError("Unreviewed or nonmanual row")
        raw = source[row["project_key"]]
        if raw["source_sha256"] != row["source_sha256"] or row["source_sha256"] in seen_hashes:
            raise ValueError("Source identity mismatch or duplicate")
        seen_hashes.add(row["source_sha256"])
        types = mapping["positive_types"].get(row["video_id"], [])
        if len(set(types)) != len(types) or set(types) - set(ids):
            raise ValueError("Unknown or duplicate human type")
        path = (media_manifest.parent / raw["file"]).resolve()
        if not path.is_relative_to(media_manifest.parent.resolve()) or not path.is_file():
            raise ValueError("Media path escapes the dataset or is missing")
        records.append(
            {
                "video_id": row["video_id"],
                "project_key": row["project_key"],
                "source_sha256": row["source_sha256"],
                "media_sha256": sha(path),
                "media": path.relative_to(ROOT).as_posix(),
                "duration_sec": raw["duration_sec"],
                "previous_split": row["split"],
                "label": row["label"],
                "positive_types": types,
                "rule_labels": {
                    rule: (1 if rule in types else None) if row["label"] else 0 for rule in ids
                },
                "event_anchors": [
                    {"type": rule, "timecode_sec": stamp}
                    for rule in types
                    for stamp in row["timestamps_sec"]
                ],
                "human_comment": row["comment"],
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    prompt = output / "prompt.txt"
    prompt.write_text(manual_prompt(catalog), encoding="utf-8")
    write_new(output / "rules.json", catalog)
    old_service = read(previous_service)
    old_path = ROOT / old_service["experiment_config"]
    old = read(old_path)
    experiment = copy.deepcopy(old)
    for field in (
        "deployment_source_config",
        "deployment_source_config_sha256",
        "deployment_evaluation_dir",
        "smoke_keys",
        "smoke_selection",
    ):
        experiment.pop(field, None)
    experiment.update(
        {
            "experiment_id": output.name,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "prompt_file": prompt.relative_to(ROOT).as_posix(),
            "prompt_sha256": sha(prompt),
            "rules_file": (output / "rules.json").relative_to(ROOT).as_posix(),
            "rules_sha256": sha(output / "rules.json"),
            "rule_ids": ids,
            "split": "manual40_exposed_diagnostic",
            "expected_count": len(records),
            "predecessor": old["experiment_id"],
            "predecessor_config_sha256": sha(old_path),
            "manual_dataset_sha256": sha(labels_path),
            "notes": [
                "All criteria derived from authoritative human comments; "
                "original labels are never used.",
                "Weights, sampling, generation and aggregation unchanged; "
                "criteria/prompt expanded.",
                "All 40 clips previously exposed; previous holdout is not "
                "independent for this diagnostic.",
                "Unmentioned types in positive videos are UNKNOWN, never assumed negative.",
                "18 human-negative videos provide negative labels for all selected criteria.",
                "No human IDs, comments, labels or event anchors are sent with inference requests.",
                "Per-type metrics are diagnostic; official temporal metric is not evaluated.",
            ],
        }
    )
    for relative in ("experiments/forte_qwen_baseline.py", "src/forte/qwen_baseline.py"):
        origin = old_path.parent / "frozen_sources" / relative
        if not origin.is_file():
            origin = ROOT / relative
        if sha(origin) != old["code_sha256"][relative]:
            raise ValueError("Predecessor frozen code mismatch")
        target = output / "frozen_sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    write_new(output / "config.json", experiment)
    write_new(
        output / "service_config.json",
        {
            **old_service,
            "experiment_config": (output / "config.json").relative_to(ROOT).as_posix(),
        },
    )
    write_new(
        output / "typed_manual_labels.json",
        {
            "label_authority": "user_manual",
            "source_sha256": sha(labels_path),
            "mapping_sha256": sha(mapping_path),
            "original_labels_used": False,
            "type_mapping_authority": (
                "assistant mapping of explicit human comments, before inference"
            ),
            "unknown_policy": "unmentioned types in a positive video are not scored as negative",
            "rule_ids": ids,
            "videos": records,
        },
    )
    plan = {
        "status": "FROZEN_BEFORE_INFERENCE",
        "experiment_id": output.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "videos": len(records),
        "human_positive": sum(row["label"] for row in records),
        "positive_by_type": {
            rule: sum(rule in row["positive_types"] for row in records) for rule in ids
        },
        "config_sha256": sha(output / "config.json"),
        "typed_labels_sha256": sha(output / "typed_manual_labels.json"),
        "rules_sha256": experiment["rules_sha256"],
        "prompt_sha256": sha(prompt),
        "evaluation": (
            "one fixed configuration; all40; type presence per video; "
            "unknown pairs excluded; failures reported separately"
        ),
    }
    write_new(output / "evaluation_plan.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--media-manifest", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=ROOT / "configs/forte_rules.manual_v1.json")
    parser.add_argument("--service", type=Path, default=ROOT / "configs/forte_service.example.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = prepare(
        labels_path=args.labels.resolve(),
        media_manifest=args.media_manifest.resolve(),
        mapping_path=args.mapping.resolve(),
        catalog_path=args.catalog.resolve(),
        previous_service=args.service.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(plan, ensure_ascii=False))


if __name__ == "__main__":
    main()

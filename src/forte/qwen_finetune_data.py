"""Prepare a private, grouped binary Qwen pilot without touching holdout media.

Positive training windows use existing point annotations (weak supervision).
Validation scans complete clips without annotation-dependent window selection.
This does not create event types, regulatory explanations or negative labels for
unannotated portions of positive clips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def label_of(record: dict) -> int:
    labels = [a.get("class") for a in record["annotations"]]
    if not labels or any(type(x) is not int or x not in (0, 1) for x in labels):
        raise ValueError("Invalid binary annotation")
    if len(set(labels)) != 1:
        raise ValueError("Conflicting duplicate annotations")
    return labels[0]


def split_records(manifest: dict, seed: int = 2026) -> tuple[list, list]:
    """Stratify whole source groups; never inspect holdout labels or media."""
    development = [r for r in manifest["videos"] if r["split"] == "development"]
    keys = [r["key"] for r in development]
    hashes = [r["source_sha256"] for r in development]
    if len(keys) != len(set(keys)) or len(hashes) != len(set(hashes)):
        raise ValueError("Development manifest must contain one record per source group")
    holdout_hashes = {r["source_sha256"] for r in manifest["videos"] if r["split"] == "holdout"}
    if set(hashes) & holdout_hashes:
        raise ValueError("Source group overlaps holdout")
    rng = random.Random(seed)
    train, validation = [], []
    for label in (0, 1):
        group = sorted((r for r in development if label_of(r) == label), key=lambda r: r["key"])
        if len(group) < 5:
            raise ValueError("Need at least five source groups per class")
        rng.shuffle(group)
        count = max(1, round(len(group) * 0.25))
        validation.extend(group[:count])
        train.extend(group[count:])
    return sorted(train, key=lambda r: r["key"]), sorted(validation, key=lambda r: r["key"])


def full_windows(count: int, width: int = 16, stride: int = 8) -> list[list[int]]:
    if count < 1 or not 1 <= stride <= width:
        raise ValueError("Invalid window dimensions")
    if count <= width:
        return [list(range(count))]
    starts = list(range(0, count - width + 1, stride))
    if starts[-1] != count - width:
        starts.append(count - width)
    return [list(range(start, start + width)) for start in starts]


def training_windows(record: dict, frames: list[dict], width: int = 16) -> list[list[int]]:
    count = len(frames)
    if label_of(record) == 0:
        windows = full_windows(count, width)
        indices = sorted({round(i * (len(windows) - 1) / 2) for i in range(3)})
        return [windows[i] for i in indices]
    starts = set()
    for annotation in record["annotations"]:
        timestamp = annotation.get("timestamp_sec")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or not 0 <= timestamp < record["duration_sec"]
        ):
            raise ValueError("Positive training record needs an in-range event timestamp")
        nearest = min(range(count), key=lambda i: abs(frames[i]["pts_sec"] - timestamp))
        if abs(frames[nearest]["pts_sec"] - timestamp) > 0.55:
            raise ValueError("Annotated moment is absent from sampled timeline")
        for offset in (-4, 0, 4):
            starts.add(max(0, min(nearest - width // 2 + offset, count - width)))
    return [list(range(start, min(start + width, count))) for start in sorted(starts)]


def extract_frames(video: Path, directory: Path, step: float = 0.5) -> list[dict]:
    import av
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=False)
    records = []
    first_pts = None
    previous = -1.0
    next_target = 0.0
    with av.open(str(video), options={"protocol_whitelist": "file,pipe"}) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Expected exactly one video stream")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                raise ValueError("Invalid video frame")
            absolute = float(frame.pts * frame.time_base)
            first_pts = absolute if first_pts is None else first_pts
            timestamp = absolute - first_pts
            if not math.isfinite(timestamp) or timestamp <= previous:
                raise ValueError("Invalid PTS ordering")
            previous = timestamp
            if timestamp + 1e-7 < next_target:
                continue
            image = frame.to_image().convert("RGB")
            image.thumbnail((896, 896), Image.Resampling.LANCZOS)
            destination = directory / f"{len(records):05d}.jpg"
            image.save(destination, quality=90)
            records.append(
                {"file": destination.name, "pts_sec": timestamp, "sha256": sha256(destination)}
            )
            next_target += step
    if not records:
        raise ValueError("No frames decoded")
    return records


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(root: Path, output: Path, seed: int = 2026) -> dict:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    before = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train, validation = split_records(manifest, seed)
    output.mkdir(parents=True, exist_ok=False)
    samples = {"train": [], "validation": []}
    provenance = []
    for split, records in [("train", train), ("validation", validation)]:
        for record in records:
            video = (root / record["file"]).resolve()
            if not video.is_relative_to(root):
                raise ValueError("Video escapes data root")
            digest = sha256(video)
            frames = extract_frames(video, output / "frames" / record["key"])
            windows = (
                training_windows(record, frames) if split == "train" else full_windows(len(frames))
            )
            for index, window in enumerate(windows):
                samples[split].append(
                    {
                        "id": f"{record['key']}_w{index:03d}",
                        "video_key": record["key"],
                        "label": label_of(record),
                        "label_scope": "weak_window" if split == "train" else "video_only",
                        "images": [f"frames/{record['key']}/{frames[i]['file']}" for i in window],
                        "pts_sec": [frames[i]["pts_sec"] for i in window],
                    }
                )
            if digest != sha256(video):
                raise ValueError("Source video changed during preparation")
            provenance.append(
                {
                    "key": record["key"],
                    "split": split,
                    "source_sha256": record["source_sha256"],
                    "media_sha256": digest,
                    "frames": frames,
                }
            )
            print(
                json.dumps(
                    {
                        "prepared": len(provenance),
                        "total": len(train) + len(validation),
                        "split": split,
                        "frames": len(frames),
                    }
                ),
                flush=True,
            )
    for split, rows in samples.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if sha256(manifest_path) != before:
        raise ValueError("Source manifest changed")
    write_json(output / "provenance.json", provenance)
    summary = {
        "status": "PREPARED_NOT_TRAINED",
        "seed": seed,
        "manifest_sha256": before,
        "train_videos": len(train),
        "validation_videos": len(validation),
        "train_class_counts": dict(Counter(str(label_of(r)) for r in train)),
        "validation_class_counts": dict(Counter(str(label_of(r)) for r in validation)),
        "train_windows": len(samples["train"]),
        "validation_windows": len(samples["validation"]),
        "holdout_media_read": False,
        "model_weights_trained": False,
        "sampling": {"step_sec": 0.5, "frames_per_window": 16, "validation_stride": 8},
        "limitations": [
            "Weak positive windows derived from point annotations; event type is unknown.",
            "No negative labels assigned to unannotated windows from positive training clips.",
            "Validation labels are video-level ONLY, never used as per-window targets.",
            "Related cameras/sessions have not been manually grouped; exact duplicates excluded.",
            "Development previously inspected; validation is internal, not an independent test.",
            "Binary pilot cannot replace the API event explanation contract.",
        ],
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.output, args.seed), ensure_ascii=False))


if __name__ == "__main__":
    main()

import pytest

from src.forte.qwen_finetune_data import full_windows, split_records, training_windows


def fixture_manifest():
    return {
        "videos": [
            {
                "key": str(i),
                "source_sha256": str(i),
                "split": "development",
                "duration_sec": 30,
                "annotations": [{"class": i % 2, "timestamp_sec": 15}],
            }
            for i in range(32)
        ]
        + [{"key": "holdout", "source_sha256": "sealed", "split": "holdout"}]
    }


def test_split_is_reproducible_and_holdout_labels_are_not_needed():
    manifest = fixture_manifest()
    train, validation = split_records(manifest)
    assert (train, validation) == split_records(manifest)
    assert len(train) == 24 and len(validation) == 8
    assert {r["key"] for r in train}.isdisjoint(r["key"] for r in validation)
    assert "holdout" not in {r["key"] for r in train + validation}
    assert {a["class"] for r in validation for a in r["annotations"]} == {0, 1}


def test_duplicate_sources_and_holdout_overlap_are_rejected():
    manifest = fixture_manifest()
    manifest["videos"][0]["source_sha256"] = "sealed"
    with pytest.raises(ValueError, match="holdout"):
        split_records(manifest)
    manifest["videos"][0]["source_sha256"] = "1"
    with pytest.raises(ValueError, match="one record"):
        split_records(manifest)


@pytest.mark.parametrize("count", [1, 7, 16, 17, 31, 32, 33, 286])
def test_validation_windows_cover_complete_timeline(count):
    windows = full_windows(count)
    assert {i for window in windows for i in window} == set(range(count))
    assert all(0 < len(window) <= 16 for window in windows)
    assert len({tuple(window) for window in windows}) == len(windows)


def test_positive_windows_contain_annotated_moment():
    record = fixture_manifest()["videos"][1]
    frames = [{"pts_sec": i / 2} for i in range(60)]
    windows = training_windows(record, frames)
    assert len(windows) == 3
    assert all(30 in window for window in windows)
    record["annotations"][0]["timestamp_sec"] = 31
    with pytest.raises(ValueError, match="in-range"):
        training_windows(record, frames)

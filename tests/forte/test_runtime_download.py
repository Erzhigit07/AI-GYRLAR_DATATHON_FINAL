"""The bootstrap must reject corrupt weights and ZIP traversal before use."""

import hashlib
import zipfile

import pytest

from scripts.download_forte_runtime import safe_extract, verified


def test_corrupt_cached_weight_is_not_accepted(tmp_path):
    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"bad")
    expected = {"size": 3, "sha256": hashlib.sha256(b"new").hexdigest()}
    assert not verified(weight, expected)
    weight.write_bytes(b"new")
    assert verified(weight, expected)


@pytest.mark.parametrize("member", ["../escape", "/absolute", "C:/escape", "..\\escape"])
def test_archive_paths_checked_before_any_extraction(tmp_path, member):
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("safe.dll", b"synthetic")
        zipped.writestr(member, b"synthetic")
    target = tmp_path / "runtime"
    with pytest.raises(ValueError):
        safe_extract(archive, target)
    assert not target.exists()

"""Synthetic fixtures, never competition data."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_data
from src.quality import quality_report, save_quality_report


@pytest.mark.parametrize("suffix", ["csv", "xlsx", "parquet"])
def test_read_formats_preserves_source_and_rows(tmp_path, suffix):
    original = pd.DataFrame(
        {"id": [1, 2, 3], "value": [1.5, np.nan, 3.5], "city": ["Алматы", "Астана", "Шымкент"]}
    )
    path = tmp_path / f"synthetic_test_only.{suffix}"
    writer = {"csv": original.to_csv, "xlsx": original.to_excel, "parquet": original.to_parquet}
    writer[suffix](path, index=False)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = load_data(path)
    pd.testing.assert_frame_equal(actual, original)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_encoding_and_csv_options(tmp_path):
    path = tmp_path / "synthetic_test_only.csv"
    content = "id;город\n001;Москва\n002;Казань\n"
    path.write_text(content, encoding="cp1251")
    result = load_data(path, encoding="cp1251", sep=";", dtype={"id": "string"})
    assert result["id"].tolist() == ["001", "002"]
    assert result["город"].tolist() == ["Москва", "Казань"]
    path.write_text(content, encoding="utf-16")
    with pytest.warns(UserWarning, match="кодировка"):
        result = load_data(path, sep=";")
    assert result["город"].tolist() == ["Москва", "Казань"]


def test_loader_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="не найден"):
        load_data(tmp_path / "absent.csv")
    path = tmp_path / "unsupported.txt"
    path.write_text("not data")
    with pytest.raises(ValueError, match="не поддерживается"):
        load_data(path)
    path = tmp_path / "synthetic_test_only.csv"
    path.write_text('a,b\n1,"unfinished')
    with pytest.raises(ValueError, match="Не удалось прочитать"):
        load_data(path)
    with pytest.raises(ValueError, match="Пропуск"):
        load_data(path, on_bad_lines="skip")
    path.write_text("a,b\n")
    with pytest.raises(ValueError, match="не содержит строк"):
        load_data(path)
    broken_xlsx = tmp_path / "broken.xlsx"
    broken_xlsx.write_bytes(b"not an Excel archive")
    with pytest.raises(ValueError, match="Не удалось прочитать"):
        load_data(broken_xlsx)


def test_quality_signals_do_not_clean_data(tmp_path):
    frame = pd.DataFrame(
        {"value": [1, 1, 2, 3, 4, 5, 1000, np.inf, np.nan], "constant": ["same"] * 9}
    )
    original = frame.copy(deep=True)
    report = quality_report(frame)
    assert (report["rows"], report["columns"]) == (9, 2)
    assert report["duplicate_rows"] == 1
    assert report["missing"]["value"] == 1
    assert report["flags"]["value"]["infinite"] == 1
    assert report["flags"]["value"]["iqr_outliers"] == 1
    assert report["flags"]["constant"]["constant"]
    destination = save_quality_report(frame, tmp_path / "quality")
    assert (destination / "summary.md").is_file()
    assert (destination / "quality.json").is_file()
    with pytest.raises(FileExistsError):
        save_quality_report(frame, destination)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("options", [{}, {"index_col": None}, {"index_col": False}])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_csv_rejects_excess_fields_without_changing_source(tmp_path, options, encoding):
    path = tmp_path / "synthetic_excess_fields.csv"
    path.write_text("id,value\n1,10,999\n2,20,888\n", encoding=encoding)
    before = path.read_bytes()
    if encoding == "utf-16":
        with pytest.warns(UserWarning, match="кодировка"):
            with pytest.raises(ValueError, match="Не удалось прочитать"):
                load_data(path, **options)
    else:
        with pytest.raises(ValueError, match="Не удалось прочитать"):
            load_data(path, **options)
    assert path.read_bytes() == before


def test_csv_explicit_index_and_quoted_fields_still_work(tmp_path):
    path = tmp_path / "synthetic_explicit_index.csv"
    path.write_text(
        'id,value\n001,"a,b"\n002,"line one\nline two"\n', encoding="utf-8", newline="\n"
    )
    before = path.read_bytes()
    plain = load_data(path, dtype={"id": "string"})
    assert plain.id.tolist() == ["001", "002"]
    assert plain.value.tolist() == ["a,b", "line one\nline two"]
    for index_col in [0, "id"]:
        indexed = load_data(path, index_col=index_col, dtype={"id": "string"})
        assert indexed.index.tolist() == ["001", "002"]
        assert indexed.value.tolist() == plain.value.tolist()
    pd.testing.assert_frame_equal(load_data(path, index_col=False, dtype={"id": "string"}), plain)
    assert path.read_bytes() == before

"""Descriptive checks. Flags are hypotheses for inspection, never automatic cleaning."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def quality_report(frame: pd.DataFrame) -> dict:
    if not frame.columns.is_unique:
        raise ValueError("Для отчёта нужны уникальные названия столбцов.")
    numeric = frame.select_dtypes(include="number")
    flags = {}
    for name in frame.columns:
        column = frame[name]
        entry = {
            "constant": bool(column.nunique(dropna=True) <= 1),
            "all_missing": bool(column.isna().all()),
        }
        if name in numeric:
            finite = column.replace([np.inf, -np.inf], np.nan).dropna()
            q1, q3 = finite.quantile([0.25, 0.75]) if len(finite) else (np.nan, np.nan)
            iqr = q3 - q1
            entry["infinite"] = int(np.isinf(column.to_numpy(dtype=float, na_value=np.nan)).sum())
            entry["iqr_outliers"] = (
                int(((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
            )
            entry["iqr_check_applicable"] = bool(iqr > 0)
        else:
            entry["high_cardinality"] = bool(len(frame) and column.nunique() / len(frame) > 0.9)
        flags[str(name)] = entry
    # Infinite values are counted above; finite-only moments avoid undefined mean/std.
    summary_frame = frame.copy()
    for name in numeric:
        summary_frame[name] = numeric[name].replace([np.inf, -np.inf], np.nan)
    statistics = (
        summary_frame.describe(include="all").to_json(date_format="iso") if frame.shape[1] else "{}"
    )
    return {
        "rows": len(frame),
        "columns": frame.shape[1],
        "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
        "missing": {str(c): int(n) for c, n in frame.isna().sum().items()},
        "missing_fraction": {str(c): float(n) for c, n in frame.isna().mean().fillna(0).items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "statistics": json.loads(statistics),
        "flags": flags,
        "limitations": "Статистика числовых столбцов рассчитана без inf. "
        "IQR/константность/кардинальность — сигналы для проверки, не доказательство "
        "ошибки. Утечки и бизнес-ограничения требуют знания задачи. Данные не изменены.",
    }


def save_quality_report(frame: pd.DataFrame, directory: str | Path) -> Path:
    """Use a new directory so reports and source files are never overwritten."""
    report = quality_report(frame)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    flagged = [
        name
        for name, checks in report["flags"].items()
        if any(
            checks.get(key)
            for key in ["constant", "all_missing", "infinite", "iqr_outliers", "high_cardinality"]
        )
    ]
    summary = (
        f"# Качество данных\n\nСтрок: {report['rows']}; столбцов: {report['columns']}.\n\n"
        f"Пропусков: {int(frame.isna().sum().sum())}. "
        f"Дубликатов строк: {report['duplicate_rows']}.\n\n"
        f"Столбцы для дополнительной проверки: {', '.join(flagged) or 'не отмечены'}.\n\n"
        f"{report['limitations']}\n"
    )
    (directory / "summary.md").write_text(summary, encoding="utf-8")
    return directory

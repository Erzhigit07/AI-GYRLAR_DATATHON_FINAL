"""Local plots; no sampling, dropping rows or inferred target semantics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_eda(
    frame: pd.DataFrame,
    directory: str | Path,
    *,
    target: str | None = None,
    max_columns: int = 20,
    top_categories: int = 20,
) -> list[Path]:
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Нужна непустая таблица с уникальными названиями столбцов.")
    if target is not None and target not in frame:
        raise ValueError(f"Целевая переменная отсутствует: {target}")
    if max_columns < 1 or top_categories < 1:
        raise ValueError("Лимиты графиков должны быть положительными.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    paths = []
    names = list(frame.columns[:max_columns])
    if target is not None and target not in names:
        names.append(target)
    for index, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(8, 4))
        column = frame[name]
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            values = column.replace([np.inf, -np.inf], np.nan).dropna()
            ax.hist(values, bins=30)
            ax.set_ylabel("Число строк (без NA/inf)")
        else:
            counts = column.astype("string").value_counts()
            missing_count = int(column.isna().sum())
            if missing_count:
                missing_label = "<MISSING>"
                suffix = 1
                while missing_label in counts.index:
                    missing_label = f"<MISSING> ({suffix})"
                    suffix += 1
                counts.loc[missing_label] = missing_count
                counts = counts.sort_values(ascending=False, kind="stable")
            counts.head(top_categories).plot.bar(ax=ax)
            ax.set_ylabel(f"Число строк; показано top {top_categories} из {len(counts)}")
        ax.set_title(f"{'Target: ' if name == target else ''}{name}")
        fig.tight_layout()
        path = directory / f"distribution_{index:03d}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    # Only non-constant numeric variables, with at least two paired observations.
    numeric = frame[names].select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] >= 2:
        corr = numeric.corr(min_periods=2)
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
        ax.set_yticks(range(len(corr)), corr.columns)
        ax.set_title("Pearson: только числовые столбцы; корреляция не означает причинность")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        path = directory / "correlations.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths

"""Inference API; competition-specific submission formatting must be implemented later."""

import joblib
import numpy as np
import pandas as pd


def predict_frame(artifact_path, frame: pd.DataFrame) -> pd.DataFrame:
    """Load only trusted joblib files: unpickling can execute code."""
    artifact = joblib.load(artifact_path)
    config = artifact["config"]
    required = config["features"] + ([config["id_column"]] if config["id_column"] else [])
    missing = [column for column in required if column not in frame]
    if missing or not frame.columns.is_unique or frame.empty:
        raise ValueError(f"Проверьте схему/число строк test; отсутствуют столбцы: {missing}")
    predicted = artifact["pipeline"].predict(frame[config["features"]])
    if len(predicted) != len(frame) or pd.isna(predicted).any():
        raise ValueError("Некорректное число предсказаний или пропуски в предсказаниях.")
    if config["task"] == "regression" and not np.isfinite(predicted).all():
        raise ValueError("Предсказания регрессии содержат inf.")
    result = pd.DataFrame({config["target"]: predicted}, index=frame.index)
    if config["id_column"]:
        ids = frame[config["id_column"]]
        if ids.isna().any() or not ids.is_unique:
            raise ValueError("ID в test должен быть непустым и уникальным.")
        result.insert(0, config["id_column"], ids)
    return result

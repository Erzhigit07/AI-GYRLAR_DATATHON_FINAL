"""Holdout indices with strict chronological or group boundaries."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import Config


def validate_frame(frame: pd.DataFrame, config: Config) -> None:
    required = [
        config.target,
        *config.features,
        config.id_column,
        config.group_column,
        config.time_column,
    ]
    missing = [c for c in required if c is not None and c not in frame]
    if missing:
        raise ValueError(f"В данных отсутствуют столбцы: {missing}")
    if not frame.columns.is_unique or len(frame) < 3:
        raise ValueError("Нужны уникальные названия столбцов и минимум 3 строки.")
    target = frame[config.target]
    if target.isna().any():
        raise ValueError("Target содержит пропуски: задайте обоснованную политику до обучения.")
    if config.id_column:
        ids = frame[config.id_column]
        if ids.isna().any() or not ids.is_unique:
            raise ValueError("ID должен быть непустым и уникальным.")
    if config.task == "classification":
        if target.nunique() < 2:
            raise ValueError("Для классификации нужны минимум два класса.")
        if config.average == "binary" and (
            target.nunique() != 2 or config.positive_label not in set(target)
        ):
            raise ValueError("binary average требует два класса и существующий positive_label.")
    elif not pd.api.types.is_numeric_dtype(target) or not np.isfinite(target).all():
        raise ValueError("Target регрессии должен содержать конечные числовые значения.")
    if config.target_transform == "log" and (target <= 0).any():
        raise ValueError("Для log-преобразования target должен быть строго положительным.")


def split_indices(frame: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    if config.validation == "group":
        groups = frame[config.group_column]
        if groups.isna().any() or groups.nunique() < 2:
            raise ValueError("Нужны минимум две группы без пропусков.")
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=config.validation_size, random_state=config.seed
        )
        train, valid = next(splitter.split(frame, groups=groups))
    elif config.validation == "time":
        values = frame[config.time_column]
        if pd.api.types.is_numeric_dtype(values):
            raise ValueError("Преобразуйте числовое время в datetime с явным указанием единицы.")
        times = pd.to_datetime(values, errors="raise", utc=True)
        if times.isna().any() or times.nunique() < 2:
            raise ValueError("Нужны минимум два различных времени без пропусков.")
        cut = min(len(frame) - 1, max(1, int(len(frame) * (1 - config.validation_size))))
        boundary = times.sort_values().iloc[cut]
        train, valid = (
            indices[(times < boundary).to_numpy()],
            indices[(times >= boundary).to_numpy()],
        )
    elif config.validation == "random":
        train, valid = train_test_split(
            indices,
            test_size=config.validation_size,
            random_state=config.seed,
            stratify=frame[config.target] if config.task == "classification" else None,
        )
    else:
        raise ValueError("Стратегия validation должна быть указана явно.")
    if not len(train) or not len(valid):
        raise ValueError("Разбиение создало пустую часть: пересмотрите validation_size.")
    if config.task == "classification":
        train_classes = set(frame.iloc[train][config.target])
        if len(train_classes) < 2 or not set(frame.iloc[valid][config.target]) <= train_classes:
            raise ValueError(
                "В train должны присутствовать все классы validation и минимум 2 класса."
            )
    return train, valid

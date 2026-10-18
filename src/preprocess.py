"""Imputation, scaling and encoding are fitted only through an sklearn Pipeline."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import Config


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize nullable/mixed categories while preserving every row and missing value."""
    result = frame.copy()
    for name in result:
        column = result[name]
        if pd.api.types.is_datetime64_any_dtype(column):
            raise ValueError(
                f"Признак {name}: для datetime сначала задайте осмысленное преобразование."
            )
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            result[name] = column.to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(result[name]).any():
                raise ValueError(f"Признак {name} содержит inf; проверьте источник данных.")
        else:
            result[name] = column.astype("string").astype(object).where(column.notna(), np.nan)
    return result


def build_pipeline(config: Config, name: str) -> Pipeline:
    parameters = dict(config.model_params.get(name, {}))
    classification = config.task == "classification"
    if name == "baseline":
        estimator = (
            DummyClassifier(strategy="most_frequent") if classification else DummyRegressor()
        )
    elif name == "linear":
        estimator = (
            LogisticRegression(**{"max_iter": 1000, **parameters, "random_state": config.seed})
            if classification
            else LinearRegression(**parameters)
        )
    elif name == "ridge":
        if classification:
            raise ValueError("Ridge поддерживает только регрессию.")
        estimator = Ridge(
            **{"alpha": 1.0, "solver": "lsqr", **parameters, "random_state": config.seed}
        )
    elif name == "random_forest":
        model = RandomForestClassifier if classification else RandomForestRegressor
        estimator = model(
            **{"n_estimators": 100, "n_jobs": 1, **parameters, "random_state": config.seed}
        )
    elif name == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as error:
            raise ImportError("Для CatBoost: pip install -r requirements-catboost.txt") from error
        model = CatBoostClassifier if classification else CatBoostRegressor
        estimator = model(
            **{
                "iterations": 100,
                "thread_count": 1,
                **parameters,
                "random_seed": config.seed,
                "verbose": False,
                "allow_writing_files": False,
            }
        )
    else:
        raise ValueError(f"Неизвестная модель: {name}")
    if config.target_transform == "log":
        if classification:
            raise ValueError("Log-преобразование поддерживает только регрессию.")
        estimator = TransformedTargetRegressor(
            regressor=estimator, func=np.log, inverse_func=np.exp
        )
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    columns = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_exclude="number")),
        ]
    )
    return Pipeline(
        [
            ("normalize", FunctionTransformer(normalize_features, validate=False)),
            ("preprocess", columns),
            ("model", estimator),
        ]
    )

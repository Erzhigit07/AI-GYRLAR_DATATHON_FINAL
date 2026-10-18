"""Only explicitly selected task-compatible metrics are computed."""

import numpy as np
from sklearn import metrics

from src.config import Config


def log_rmse(actual, predicted) -> float:
    """RMSE of natural logs, deliberately NOT RMSLE/log1p. No clipping invalid values."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if (
        not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
        or (actual <= 0).any()
        or (predicted <= 0).any()
    ):
        raise ValueError("log_rmse требует строго положительные конечные цены и предсказания.")
    return float(metrics.root_mean_squared_error(np.log(actual), np.log(predicted)))


def evaluate(actual, predicted, config: Config, *, labels=None) -> dict[str, float]:
    config.validate()
    if len(actual) != len(predicted):
        raise ValueError("Число предсказаний не совпадает с числом строк.")
    result = {}
    for name in config.metrics:
        if name == "accuracy":
            value = metrics.accuracy_score(actual, predicted)
        elif name in {"precision", "recall", "f1"}:
            options = {"average": config.average, "zero_division": "warn"}
            if config.average == "binary":
                options["pos_label"] = config.positive_label
            else:
                options["labels"] = labels
            value = getattr(metrics, f"{name}_score")(actual, predicted, **options)
        elif name == "mae":
            value = metrics.mean_absolute_error(actual, predicted)
        elif name == "rmse":
            value = metrics.root_mean_squared_error(actual, predicted)
        elif name == "log_rmse":
            value = log_rmse(actual, predicted)
        elif name == "r2":
            if len(actual) < 2 or np.ptp(np.asarray(actual, dtype=float)) == 0:
                raise ValueError("R² не определён для <2 строк или постоянного target validation.")
            value = metrics.r2_score(actual, predicted, force_finite=False)
        else:
            raise ValueError(f"Неподдерживаемая метрика: {name}")
        if not np.isfinite(value):
            raise ValueError(f"Метрика {name} не является конечным числом.")
        result[name] = float(value)
    return result

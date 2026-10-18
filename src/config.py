"""Explicit task and validation settings; no inferred competition assumptions."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CLASSIFICATION_METRICS = {"accuracy", "precision", "recall", "f1"}
REGRESSION_METRICS = {"mae", "rmse", "r2", "log_rmse"}


@dataclass
class Config:
    data_path: str | None = None
    dataset_label: str | None = None
    task: str | None = None
    target: str | None = None
    features: list[str] = field(default_factory=list)
    id_column: str | None = None
    validation: str | None = None
    group_column: str | None = None
    time_column: str | None = None
    validation_size: float = 0.2
    seed: int = 42
    metrics: list[str] = field(default_factory=list)
    average: str | None = None
    positive_label: str | int | float | None = None
    models: list[str] = field(default_factory=lambda: ["linear"])
    model_params: dict = field(default_factory=dict)
    read_options: dict = field(default_factory=dict)
    output_dir: str = "outputs/experiments"
    data_kind: str = "real"
    target_transform: str = "none"

    def validate(self):
        if self.task not in {"classification", "regression"}:
            raise ValueError("Укажите task: classification или regression после изучения задания.")
        if self.target_transform not in {"none", "log"}:
            raise ValueError("target_transform должен быть none или log.")
        if self.task != "regression" and self.target_transform != "none":
            raise ValueError("Преобразование target доступно только для регрессии.")
        if not self.data_path or not self.target or not self.dataset_label:
            raise ValueError("Укажите data_path, target и dataset_label (источник/версия данных).")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("Укажите непустой список уникальных features после проверки утечек.")
        excluded = {self.target, self.id_column, self.group_column, self.time_column}
        if excluded.intersection(self.features):
            raise ValueError("Target, ID, group и time запрещены в списке features.")
        if self.target in {self.id_column, self.group_column, self.time_column}:
            raise ValueError("Target не должен одновременно быть ID, group или time.")
        if self.validation not in {"random", "group", "time"}:
            raise ValueError("Выберите validation: random, group или time по структуре данных.")
        if self.group_column and self.time_column:
            raise ValueError(
                "Группы вместе со временем требуют специального split, пока не реализован."
            )
        if (self.validation == "group") != bool(self.group_column):
            raise ValueError("group_column должен быть задан только для group validation.")
        if (self.validation == "time") != bool(self.time_column):
            raise ValueError("time_column должен быть задан только для time validation.")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size должен быть между 0 и 1.")
        allowed = CLASSIFICATION_METRICS if self.task == "classification" else REGRESSION_METRICS
        if not self.metrics or not set(self.metrics) <= allowed:
            raise ValueError(f"Выберите применимые к задаче metrics из {sorted(allowed)}.")
        if set(self.metrics) & {"precision", "recall", "f1"}:
            if self.average not in {"binary", "macro", "micro", "weighted"}:
                raise ValueError("Для precision/recall/f1 задайте average явно.")
            if self.average == "binary" and self.positive_label is None:
                raise ValueError("Для binary average задайте positive_label явно.")
        if not set(self.models) <= {"linear", "ridge", "random_forest", "catboost"}:
            raise ValueError("Поддерживаются linear, ridge, random_forest и catboost.")
        if self.task != "regression" and "ridge" in self.models:
            raise ValueError("ridge доступен только для регрессии.")
        if len(self.models) != len(set(self.models)):
            raise ValueError("Список models содержит дубликаты.")
        if not set(self.model_params) <= set(self.models):
            raise ValueError("model_params содержит параметры невыбранной модели.")
        if self.data_kind not in {"real", "synthetic_test"}:
            raise ValueError("data_kind должен быть real или synthetic_test.")

    @classmethod
    def load(cls, path: str | Path):
        config = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        config.validate()
        return config

    def as_dict(self):
        return asdict(self)

"""SYNTHETIC TEST DATA ONLY. All files and experiment artifacts stay in pytest tmp_path."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config


@pytest.fixture
def synthetic_config(tmp_path):
    def create(task="classification", classes=2):
        rng = np.random.default_rng(42)
        n = 90
        values = rng.normal(size=n)
        frame = pd.DataFrame(
            {
                "id": np.arange(n),
                "value": values,
                "category": pd.Series(np.where(values > 0, "A", "B"), dtype="string"),
                "empty": np.nan,
            }
        )
        frame["target"] = np.arange(n) % classes if task == "classification" else values * 3
        frame.loc[::11, "value"] = np.nan
        frame.loc[::13, "category"] = pd.NA
        path = tmp_path / "synthetic_test_only.csv"
        frame.to_csv(path, index=False)
        config = Config(
            data_path=str(path),
            dataset_label="SYNTHETIC TEST ONLY seed=42",
            task=task,
            target="target",
            features=["value", "category", "empty"],
            id_column="id",
            validation="random",
            metrics=["accuracy"] if task == "classification" else ["mae", "rmse", "r2"],
            models=["linear", "random_forest"],
            model_params={"random_forest": {"n_estimators": 8}},
            output_dir=str(tmp_path / "synthetic_test_runs"),
            data_kind="synthetic_test",
        )
        return config, frame

    return create

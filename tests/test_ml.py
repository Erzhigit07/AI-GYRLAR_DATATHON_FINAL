"""ML smoke tests use only clearly labelled synthetic data in temporary directories."""

import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.evaluate import evaluate
from src.predict import predict_frame
from src.preprocess import build_pipeline
from src.train import file_hash, train
from src.validation import split_indices, validate_frame


@pytest.mark.parametrize(
    "task,classes", [("classification", 2), ("classification", 3), ("regression", 2)]
)
def test_training_prediction_serialization(synthetic_config, task, classes):
    config, frame = synthetic_config(task, classes)
    before = file_hash(config.data_path)
    run = train(config, refit="linear")
    assert file_hash(config.data_path) == before
    result = pd.read_csv(run / "metrics.csv")
    assert set(result["model"]) == {"baseline", "linear", "random_forest"}
    assert set(result["metric"]) == set(config.metrics)
    assert np.isfinite(result["value"]).all()
    assert result.loc[result.model == "baseline", "improvement"].eq(0).all()
    assert set(result.data_kind) == {"synthetic_test"}
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["train_rows"] + metadata["validation_rows"] == len(frame)
    assert metadata["data_sha256"] == before
    assignments = pd.read_csv(run / "split.csv")
    assert assignments.row_position.is_unique and len(assignments) == len(frame)
    test = frame.drop(columns="target").iloc[::-1].copy()
    test["category"] = "unseen-test-category"
    for model in ["baseline", "linear", "random_forest", "refit"]:
        predicted = predict_frame(run / f"{model}.joblib", test)
        assert len(predicted) == len(test)
        assert predicted.index.equals(test.index)
        assert predicted.id.tolist() == test.id.tolist()
        assert not predicted.isna().any().any()
        if task == "classification":
            assert set(predicted.target) <= set(frame.target)
        else:
            assert np.isfinite(predicted.target).all()
    with pytest.raises(ValueError, match="схему"):
        predict_frame(run / "linear.joblib", test.drop(columns="value"))


def test_learned_preprocessing_has_no_validation_leakage(synthetic_config):
    config, _ = synthetic_config()
    x = pd.DataFrame(
        {
            "n": [1.0, 3.0, np.nan, 5.0],
            "c": pd.Series(["a", "a", None, "a"], dtype="string"),
            "bool": pd.Series([True, False, None, True], dtype="boolean"),
            "empty": np.nan,
        }
    )
    pipeline = build_pipeline(config, "linear").fit(x, [0, 1, 0, 1])
    transforms = pipeline.named_steps["preprocess"].named_transformers_
    assert transforms["numeric"].named_steps["impute"].statistics_[0] == 3
    assert transforms["categorical"].named_steps["encode"].categories_[0].tolist() == ["a"]
    test = x.copy()
    test["n"] = 1e9
    test["c"] = "validation-only"
    assert pipeline[:-1].transform(test).shape[0] == len(test)
    assert len(pipeline.predict(test)) == len(test)
    assert transforms["numeric"].named_steps["impute"].statistics_[0] == 3
    assert transforms["categorical"].named_steps["encode"].categories_[0].tolist() == ["a"]


@pytest.mark.parametrize("strategy", ["group", "time"])
def test_validation_boundaries(synthetic_config, strategy):
    config, frame = synthetic_config()
    config.validation = strategy
    if strategy == "group":
        config.group_column = "group"
        frame["group"] = np.repeat(np.arange(15), 6)
    else:
        config.time_column = "time"
        frame["time"] = np.repeat(pd.date_range("2020-01-01", periods=15), 6)
    config.validate()
    validate_frame(frame, config)
    training, validation = split_indices(frame, config)
    assert set(training).isdisjoint(validation)
    assert len(training) + len(validation) == len(frame)
    if strategy == "group":
        assert set(frame.iloc[training].group).isdisjoint(frame.iloc[validation].group)
    else:
        assert frame.iloc[training].time.max() < frame.iloc[validation].time.min()


def test_invalid_config_and_data(synthetic_config):
    with pytest.raises(ValueError, match="task"):
        Config.load("configs/template.json")
    config, frame = synthetic_config()
    config.features.append("target")
    with pytest.raises(ValueError, match="запрещены"):
        config.validate()
    config.features.remove("target")
    config.metrics = ["rmse"]
    with pytest.raises(ValueError, match="metrics"):
        config.validate()
    config.metrics = ["accuracy"]
    frame.loc[0, "target"] = np.nan
    with pytest.raises(ValueError, match="пропуски"):
        validate_frame(frame, config)
    config.group_column = "group"
    with pytest.raises(ValueError, match="group_column"):
        config.validate()


@pytest.mark.parametrize(
    "average,actual,predicted",
    [
        ("binary", [0, 0, 1, 1], [0, 1, 1, 1]),
        ("macro", [0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 2, 1]),
    ],
)
def test_explicit_classification_metrics(synthetic_config, average, actual, predicted):
    config, _ = synthetic_config()
    config.metrics = ["accuracy", "precision", "recall", "f1"]
    config.average = average
    config.positive_label = 1 if average == "binary" else None
    scores = evaluate(actual, predicted, config, labels=np.unique(actual))
    assert set(scores) == set(config.metrics)
    if average == "binary":
        assert scores == pytest.approx(
            {"accuracy": 0.75, "precision": 2 / 3, "recall": 1, "f1": 0.8}
        )


def test_regression_metric_validity(synthetic_config):
    config, _ = synthetic_config("regression")
    scores = evaluate([1, 2, 3], [1, 2, 3], config)
    assert scores == {"mae": 0, "rmse": 0, "r2": 1}
    with pytest.raises(ValueError, match="R²"):
        evaluate([1, 1], [1, 1], config)


def test_reproducible_runs(synthetic_config):
    config, frame = synthetic_config()
    a, b = train(config), train(config)
    assert a != b
    scores_a = pd.read_csv(a / "metrics.csv").drop(columns="fit_seconds")
    scores_b = pd.read_csv(b / "metrics.csv").drop(columns="fit_seconds")
    pd.testing.assert_frame_equal(scores_a, scores_b)
    pd.testing.assert_frame_equal(
        predict_frame(a / "linear.joblib", frame), predict_frame(b / "linear.joblib", frame)
    )
    assert joblib.load(a / "linear.joblib")["fit_scope"] == "train_split"


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_optional_catboost(synthetic_config, task):
    pytest.importorskip(
        "catboost", reason="Install requirements-catboost.txt to test optional model"
    )
    config, frame = synthetic_config(task)
    config.models = ["catboost"]
    config.model_params = {"catboost": {"iterations": 3, "depth": 2}}
    run = train(config)
    assert len(predict_frame(run / "catboost.joblib", frame)) == len(frame)


def test_cli_runs_only_with_explicit_config(synthetic_config, tmp_path):
    config, _ = synthetic_config()
    config.models = []
    config.model_params = {}
    path = tmp_path / "synthetic_test_config.json"
    path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "--config", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stderr
    assert len(list((tmp_path / "synthetic_test_runs").glob("*/metrics.csv"))) == 1
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "--config", "configs/template.json"],
        capture_output=True,
    )
    assert result.returncode == 2


def test_empty_categorical_feature_and_unseen_value(synthetic_config):
    config, _ = synthetic_config()
    x = pd.DataFrame(
        {"empty_category": pd.Series([None] * 4, dtype="string"), "number": [1, 2, 3, 4]}
    )
    pipeline = build_pipeline(config, "linear").fit(x, [0, 1, 0, 1])
    x["empty_category"] = "unseen"
    assert len(pipeline.predict(x)) == 4

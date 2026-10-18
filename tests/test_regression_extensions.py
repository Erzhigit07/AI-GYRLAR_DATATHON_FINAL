"""Small labelled synthetic unit fixtures, never House Prices experimental data."""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.evaluate import evaluate, log_rmse
from src.predict import predict_frame
from src.preprocess import build_pipeline
from src.train import train
from src.validation import validate_frame


def test_log_rmse_is_natural_log_not_log1p():
    actual = np.array([1.0, np.e, np.e**2])
    predicted = actual * np.e
    assert log_rmse(actual, predicted) == pytest.approx(1.0)
    assert log_rmse(actual, actual) == 0
    assert log_rmse(actual, predicted) != pytest.approx(
        np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2))
    )


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_log_metric_rejects_invalid_prices(bad):
    with pytest.raises(ValueError, match="положительные"):
        log_rmse([1, 2], [bad, 2])
    with pytest.raises(ValueError, match="положительные"):
        log_rmse([bad, 2], [1, 2])


def test_log_ridge_roundtrip_and_geometric_baseline(synthetic_config):
    config, frame = synthetic_config("regression")
    frame["target"] = np.exp(np.linspace(2, 5, len(frame)))
    frame.to_csv(config.data_path, index=False)
    config.models = ["ridge"]
    config.model_params = {"ridge": {"alpha": 10.0}}
    config.target_transform = "log"
    config.metrics = ["log_rmse", "mae", "rmse", "r2"]
    run = train(config, refit="ridge")
    result = predict_frame(run / "refit.joblib", frame.iloc[::-1])
    assert (result.target > 0).all() and np.isfinite(result.target).all()
    assert result.id.tolist() == frame.iloc[::-1].id.tolist()
    baseline = joblib.load(run / "baseline.joblib")["pipeline"]
    assignments = pd.read_csv(run / "split.csv")
    training = assignments.loc[assignments.split == "train", "row_position"]
    expected = np.exp(np.log(frame.iloc[training].target).mean())
    np.testing.assert_allclose(baseline.predict(frame[config.features]), expected)
    scores = pd.read_csv(run / "metrics.csv")
    row = scores[(scores.model == "ridge") & (scores.metric == "log_rmse")].iloc[0]
    assert row.improvement == pytest.approx(row.baseline_value - row.value)
    assert json.loads((run / "metadata.json").read_text())["target_transform"] == "log"


def test_regression_options_not_allowed_for_classification(synthetic_config):
    config, _ = synthetic_config()
    config.target_transform = "log"
    with pytest.raises(ValueError, match="регрессии"):
        config.validate()
    config.target_transform = "none"
    config.models = ["ridge"]
    with pytest.raises(ValueError, match="регрессии"):
        config.validate()


def test_nonpositive_training_target_rejected(synthetic_config):
    config, frame = synthetic_config("regression")
    config.target_transform = "log"
    with pytest.raises(ValueError, match="положительным"):
        validate_frame(frame, config)


def test_mean_baseline_is_unchanged_by_default(synthetic_config):
    config, frame = synthetic_config("regression")
    model = build_pipeline(config, "baseline").fit(frame[config.features], frame.target)
    np.testing.assert_allclose(model.predict(frame[config.features]), frame.target.mean())


def test_undefined_precision_warning_is_saved(synthetic_config):
    config, _ = synthetic_config()
    config.models, config.model_params = [], {}
    config.metrics, config.average, config.positive_label = ["precision"], "binary", 1
    with pytest.warns(UndefinedMetricWarning):
        run = train(config)
    log = json.loads((run / "warnings.json").read_text())
    assert log[0]["model"] == "baseline" and log[0]["category"] == "UndefinedMetricWarning"


def test_read_options_preserve_meaningful_absence_tokens(tmp_path):
    path = tmp_path / "synthetic_semantics_test.csv"
    path.write_text("feature,area,material\nNA,NA,None\nGravel,10,Stone\n", encoding="utf-8")
    frame = load_data(path, keep_default_na=False, na_values={"area": ["NA"]})
    assert frame.feature.iloc[0] == "NA"
    assert frame.material.iloc[0] == "None"
    assert pd.isna(frame.area.iloc[0])


def test_log_metric_is_opt_in(synthetic_config):
    config, _ = synthetic_config("regression")
    scores = evaluate([1, 2, 3], [1, 2, 3], config)
    assert "log_rmse" not in scores


def test_refit_warning_is_saved_and_emitted(synthetic_config, monkeypatch):
    config, frame = synthetic_config("regression")
    config.models, config.model_params = [], {}
    original_fit = Pipeline.fit

    def fit_with_refit_warning(self, x, y=None, **params):
        if "model" in self.named_steps and len(x) == len(frame):
            warnings.warn("SYNTHETIC refit-only warning", UserWarning, stacklevel=2)
        return original_fit(self, x, y, **params)

    monkeypatch.setattr(Pipeline, "fit", fit_with_refit_warning)
    with pytest.warns(UserWarning, match="SYNTHETIC refit-only warning"):
        run = train(config, refit="baseline")
    log = json.loads((run / "warnings.json").read_text(encoding="utf-8"))
    assert log == [
        {
            "model": "baseline",
            "phase": "refit",
            "category": "UserWarning",
            "message": "SYNTHETIC refit-only warning",
        }
    ]
    assert (run / "refit.joblib").is_file()

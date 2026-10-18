"""Local real-data House Prices rehearsal. No network calls or synthetic data.

Selection: minimum mean 5-fold log-RMSE inside the training partition only.
Outer holdout: shuffled 80/20 split, seed 42, identical across all models.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.audit_titanic import audit_titanic
from experiments.titanic_rehearsal import copy_original, save_json, verify_fitted_preprocessing
from src.config import Config
from src.data_loader import load_data
from src.eda import plot_eda
from src.evaluate import evaluate, log_rmse
from src.predict import predict_frame
from src.preprocess import build_pipeline, normalize_features
from src.quality import save_quality_report
from src.train import file_hash, train
from src.validation import split_indices, validate_frame


def cross_validate(frame, positions, config, directory):
    subset = frame.iloc[positions]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=config.seed).split(subset))
    records, predictions = (
        [],
        {"Id": subset.Id.to_numpy(), "SalePrice": subset.SalePrice.to_numpy()},
    )
    assignments = np.full(len(subset), -1, dtype=int)
    for fold, (_, valid) in enumerate(folds):
        assignments[valid] = fold
    assert (assignments >= 0).all()
    pd.DataFrame({"row_position": positions, "fold": assignments}).to_csv(
        directory / "inner_cv_assignments.csv", index=False
    )
    for name in ["baseline", *config.models]:
        oof = np.full(len(subset), np.nan)
        for fold, (training, valid) in enumerate(folds):
            assert set(training).isdisjoint(valid)
            pipeline = build_pipeline(config, name)
            start = perf_counter()
            pipeline.fit(subset.iloc[training][config.features], subset.iloc[training].SalePrice)
            oof[valid] = pipeline.predict(subset.iloc[valid][config.features])
            value = log_rmse(subset.iloc[valid].SalePrice, oof[valid])
            records.append(
                {
                    "model": name,
                    "fold": fold,
                    "train_rows": len(training),
                    "validation_rows": len(valid),
                    "log_rmse": value,
                    "fit_predict_seconds": perf_counter() - start,
                }
            )
            print(f"CV {name}, fold {fold + 1}/5: log-RMSE={value:.6f}", flush=True)
        assert np.isfinite(oof).all() and (oof > 0).all()
        predictions[name] = oof
    scores = pd.DataFrame(records)
    scores.to_csv(directory / "cv_folds.csv", index=False)
    pd.DataFrame(predictions).to_csv(directory / "cv_oof_predictions.csv", index=False)
    summary = scores.groupby("model").agg(
        mean_log_rmse=("log_rmse", "mean"), std_log_rmse=("log_rmse", "std")
    )
    summary.to_csv(directory / "cv_summary.csv")
    selected = summary.mean_log_rmse.idxmin()
    save_json(
        directory / "selection_before_holdout.json",
        {
            "selected": selected,
            "rule": "minimum mean log-RMSE in 5 inner folds",
            "outer_holdout_used_for_selection": False,
            "hyperparameter_search": False,
        },
    )
    return selected


def analyze_models(run, config, frame, official_test, training, valid, directory):
    truth = frame.iloc[valid].SalePrice.to_numpy()
    saved = pd.read_csv(run / "validation_predictions.csv")
    np.testing.assert_array_equal(saved.row_position, valid)
    np.testing.assert_array_equal(saved.actual, truth)
    records, subgroup_records, checks = [], [], {}
    boundaries = frame.iloc[training].SalePrice.quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy()
    boundaries[0], boundaries[-1] = -np.inf, np.inf
    validation = frame.iloc[valid].copy()
    validation["PriceBand"] = pd.cut(
        validation.SalePrice, boundaries, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    for name in ["baseline", *config.models]:
        pipeline = joblib.load(run / f"{name}.joblib")["pipeline"]
        checks[name] = verify_fitted_preprocessing(pipeline, frame.iloc[training][config.features])
        output = predict_frame(run / f"{name}.joblib", validation)
        pred = output.SalePrice.to_numpy()
        np.testing.assert_allclose(pred, saved[name], rtol=1e-12)
        assert output.Id.tolist() == validation.Id.tolist()
        # Independent calculation: no use of evaluate/log_rmse to check official score.
        independent = float(np.sqrt(np.mean((np.log(truth) - np.log(pred)) ** 2)))
        metrics = evaluate(truth, pred, config)
        np.testing.assert_allclose(metrics["log_rmse"], independent, rtol=1e-12)
        training_pred = pipeline.predict(frame.iloc[training][config.features])
        records.append(
            {
                "model": name,
                **metrics,
                "train_log_rmse": log_rmse(frame.iloc[training].SalePrice, training_pred),
                "mean_log_bias": float(np.mean(np.log(pred) - np.log(truth))),
                "median_absolute_error": float(np.median(np.abs(pred - truth))),
            }
        )
        detail = validation.copy()
        detail["prediction"] = pred
        detail["log_error"] = np.log(pred) - np.log(truth)
        detail["absolute_error"] = np.abs(pred - truth)
        detail["relative_error"] = pred / truth - 1
        detail.to_csv(directory / f"{name}_holdout_errors.csv", index=False)
        detail.assign(abs_log_error=detail.log_error.abs()).nlargest(15, "abs_log_error").to_csv(
            directory / f"{name}_largest_errors.csv", index=False
        )
        for feature in ["PriceBand", "Neighborhood", "OverallQual", "YrSold"]:
            for group, part in detail.groupby(feature, observed=True, dropna=False):
                subgroup_records.append(
                    {
                        "model": name,
                        "feature": feature,
                        "group": str(group),
                        "rows": len(part),
                        "log_rmse": float(np.sqrt(np.mean(part.log_error**2))),
                        "mae": float(part.absolute_error.mean()),
                        "log_bias": float(part.log_error.mean()),
                    }
                )
        forecast = predict_frame(run / f"{name}.joblib", official_test)
        assert len(forecast) == len(official_test)
        assert forecast.Id.tolist() == official_test.Id.tolist()
        assert np.isfinite(forecast.SalePrice).all() and (forecast.SalePrice > 0).all()
        forecast.to_csv(directory / f"{name}_test_predictions.csv", index=False)
        checks[name].update(
            {
                "serialization_matches": True,
                "independent_log_metric_matches": True,
                "test_rows": len(forecast),
                "test_positive_and_finite": True,
            }
        )
    pd.DataFrame(records).to_csv(directory / "holdout_summary.csv", index=False)
    pd.DataFrame(subgroup_records).to_csv(directory / "subgroup_errors.csv", index=False)
    save_json(directory / "model_checks.json", checks)
    return records


def target_plots(frame, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(frame.SalePrice, bins=40)
    axes[0].set_title("SalePrice: training partition only")
    axes[1].hist(np.log(frame.SalePrice), bins=40)
    axes[1].set_title("Natural log(SalePrice): training partition only")
    fig.tight_layout()
    fig.savefig(directory / "target_distributions.png", dpi=130)
    plt.close(fig)


def run_rehearsal(source, titanic_result):
    source = Path(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = Path("outputs/educational/house_prices/rehearsal") / stamp
    directory.mkdir(parents=True, exist_ok=False)
    print(f"ARTIFACT_DIR={directory}", flush=True)
    save_json(directory / "titanic_audit.json", audit_titanic(titanic_result))
    original_hashes = {}
    for name in ["train.csv", "test.csv", "data_description.txt", "sample_submission.csv"]:
        destination = Path("data/raw/house_prices") / name
        original_hashes[name] = copy_original(source / name, destination)
    save_json(
        directory / "provenance.json",
        {
            "source_directory": str(source.resolve()),
            "sha256": original_hashes,
            "source_url": "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
            "status": "User-provided originals; upstream hashes not independently authenticated",
            "purpose": "EDUCATIONAL ONLY — not DATATHON results",
        },
    )
    config_path = Path("configs/educational/house_prices.json")
    config = Config.load(config_path)
    frame = load_data(config.data_path, **config.read_options)
    official_test = load_data("data/raw/house_prices/test.csv", **config.read_options)
    validate_frame(frame, config)
    assert "SalePrice" not in official_test
    assert set(frame.Id).isdisjoint(official_test.Id) and official_test.Id.is_unique
    assert set(config.features) == set(frame.columns) - {"Id", "SalePrice"}
    assert set(official_test.columns) == set(frame.columns) - {"SalePrice"}
    assert not frame[config.features].duplicated().any()
    training, valid = split_indices(frame, config)
    assert set(training).isdisjoint(valid)
    # Data quality checks across files; targeted EDA uses training partition only.
    save_quality_report(frame, directory / "quality_train")
    save_quality_report(official_test, directory / "quality_test")
    training_frame = frame.iloc[training]
    save_quality_report(training_frame, directory / "quality_training_partition")
    plots = [
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "GarageCars",
        "YearBuilt",
        "LotArea",
        "Neighborhood",
        "MSSubClass",
        "SaleCondition",
        "SalePrice",
    ]
    plot_eda(training_frame[plots], directory / "figures", target="SalePrice")
    target_plots(training_frame, directory)
    normalized_train = normalize_features(frame.iloc[training][config.features])
    normalized_validation = normalize_features(frame.iloc[valid][config.features])
    normalized_test = normalize_features(official_test[config.features])
    unknown = {
        name: {
            "validation": sorted(
                set(normalized_validation[name].dropna()) - set(normalized_train[name].dropna())
            ),
            "test": sorted(
                set(normalized_test[name].dropna()) - set(normalized_train[name].dropna())
            ),
        }
        for name in normalized_train.select_dtypes(exclude="number")
    }
    shared_feature_rows = len(frame[config.features].merge(official_test[config.features]))
    audit = {
        "train_rows": len(frame),
        "test_rows": len(official_test),
        "train_columns": len(frame.columns),
        "fit_rows": len(training),
        "holdout_rows": len(valid),
        "id_overlap": 0,
        "duplicate_train_feature_rows": 0,
        "matching_train_test_feature_pairs": shared_feature_rows,
        "unknown_categories": unknown,
        "target_statistics": frame.SalePrice.describe().to_dict(),
        "training_target_skew": float(training_frame.SalePrice.skew()),
        "training_log_target_skew": float(np.log(training_frame.SalePrice).skew()),
        "train_years": frame.YrSold.value_counts().sort_index().to_dict(),
        "test_years": official_test.YrSold.value_counts().sort_index().to_dict(),
        "validation_rationale": "New rows from same Ames market/years; not future forecasting. "
        "No property/group ID supplied. No feature duplicates in labelled rows. "
        "Spatial/latent property dependence cannot be fully ruled out.",
        "target_and_id_excluded": True,
        "sample_submission_labels_used": False,
    }
    save_json(directory / "data_audit.json", audit)
    selected = cross_validate(frame, training, config, directory)
    print(f"Selected before holdout: {selected}", flush=True)
    # Exercise the public CLI, including full-data refit and normal-scale serialization.
    execution = subprocess.run(
        [sys.executable, "-m", "src.train", "--config", str(config_path), "--refit", selected],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    (directory / "cli_stdout.txt").write_text(execution.stdout, encoding="utf-8")
    (directory / "cli_stderr.txt").write_text(execution.stderr, encoding="utf-8")
    if execution.returncode:
        raise RuntimeError(f"CLI failed: {execution.stderr}")
    run = Path(execution.stdout.strip().splitlines()[-1].split(": ", 1)[1])
    model_dir = directory / "models_analysis"
    model_dir.mkdir()
    summaries = analyze_models(run, config, frame, official_test, training, valid, model_dir)
    forecast = predict_frame(run / "refit.joblib", official_test)
    sample = load_data("data/raw/house_prices/sample_submission.csv")
    assert forecast.columns.tolist() == sample.columns.tolist() == ["Id", "SalePrice"]
    assert forecast.Id.tolist() == official_test.Id.tolist() == sample.Id.tolist()
    assert len(forecast) == len(official_test) and (forecast.SalePrice > 0).all()
    assert np.isfinite(forecast.SalePrice).all()
    forecast.to_csv(directory / "test_predictions_refit.csv", index=False)
    # Repeating the exact configuration verifies reproducibility; no parameter search.
    repeated = train(config, refit=selected)
    pd.testing.assert_frame_equal(
        pd.read_csv(run / "metrics.csv").drop(columns="fit_seconds"),
        pd.read_csv(repeated / "metrics.csv").drop(columns="fit_seconds"),
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(run / "validation_predictions.csv"),
        pd.read_csv(repeated / "validation_predictions.csv"),
    )
    np.testing.assert_allclose(
        predict_frame(repeated / "refit.joblib", official_test).SalePrice,
        forecast.SalePrice,
        rtol=1e-12,
    )
    # Diagnostic plots are after evaluation; not used to change models.
    saved = pd.read_csv(run / "validation_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(np.log(saved.actual), np.log(saved[selected]), s=12, alpha=0.65)
    limits = [np.log(saved.actual).min(), np.log(saved.actual).max()]
    axes[0].plot(limits, limits, "k--")
    axes[0].set(xlabel="Actual log(price)", ylabel="Predicted log(price)", title=selected)
    axes[1].scatter(saved.actual, np.log(saved[selected]) - np.log(saved.actual), s=12, alpha=0.65)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(xlabel="Actual SalePrice", ylabel="Log residual", title="Untuned holdout errors")
    fig.tight_layout()
    fig.savefig(directory / "holdout_errors.png", dpi=130)
    plt.close(fig)
    for name, fingerprint in original_hashes.items():
        assert (
            file_hash(source / name)
            == file_hash(Path("data/raw/house_prices") / name)
            == fingerprint
        )
    outcome = {
        "status": "COMPLETED",
        "purpose": "EDUCATIONAL ONLY",
        "directory": str(directory),
        "selected_by_inner_cv": selected,
        "primary_run": str(run),
        "repeat_run": str(repeated),
        "holdout_metrics": summaries,
        "cli_returncode": execution.returncode,
        "official_test_rows": len(forecast),
        "official_test_metrics": "UNKNOWN: no labels",
        "sample_submission_used_only_for_ids_and_columns": True,
        "hashes_unchanged": True,
        "reproducibility_passed": True,
    }
    save_json(directory / "result.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False, indent=2), flush=True)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--titanic-result", required=True)
    args = parser.parse_args()
    run_rehearsal(args.source_dir, args.titanic_result)

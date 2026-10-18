"""Run a baseline and selected candidates on the same holdout; never infer the task."""

import argparse
import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.compose import TransformedTargetRegressor

from src.config import Config
from src.data_loader import load_data
from src.evaluate import evaluate
from src.preprocess import build_pipeline
from src.validation import split_indices, validate_frame


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def train(config: Config, *, refit: str | None = None) -> Path:
    config.validate()
    names = ["baseline", *config.models]
    if refit is not None and refit not in names:
        raise ValueError("refit должен указывать явно выбранную модель или baseline.")
    fingerprint = file_hash(config.data_path)
    frame = load_data(config.data_path, **config.read_options)
    validate_frame(frame, config)
    train_idx, valid_idx = split_indices(frame, config)
    x, y = frame[config.features], frame[config.target]
    fitted, records, predictions, parameters = {}, [], {}, {}
    baseline_scores = None
    warning_log = []
    for name in names:
        pipeline = build_pipeline(config, name)
        start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
            duration = perf_counter() - start
            predicted = pipeline.predict(x.iloc[valid_idx])
            labels = pipeline.classes_ if config.task == "classification" else None
            scores = evaluate(y.iloc[valid_idx], predicted, config, labels=labels)
        for warning in caught:
            warning_log.append(
                {
                    "model": name,
                    "phase": "holdout",
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                }
            )
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
        if name == "baseline":
            baseline_scores = scores
        for metric, value in scores.items():
            direction = -1 if metric in {"mae", "rmse", "log_rmse"} else 1
            records.append(
                {
                    "model": name,
                    "metric": metric,
                    "value": value,
                    "baseline_value": baseline_scores[metric],
                    "improvement": direction * (value - baseline_scores[metric]),
                    "fit_seconds": duration,
                    "data_kind": config.data_kind,
                }
            )
        fitted[name] = pipeline
        predictions[name] = predicted
        estimator = pipeline.named_steps["model"]
        if isinstance(estimator, TransformedTargetRegressor):
            estimator = estimator.regressor_
        parameters[name] = estimator.get_params()
    final_pipeline = None
    refit_seconds = None
    if refit:
        start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            final_pipeline = build_pipeline(config, refit).fit(x, y)
        refit_seconds = perf_counter() - start
        for warning in caught:
            warning_log.append(
                {
                    "model": refit,
                    "phase": "refit",
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                }
            )
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
    if file_hash(config.data_path) != fingerprint:
        raise ValueError("Исходный файл изменился во время обучения; результаты не сохранены.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = Path(config.output_dir) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id,
        "data_sha256": fingerprint,
        "dataset_label": config.dataset_label,
        "data_kind": config.data_kind,
        "train_rows": len(train_idx),
        "validation_rows": len(valid_idx),
        "seed": config.seed,
        "validation": config.validation,
        "target_transform": config.target_transform,
        "preprocessing": "train-only: median + scaling; most-frequent + one-hot; unknown ignored",
        "parameters": parameters,
        "refit_model": refit,
        "refit_seconds": refit_seconds,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {d.metadata["Name"]: d.version for d in distributions()},
        "limitations": "Single holdout; not evidence of competition performance. "
        "Metrics describe holdout models, not optional refit model.",
    }
    for name, content in [
        ("config.json", config.as_dict()),
        ("metadata.json", metadata),
        ("warnings.json", warning_log),
    ]:
        (directory / name).write_text(
            json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    pd.DataFrame(records).to_csv(directory / "metrics.csv", index=False)
    assignments = pd.DataFrame({"row_position": range(len(frame)), "split": "train"})
    assignments.loc[valid_idx, "split"] = "validation"
    assignments.to_csv(directory / "split.csv", index=False)
    validation = pd.DataFrame(
        {"row_position": valid_idx, "actual": y.iloc[valid_idx].to_numpy(), **predictions}
    )
    validation.to_csv(directory / "validation_predictions.csv", index=False)
    for name, pipeline in fitted.items():
        joblib.dump(
            {"pipeline": pipeline, "config": config.as_dict(), "fit_scope": "train_split"},
            directory / f"{name}.joblib",
        )
    if final_pipeline is not None:
        joblib.dump(
            {"pipeline": final_pipeline, "config": config.as_dict(), "fit_scope": "all_rows"},
            directory / "refit.joblib",
        )
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--refit", help="Optional explicitly chosen model to fit on all rows")
    args = parser.parse_args()
    try:
        directory = train(Config.load(args.config), refit=args.refit)
    except (ValueError, OSError, ImportError, TypeError) as error:
        parser.exit(2, f"Ошибка: {error}\n")
    print(f"Результаты: {directory}")


if __name__ == "__main__":
    main()

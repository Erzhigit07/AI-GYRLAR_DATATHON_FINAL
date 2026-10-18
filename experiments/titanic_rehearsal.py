"""Real, user-provided Kaggle Titanic rehearsal. No generated data and no network calls.

Run from the repository root: python -m experiments.titanic_rehearsal --source-dir PATH
All row-level data, figures, models and computed reports remain under ignored outputs/.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

matplotlib.use("Agg")

from src.config import Config
from src.data_loader import load_data
from src.eda import plot_eda
from src.predict import predict_frame
from src.preprocess import normalize_features
from src.quality import save_quality_report
from src.train import file_hash, train
from src.validation import split_indices, validate_frame


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def copy_original(source, destination):
    """Copy byte-for-byte, never overwrite a different or existing source file."""
    fingerprint = file_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination) != fingerprint:
            raise ValueError(f"Existing dataset differs; refusing to overwrite {destination}")
    else:
        with destination.open("xb") as stream:
            stream.write(source.read_bytes())
    assert file_hash(destination) == fingerprint == file_hash(source)
    return fingerprint


def family_proxy(frame):
    """Surname + reported family size is an imperfect relation proxy, not a proven family ID."""
    size = frame.SibSp + frame.Parch + 1
    surname = frame.Name.str.split(",", n=1).str[0].str.strip().str.casefold()
    return (surname + "|" + size.astype(str)).where(
        size > 1, "single|" + frame.PassengerId.astype(str)
    )


def audit_split(frame, training, validation):
    left, right = frame.iloc[training], frame.iloc[validation]
    tickets = set(left.Ticket) & set(right.Ticket)
    families = family_proxy(frame)
    shared_families = set(families.iloc[training]) & set(families.iloc[validation])
    return {
        "train_rows": len(left),
        "validation_rows": len(right),
        "train_survival_rate": float(left.Survived.mean()),
        "validation_survival_rate": float(right.Survived.mean()),
        "shared_tickets": len(tickets),
        "validation_rows_with_shared_ticket": int(right.Ticket.isin(tickets).sum()),
        "shared_family_proxies": len(shared_families),
        "validation_rows_with_shared_family_proxy": int(
            families.iloc[validation].isin(shared_families).sum()
        ),
        "family_proxy_limitation": (
            "Surname + family size can join unrelated people and miss relatives."
        ),
    }


def verify_fitted_preprocessing(pipeline, training_features):
    normalized = normalize_features(training_features)
    numeric = normalized.select_dtypes(include="number")
    transformers = pipeline.named_steps["preprocess"].named_transformers_
    expected = numeric.median().fillna(0).to_numpy()
    actual = transformers["numeric"].named_steps["impute"].statistics_
    np.testing.assert_allclose(actual, expected)
    imputed = numeric.fillna(numeric.median()).fillna(0)
    np.testing.assert_allclose(transformers["numeric"].named_steps["scale"].mean_, imputed.mean())
    categorical = normalized.select_dtypes(exclude="number")
    encoder = transformers["categorical"].named_steps["encode"]
    for name, categories in zip(categorical.columns, encoder.categories_, strict=True):
        assert set(categories) == set(categorical[name].dropna())
    return {
        "numeric_medians_match_train": True,
        "scaler_means_match_train": True,
        "categories_match_train": True,
        "numeric_medians": dict(zip(numeric.columns, actual, strict=False)),
    }


def analyze_run(run, config, frame, official_test, directory):
    training, validation = split_indices(frame, config)
    truth = frame.iloc[validation].Survived.to_numpy()
    stored = pd.read_csv(run / "validation_predictions.csv")
    np.testing.assert_array_equal(stored.row_position, validation)
    np.testing.assert_array_equal(stored.actual, truth)
    result, subgroup_rows, checks = [], [], {}
    directory.mkdir()
    valid = frame.iloc[validation].copy()
    valid["AgeMissing"] = valid.Age.isna().map({True: "missing", False: "observed"})
    valid["AgeBand"] = pd.cut(
        valid.Age,
        [-np.inf, 12, 18, 40, 60, np.inf],
        labels=["<=12", "13-18", "19-40", "41-60", ">60"],
    )
    valid["AgeBand"] = valid.AgeBand.astype("string").fillna("missing")
    for name in ["baseline", *config.models]:
        artifact_path = run / f"{name}.joblib"
        pipeline = joblib.load(artifact_path)["pipeline"]
        checks[name] = verify_fitted_preprocessing(pipeline, frame.iloc[training][config.features])
        predicted = predict_frame(artifact_path, frame.iloc[validation])
        np.testing.assert_array_equal(predicted.Survived, stored[name])
        assert predicted.PassengerId.tolist() == frame.iloc[validation].PassengerId.tolist()
        train_predictions = pipeline.predict(frame.iloc[training][config.features])
        tn, fp, fn, tp = confusion_matrix(truth, predicted.Survived, labels=[0, 1]).ravel()
        result.append(
            {
                "model": name,
                "train_accuracy": float(
                    accuracy_score(frame.iloc[training].Survived, train_predictions)
                ),
                "validation_accuracy": float(accuracy_score(truth, predicted.Survived)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "errors": int(fp + fn),
            }
        )
        errors = valid.copy()
        errors["prediction"] = predicted.Survived.to_numpy()
        errors["error"] = errors.Survived != errors.prediction
        errors.loc[errors.error].to_csv(directory / f"{name}_error_cases.csv", index=False)
        for feature in ["Sex", "Pclass", "AgeMissing", "AgeBand"]:
            for group, part in errors.groupby(feature, dropna=False, observed=True):
                subgroup_rows.append(
                    {
                        "model": name,
                        "feature": feature,
                        "group": str(group),
                        "rows": len(part),
                        "errors": int(part.error.sum()),
                        "error_rate": float(part.error.mean()),
                        "actual_survival_rate": float(part.Survived.mean()),
                    }
                )
        test_predictions = predict_frame(artifact_path, official_test)
        assert len(test_predictions) == len(official_test)
        assert test_predictions.PassengerId.tolist() == official_test.PassengerId.tolist()
        assert set(test_predictions.Survived) <= {0, 1}
        test_predictions.to_csv(directory / f"{name}_test_predictions.csv", index=False)
        checks[name]["test_prediction_rows"] = len(test_predictions)
        checks[name]["test_ids_and_labels_valid"] = True
        checks[name]["serialization_matches_validation"] = True
    pd.DataFrame(result).to_csv(directory / "confusion_and_overfit.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(directory / "subgroup_errors.csv", index=False)
    save_json(directory / "checks.json", checks)
    return result


def run_rehearsal(source):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = Path("outputs/educational/titanic/rehearsal") / stamp
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(source)
    provenance = {
        "status": "user-provided Kaggle-format files; upstream checksum not independently verified",
        "source_page": "https://www.kaggle.com/competitions/titanic/data",
        "purpose": "EDUCATIONAL ONLY — not DATATHON results",
        "files": {},
    }
    for name in ["train.csv", "test.csv"]:
        destination = Path("data/raw/titanic") / name
        fingerprint = copy_original(source / name, destination)
        provenance["files"][name] = {
            "source_path": str((source / name).resolve()),
            "local_path": str(destination),
            "sha256": fingerprint,
        }
    save_json(directory / "provenance.json", provenance)
    configs = {
        name: Config.load(f"configs/educational/titanic_{name}.json")
        for name in ["group", "random"]
    }
    frame = load_data(configs["group"].data_path, **configs["group"].read_options)
    official_test = load_data("data/raw/titanic/test.csv", **configs["group"].read_options)
    validate_frame(frame, configs["group"])
    assert "Survived" not in official_test
    assert set(frame.PassengerId).isdisjoint(official_test.PassengerId)
    assert set(frame.Survived) == {0, 1}
    assert set(frame.columns) - {"Survived"} == set(official_test.columns)
    assert official_test.PassengerId.is_unique
    # Split before targeted EDA; feature list and model settings are already fixed.
    splits = {name: split_indices(frame, config) for name, config in configs.items()}
    audits = {name: audit_split(frame, *indices) for name, indices in splits.items()}
    assert audits["group"]["shared_tickets"] == 0
    save_json(directory / "leakage_audit.json", audits)
    save_quality_report(frame, directory / "quality_original_train")
    save_quality_report(official_test, directory / "quality_official_test")
    exploration = frame.iloc[splits["group"][0]]
    save_quality_report(exploration, directory / "quality_training_partition")
    plot_eda(
        exploration[[*configs["group"].features, "Survived"]],
        directory / "figures",
        target="Survived",
    )
    eda_rates = []
    for feature in ["Sex", "Pclass", "Embarked"]:
        for value, part in exploration.groupby(feature, dropna=False, observed=True):
            eda_rates.append(
                {
                    "feature": feature,
                    "value": str(value),
                    "rows": len(part),
                    "survival_rate": float(part.Survived.mean()),
                }
            )
    pd.DataFrame(eda_rates).to_csv(directory / "eda_train_only_rates.csv", index=False)
    summaries, all_metrics, warning_log = {}, [], []
    for name, config in configs.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run = train(config)
        for warning in caught:
            entry = {
                "split": name,
                "category": warning.category.__name__,
                "message": str(warning.message),
            }
            warning_log.append(entry)
            print(f"{name}: {entry['category']}: {entry['message']}", file=sys.stderr)
        metrics = pd.read_csv(run / "metrics.csv")
        metrics.insert(0, "split", name)
        metrics.insert(1, "run_id", run.name)
        all_metrics.append(metrics)
        summaries[name] = {
            "run_dir": str(run),
            "audit": audits[name],
            "errors": analyze_run(run, config, frame, official_test, directory / name),
        }
    pd.concat(all_metrics, ignore_index=True).to_csv(directory / "comparison.csv", index=False)
    save_json(directory / "warnings.json", warning_log)
    # A second identical run checks reproducibility, not an additional hyperparameter trial.
    with warnings.catch_warnings(record=True) as repeated_warnings:
        warnings.simplefilter("always")
        repeated = train(configs["group"])
    save_json(directory / "repeat_warnings.json", [str(w.message) for w in repeated_warnings])
    first = Path(summaries["group"]["run_dir"])
    pd.testing.assert_frame_equal(
        pd.read_csv(first / "metrics.csv").drop(columns="fit_seconds"),
        pd.read_csv(repeated / "metrics.csv").drop(columns="fit_seconds"),
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(first / "validation_predictions.csv"),
        pd.read_csv(repeated / "validation_predictions.csv"),
    )
    for record in provenance["files"].values():
        assert file_hash(record["source_path"]) == record["sha256"]
        assert file_hash(record["local_path"]) == record["sha256"]
    result = {
        "status": "COMPLETED",
        "purpose": "EDUCATIONAL ONLY",
        "summary_dir": str(directory),
        "runs": summaries,
        "repeat_run": str(repeated),
        "reproducible_group_run": True,
        "source_and_local_hashes_unchanged": True,
        "official_test_metrics": "NOT AVAILABLE — no ground truth; sample submission unused",
        "no_hyperparameter_search": True,
    }
    save_json(directory / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    arguments = parser.parse_args()
    run_rehearsal(arguments.source_dir)

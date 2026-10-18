"""Recompute stored real Titanic metrics independently, without training or mutation."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data_loader import load_data
from src.train import file_hash


def audit_titanic(result_path):
    result_path = Path(result_path)
    manifest = json.loads(result_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(result_path.parent / "comparison.csv")
    checked = []
    for split, entry in manifest["runs"].items():
        run = Path(entry["run_dir"])
        assert run.is_relative_to(Path("outputs/educational/titanic"))
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        assert "EDUCATIONAL ONLY" in config["dataset_label"]
        frame = load_data(config["data_path"], **config["read_options"])
        assert file_hash(config["data_path"]) == metadata["data_sha256"]
        predictions = pd.read_csv(run / "validation_predictions.csv")
        actual = frame.iloc[predictions.row_position].Survived.to_numpy()
        np.testing.assert_array_equal(actual, predictions.actual)
        scores = pd.read_csv(run / "metrics.csv")
        baseline = {}
        for name in ["baseline", *config["models"]]:
            pipeline = joblib.load(run / f"{name}.joblib")["pipeline"]
            calculated = pipeline.predict(frame.iloc[predictions.row_position][config["features"]])
            np.testing.assert_array_equal(calculated, predictions[name])
            independent = {
                "accuracy": accuracy_score(actual, calculated),
                "precision": precision_score(actual, calculated, zero_division=0),
                "recall": recall_score(actual, calculated, zero_division=0),
                "f1": f1_score(actual, calculated, zero_division=0),
            }
            if name == "baseline":
                baseline = independent
            for metric, value in independent.items():
                row = scores[(scores.model == name) & (scores.metric == metric)].iloc[0]
                joined = comparison[
                    (comparison.split == split)
                    & (comparison.model == name)
                    & (comparison.metric == metric)
                ].iloc[0]
                np.testing.assert_allclose([row.value, joined.value], value, atol=1e-12)
                np.testing.assert_allclose(row.improvement, value - baseline[metric], atol=1e-12)
                checked.append({"split": split, "model": name, "metric": metric, "value": value})
    assert not Path("reports/experiments.csv").exists(), "Review competition journal isolation"
    return {
        "status": "PASS",
        "metric_values_checked": len(checked),
        "serialized_predictions_match": True,
        "data_hashes_match": True,
        "educational_paths_and_labels_verified": True,
        "values": checked,
        "baseline_precision_note": (
            "Undefined; sklearn convention zero, warning recorded in rehearsal"
        ),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_path")
    args = parser.parse_args()
    print(json.dumps(audit_titanic(args.result_path), indent=2))

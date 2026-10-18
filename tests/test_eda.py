"""Notebook and plots tested exclusively on synthetic temporary fixtures."""

import sys
from pathlib import Path

import matplotlib
import nbformat
import pandas as pd
import pytest
from jupyter_client import KernelManager
from matplotlib.figure import Figure
from nbclient import NotebookClient

matplotlib.use("Agg")

from src.eda import plot_eda


def test_plots_and_target(tmp_path):
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "category": ["a", "b", "a"]})
    original = frame.copy(deep=True)
    paths = plot_eda(frame, tmp_path / "figures", target="category")
    assert len(paths) == 4
    assert all(path.stat().st_size > 0 for path in paths)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="отсутствует"):
        plot_eda(frame, tmp_path / "invalid", target="absent")


def test_plot_distinguishes_missing_from_literal_categories(tmp_path, monkeypatch):
    frame = pd.DataFrame({"category": ["<MISSING>", "<MISSING> (1)", None, None, "normal"]})
    original = frame.copy(deep=True)
    bars = {}
    original_savefig = Figure.savefig

    def capture_bars(self, *args, **kwargs):
        axis = self.axes[0]
        bars.update(
            (label.get_text(), bar.get_height())
            for label, bar in zip(axis.get_xticklabels(), axis.patches, strict=True)
        )
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_bars)
    paths = plot_eda(frame, tmp_path / "synthetic_missing_categories")
    assert bars["<MISSING>"] == 1
    assert bars["<MISSING> (1)"] == 1
    assert bars["normal"] == 1
    missing_labels = set(bars) - {"<MISSING>", "<MISSING> (1)", "normal"}
    assert len(missing_labels) == 1
    assert bars[missing_labels.pop()] == 2
    assert sum(bars.values()) == len(frame)
    assert paths[0].is_file()
    pd.testing.assert_frame_equal(frame, original)


def test_notebook_executes_on_synthetic_test_data(synthetic_config, tmp_path):
    config, _ = synthetic_config()
    notebook = nbformat.read("notebooks/eda.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("execution_count") is None for cell in notebook.cells)
    assert all(not cell.get("outputs") for cell in notebook.cells)
    for cell in notebook.cells:
        if "parameters" in cell.metadata.get("tags", []):
            cell.source += (
                f"\nDATA_PATH = {config.data_path!r}\nTARGET = 'target'\n"
                f"OUTPUT_DIR = Path({str(tmp_path / 'notebook_output')!r})\n"
            )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(
        notebook, km=manager, timeout=90, resources={"metadata": {"path": str(Path.cwd())}}
    )
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    assert (tmp_path / "notebook_output/quality/summary.md").is_file()
    assert list((tmp_path / "notebook_output/figures").glob("*.png"))

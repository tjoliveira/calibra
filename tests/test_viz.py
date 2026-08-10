"""Tests for llm_uq.viz — verify files are created and ax= parameter works."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests

import matplotlib.pyplot as plt
import pytest

from llm_uq import viz


@pytest.fixture(autouse=True)
def close_all():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Standalone (no ax) — check file is written
# ---------------------------------------------------------------------------


def test_roc_curve_saves_file(tmp_path):
    path = str(tmp_path / "roc.png")
    viz.roc_curve([0.1, 0.9], [0, 1], method_name="entropy", save_path=path)
    assert (tmp_path / "roc.png").exists()


def test_correlation_saves_file(tmp_path):
    path = str(tmp_path / "corr.png")
    viz.correlation([0.1, 0.9], [0.8, 0.2], method_name="entropy", save_path=path)
    assert (tmp_path / "corr.png").exists()


def test_calibration_curve_saves_file(tmp_path):
    path = str(tmp_path / "cal.png")
    viz.calibration_curve([0.1, 0.5, 0.9], [0.0, 0.5, 1.0], method_name="entropy",
                          ece=0.05, save_path=path)
    assert (tmp_path / "cal.png").exists()


def test_distribution_saves_file(tmp_path):
    path = str(tmp_path / "dist.png")
    viz.distribution([0.1, 0.5, 0.9], method_name="entropy", save_path=path)
    assert (tmp_path / "dist.png").exists()


def test_method_comparison_saves_file(tmp_path):
    path = str(tmp_path / "cmp.png")
    results = {"entropy": {"auroc": 0.8}, "max_probability": {"auroc": 0.6}}
    viz.method_comparison(results, metric="auroc", save_path=path)
    assert (tmp_path / "cmp.png").exists()


def test_token_heatmap_saves_file(tmp_path):
    path = str(tmp_path / "hm.png")
    tok_u = [[0.1, 0.2, 0.3], [0.4, 0.5], [0.6]]
    toks = [["a", "b", "c"], ["d", "e"], ["f"]]
    viz.token_heatmap(tok_u, toks, method_name="entropy", save_path=path)
    assert (tmp_path / "hm.png").exists()


# ---------------------------------------------------------------------------
# ax= parameter — function returns the axes without closing the figure
# ---------------------------------------------------------------------------


def test_roc_curve_returns_ax_when_given():
    fig, ax = plt.subplots()
    returned = viz.roc_curve([0.1, 0.9], [0, 1], ax=ax)
    assert returned is ax


def test_distribution_returns_ax_when_given():
    fig, ax = plt.subplots()
    returned = viz.distribution([0.1, 0.9], ax=ax)
    assert returned is ax


def test_calibration_curve_returns_ax_when_given():
    fig, ax = plt.subplots()
    returned = viz.calibration_curve([0.1, 0.5, 0.9], [0.0, 0.5, 1.0], ax=ax)
    assert returned is ax


# ---------------------------------------------------------------------------
# No global style mutation on import
# ---------------------------------------------------------------------------


def test_no_global_style_mutation_on_import():
    import matplotlib
    before = dict(matplotlib.rcParams)
    import importlib
    import llm_uq.viz
    importlib.reload(llm_uq.viz)
    after = dict(matplotlib.rcParams)
    # figure.figsize should not have been changed by the module reload
    assert before.get("figure.figsize") == after.get("figure.figsize")

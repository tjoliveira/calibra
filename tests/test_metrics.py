"""Tests for calibra.metrics — pure numeric functions, no mocking needed."""

import math
import numpy as np
import pytest

from calibra import metrics


def test_auroc_perfect_predictor():
    u = [0.1, 0.9]
    e = [0, 1]
    assert metrics.auroc(u, e) == pytest.approx(1.0)


def test_auroc_inverse_predictor():
    u = [0.9, 0.1]
    e = [0, 1]
    assert metrics.auroc(u, e) == pytest.approx(0.0)


def test_auroc_all_same_class_returns_nan():
    result = metrics.auroc([0.1, 0.2, 0.3], [0, 0, 0])
    assert math.isnan(result)


def test_auroc_handles_inf():
    result = metrics.auroc([float("inf"), 0.5], [1, 0])
    assert isinstance(result, float)
    assert not math.isnan(result)


def test_auroc_handles_nan():
    result = metrics.auroc([float("nan"), 0.5], [1, 0])
    assert isinstance(result, float)


def test_ece_returns_tuple():
    result = metrics.ece([0.1, 0.9], [0.0, 1.0])
    assert isinstance(result, tuple)
    score, stats = result
    assert isinstance(score, float)
    assert "bin_boundaries" in stats


def test_ece_shape_of_bin_stats():
    _, stats = metrics.ece(list(range(20)), [i % 2 for i in range(20)])
    assert len(stats["bin_accuracies"]) == 10
    assert len(stats["bin_confidences"]) == 10
    assert len(stats["bin_counts"]) == 10


def test_correlations_returns_pearson_and_spearman():
    u = [0.1, 0.5, 0.9]
    e = [0.0, 0.5, 1.0]
    result = metrics.correlations(u, e)
    assert "pearson" in result
    assert "spearman" in result
    coef, pval = result["pearson"]
    assert isinstance(coef, float)
    assert isinstance(pval, float)


def test_correlations_invalid_method_raises():
    with pytest.raises(ValueError):
        metrics.correlations([1.0], [1.0], method="invalid")


def test_correlations_pearson_only():
    result = metrics.correlations([0.1, 0.9], [0.0, 1.0], method="pearson")
    assert "pearson" in result
    assert "spearman" not in result


def test_summarize_returns_metric_summary():
    summary = metrics.summarize([0.1, 0.9], [0.0, 1.0])
    assert isinstance(summary.auroc, float)
    assert isinstance(summary.ece, float)
    assert isinstance(summary.error_rate, float)
    assert summary.total_samples == 2
    assert "mean" in summary.uncertainty_stats


def test_summarize_binary_skips_correlations():
    summary = metrics.summarize([0.1, 0.9], [0.0, 1.0])
    assert math.isnan(summary.pearson)
    assert math.isnan(summary.spearman)


def test_summarize_continuous_computes_correlations():
    u = [float(i) / 10 for i in range(10)]
    e = [float(i) / 10 for i in range(10)]
    summary = metrics.summarize(u, e)
    assert not math.isnan(summary.spearman)


def test_uncertainty_stats_keys():
    stats = metrics.uncertainty_stats([1.0, 2.0, 3.0])
    for key in ("mean", "std", "min", "max", "median", "q25", "q75"):
        assert key in stats
    assert stats["mean"] == pytest.approx(2.0)


def test_sanitise_inf_via_auroc():
    # Pass inf through auroc — should not raise
    result = metrics.auroc([float("inf"), float("inf"), 0.1], [1, 1, 0])
    assert isinstance(result, float)

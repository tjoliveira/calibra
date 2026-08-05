"""Evaluation metrics for uncertainty quantification."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitise(arr: np.ndarray) -> np.ndarray:
    """Replace non-finite values with finite equivalents in-place.

    Infinities are replaced by max-finite + 1; NaNs by the median of finite
    values. When *all* values are non-finite, everything becomes 1.0.
    """
    arr = arr.copy().astype(float)
    finite_mask = np.isfinite(arr)
    if np.all(finite_mask):
        return arr
    if np.any(finite_mask):
        max_finite = np.max(arr[finite_mask])
        arr[np.isinf(arr) & (arr > 0)] = max_finite + 1.0
        arr[np.isinf(arr) & (arr < 0)] = -(max_finite + 1.0)
        arr[np.isnan(arr)] = np.median(arr[finite_mask])
    else:
        arr[:] = 1.0
    return arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class MetricSummary:
    """Aggregated metrics returned by :func:`summarize`.

    Attributes:
        auroc: Area under the ROC curve (binary error prediction).
        ece: Expected Calibration Error.
        pearson: Pearson correlation between uncertainty and errors.
        spearman: Spearman correlation between uncertainty and errors.
        error_rate: Mean error rate across samples.
        total_samples: Number of evaluated samples.
        uncertainty_stats: Descriptive statistics for uncertainty values.
    """

    auroc: float
    ece: float
    pearson: float
    spearman: float
    error_rate: float
    total_samples: int
    uncertainty_stats: dict[str, float]


def auroc(uncertainties: list[float], errors: list[float]) -> float:
    """Compute AUROC with uncertainty as the predictor of errors.

    Args:
        uncertainties: Per-sample uncertainty scores (higher = more uncertain).
        errors: Per-sample error values.  For binary tasks use 0.0 (correct) /
            1.0 (incorrect); for continuous tasks any value is accepted and
            thresholded at 0.5.

    Returns:
        AUROC score in [0, 1], or ``float("nan")`` when it cannot be computed
        (e.g. all samples belong to the same class).
    """
    u = _sanitise(np.array(uncertainties, dtype=float))
    e = np.array(errors, dtype=float)
    binary_errors = (e > 0.5).astype(int)

    if len(np.unique(binary_errors)) < 2:
        logger.warning("AUROC: all samples in the same class — returning NaN.")
        return float("nan")

    try:
        return float(roc_auc_score(binary_errors, u))
    except Exception as exc:
        logger.warning("AUROC computation failed: %s", exc)
        return float("nan")


def ece(
    uncertainties: list[float], errors: list[float], n_bins: int = 10
) -> tuple[float, dict]:
    """Compute Expected Calibration Error.

    Args:
        uncertainties: Per-sample uncertainty scores.
        errors: Per-sample error values (0.0 = correct, 1.0 = incorrect; or
            continuous quality-error scores).
        n_bins: Number of equal-width confidence bins.

    Returns:
        A ``(ece_score, bin_stats)`` tuple where ``bin_stats`` is a dict with
        keys ``bin_boundaries``, ``bin_accuracies``, ``bin_confidences``, and
        ``bin_counts``.
    """
    u = _sanitise(np.array(uncertainties, dtype=float))
    e = np.array(errors, dtype=float)

    # Normalise uncertainties to confidence scores in [0, 1]
    u_min, u_max = u.min(), u.max()
    confidences = 1.0 - (u - u_min) / (u_max - u_min + 1e-10)

    boundaries = np.linspace(0, 1, n_bins + 1)
    ece_score = 0.0
    bin_stats: dict = {
        "bin_boundaries": boundaries.tolist(),
        "bin_accuracies": [],
        "bin_confidences": [],
        "bin_counts": [],
    }

    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_stats["bin_accuracies"].append(0.0)
            bin_stats["bin_confidences"].append(0.0)
            bin_stats["bin_counts"].append(0)
            continue

        acc = float(1.0 - e[mask].mean())
        conf = float(confidences[mask].mean())
        prop = mask.mean()
        ece_score += abs(conf - acc) * prop

        bin_stats["bin_accuracies"].append(acc)
        bin_stats["bin_confidences"].append(conf)
        bin_stats["bin_counts"].append(int(mask.sum()))

    return float(ece_score), bin_stats


def correlations(
    uncertainties: list[float],
    errors: list[float],
    method: str = "all",
) -> dict[str, tuple[float, float]]:
    """Compute correlation between uncertainty and errors.

    Args:
        uncertainties: Per-sample uncertainty scores.
        errors: Per-sample error values.
        method: Which correlation to compute: ``"pearson"``, ``"spearman"``,
            or ``"all"`` (default).

    Returns:
        Dict mapping correlation name to ``(coefficient, p_value)`` tuples.

    Raises:
        ValueError: If ``method`` is not one of the accepted values.
    """
    valid = {"all", "pearson", "spearman"}
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got {method!r}")

    u = _sanitise(np.array(uncertainties, dtype=float))
    e = np.array(errors, dtype=float)
    result: dict[str, tuple[float, float]] = {}

    if method in {"all", "pearson"}:
        try:
            coef, pval = pearsonr(u, e)
            result["pearson"] = (float(coef), float(pval))
        except Exception as exc:
            logger.warning("Pearson correlation failed: %s", exc)
            result["pearson"] = (float("nan"), float("nan"))

    if method in {"all", "spearman"}:
        try:
            coef, pval = spearmanr(u, e)
            result["spearman"] = (float(coef), float(pval))
        except Exception as exc:
            logger.warning("Spearman correlation failed: %s", exc)
            result["spearman"] = (float("nan"), float("nan"))

    return result


def uncertainty_stats(uncertainties: list[float]) -> dict[str, float]:
    """Compute descriptive statistics for uncertainty values.

    Args:
        uncertainties: Per-sample uncertainty scores.

    Returns:
        Dict with keys ``mean``, ``std``, ``min``, ``max``, ``median``,
        ``q25``, and ``q75``.
    """
    u = np.array(uncertainties, dtype=float)
    return {
        "mean": float(np.mean(u)),
        "std": float(np.std(u)),
        "min": float(np.min(u)),
        "max": float(np.max(u)),
        "median": float(np.median(u)),
        "q25": float(np.percentile(u, 25)),
        "q75": float(np.percentile(u, 75)),
    }


def summarize(
    uncertainties: list[float], errors: list[float], n_bins: int = 10
) -> MetricSummary:
    """Compute all metrics in one call.

    Args:
        uncertainties: Per-sample uncertainty scores.
        errors: Per-sample error values.
        n_bins: Number of ECE bins.

    Returns:
        :class:`MetricSummary` with all computed metrics.
    """
    e = np.array(errors, dtype=float)
    is_binary = np.all(np.isin(e, [0.0, 1.0]))

    ece_score, _ = ece(uncertainties, errors, n_bins)
    auroc_score = auroc(uncertainties, errors)
    stats = uncertainty_stats(uncertainties)

    if is_binary:
        pearson_val = float("nan")
        spearman_val = float("nan")
    else:
        corr = correlations(uncertainties, errors)
        pearson_val = corr["pearson"][0]
        spearman_val = corr["spearman"][0]

    return MetricSummary(
        auroc=auroc_score,
        ece=ece_score,
        pearson=pearson_val,
        spearman=spearman_val,
        error_rate=float(e.mean()),
        total_samples=len(errors),
        uncertainty_stats=stats,
    )

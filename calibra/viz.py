"""Visualisation utilities for uncertainty quantification results.

All functions accept an optional ``ax`` parameter. When provided the function
draws into that axes object and returns it, allowing composition into
multi-panel figures. When ``ax`` is ``None`` a new figure is created, saved
(if ``save_path`` is given), and closed.

Call :func:`set_style` once at the start of your script or notebook to apply
the default calibra visual theme.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve as _roc_curve, roc_auc_score

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_DEFAULT_FIG_SIZE = (10, 7)


def set_style() -> None:
    """Apply the default calibra matplotlib/seaborn style.

    Call this once at the start of a script or notebook session.
    """
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = _DEFAULT_FIG_SIZE


def _save_and_close(fig: plt.Figure, save_path: str | None) -> None:
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sanitise(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy().astype(float)
    mask = np.isfinite(arr)
    if not np.all(mask):
        if np.any(mask):
            arr[~mask] = np.max(arr[mask]) + 1.0
        else:
            arr[:] = 1.0
    return arr


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------


def roc_curve(
    uncertainties: list[float],
    errors: list[int],
    method_name: str = "",
    save_path: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot a ROC curve with AUC annotation.

    Args:
        uncertainties: Per-sample uncertainty scores (higher = more uncertain).
        errors: Binary error labels (0 = correct, 1 = incorrect).
        method_name: Label shown in the plot title.
        save_path: If given, save the figure to this path (PNG, 300 dpi).
        ax: Axes to draw into. When ``None`` a new figure is created.

    Returns:
        The :class:`~matplotlib.axes.Axes` that was drawn into.
    """
    standalone = ax is None
    fig, ax = (plt.subplots(figsize=_DEFAULT_FIG_SIZE) if standalone else (ax.figure, ax))

    u = _sanitise(np.array(uncertainties, dtype=float))
    e = np.array(errors)

    try:
        fpr, tpr, _ = _roc_curve(e, u)
        auc = roc_auc_score(e, u)
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    except ValueError as exc:
        ax.text(0.5, 0.5, f"Cannot compute ROC curve:\n{exc}",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve{': ' + method_name if method_name else ''}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    if standalone:
        _save_and_close(fig, save_path)
    return ax


def correlation(
    uncertainties: list[float],
    scores: list[float],
    method_name: str = "",
    save_path: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Scatter-plot uncertainty vs. a continuous quality score.

    Args:
        uncertainties: Per-sample uncertainty scores.
        scores: Per-sample quality scores (higher = better).
        method_name: Label shown in the plot title.
        save_path: If given, save the figure to this path.
        ax: Axes to draw into. When ``None`` a new figure is created.

    Returns:
        The :class:`~matplotlib.axes.Axes` that was drawn into.
    """
    standalone = ax is None
    fig, ax = (plt.subplots(figsize=_DEFAULT_FIG_SIZE) if standalone else (ax.figure, ax))

    u = np.array(uncertainties, dtype=float)
    s = np.array(scores, dtype=float)

    try:
        rho, _ = spearmanr(u, s)
    except Exception:
        rho = float("nan")

    ax.scatter(u, s, alpha=0.5, s=25, color="steelblue")

    try:
        if len(u) >= 2 and np.var(u) > 0:
            z = np.polyfit(u, s, 1)
            ax.plot(u, np.poly1d(z)(u), "r--", alpha=0.7, linewidth=1.5, label="Trend")
    except Exception:
        pass

    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes,
            fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Score (higher = better)")
    ax.set_title(f"Uncertainty vs Score{': ' + method_name if method_name else ''}")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.grid(True, alpha=0.3)

    if standalone:
        _save_and_close(fig, save_path)
    return ax


def calibration_curve(
    uncertainties: list[float],
    errors: list[float],
    method_name: str = "",
    n_bins: int = 10,
    ece: float | None = None,
    save_path: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot a reliability diagram (calibration curve).

    Args:
        uncertainties: Per-sample uncertainty scores.
        errors: Per-sample error values.
        method_name: Label shown in the plot title.
        n_bins: Number of confidence bins.
        ece: Pre-computed ECE value to annotate on the title.
        save_path: If given, save the figure to this path.
        ax: Axes to draw into. When ``None`` a new figure is created.

    Returns:
        The :class:`~matplotlib.axes.Axes` that was drawn into.
    """
    standalone = ax is None
    fig, ax = (plt.subplots(figsize=_DEFAULT_FIG_SIZE) if standalone else (ax.figure, ax))

    u = _sanitise(np.array(uncertainties, dtype=float))
    e = np.array(errors, dtype=float)

    u_min, u_max = u.min(), u.max()
    confidences = 1.0 - (u - u_min) / (u_max - u_min + 1e-10)

    boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts, bin_centers = [], [], [], []

    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(float(1.0 - e[mask].mean()))
        bin_confs.append(float(confidences[mask].mean()))
        bin_counts.append(int(mask.sum()))
        bin_centers.append((lo + hi) / 2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration")
    if bin_confs:
        ax.plot(bin_confs, bin_accs, "o-", linewidth=2, markersize=7, label="Model calibration")

    ax2 = ax.twinx()
    if bin_confs:
        gap = min(bin_confs[i + 1] - bin_confs[i]
                  for i in range(len(bin_confs) - 1)) if len(bin_confs) > 1 else 0.08
        bar_w = min(0.08, gap * 0.8) if gap > 0 else 0.08
        ax2.bar(bin_confs, bin_counts, width=bar_w, alpha=0.25, color="gray", label="Count")
    ax2.set_ylabel("Sample count")
    ax2.legend(loc="upper right")

    title = f"Calibration Curve{': ' + method_name if method_name else ''}"
    if ece is not None:
        title += f"  (ECE = {ece:.3f})"
    ax.set_title(title)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if standalone:
        _save_and_close(fig, save_path)
    return ax


def method_comparison(
    results: dict[str, dict[str, Any]],
    metric: str = "auroc",
    save_path: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Horizontal bar chart comparing uncertainty methods on a single metric.

    Args:
        results: Mapping of method name → metric dict (as stored in
            :attr:`BenchmarkResult.metrics`).
        metric: Metric key to compare (e.g. ``"auroc"``, ``"ece"``).
        save_path: If given, save the figure to this path.
        ax: Axes to draw into. When ``None`` a new figure is created.

    Returns:
        The :class:`~matplotlib.axes.Axes` that was drawn into.
    """
    standalone = ax is None
    fig, ax = (plt.subplots(figsize=(12, 6)) if standalone else (ax.figure, ax))

    methods = list(results.keys())
    values = [results[m].get(metric, 0) or 0 for m in methods]
    lower_is_better = metric in {"ece", "brier_score"}

    bars = ax.barh(methods, values, alpha=0.7)
    for bar, val in zip(bars, values):
        good = val < 0.1 if lower_is_better else val > 0.7
        ok = val < 0.3 if lower_is_better else val > 0.5
        bar.set_color("green" if good else ("orange" if ok else "red"))
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f"  {val:.3f}", va="center", fontsize=10)

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Method")
    ax.set_title(f"Method Comparison — {metric.replace('_', ' ').title()}")
    ax.grid(True, alpha=0.3, axis="x")

    if standalone:
        _save_and_close(fig, save_path)
    return ax


def task_comparison(
    task_results: dict[str, dict[str, dict[str, Any]]],
    metric: str = "auroc",
    save_path: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Grouped bar chart comparing methods across tasks.

    Args:
        task_results: Mapping of task → method → metric dict.
        metric: Metric key to compare.
        save_path: If given, save the figure to this path.
        ax: Axes to draw into. When ``None`` a new figure is created.

    Returns:
        The :class:`~matplotlib.axes.Axes` that was drawn into.
    """
    standalone = ax is None
    fig, ax = (plt.subplots(figsize=(14, 8)) if standalone else (ax.figure, ax))

    tasks = list(task_results.keys())
    all_methods = sorted({m for tr in task_results.values() for m in tr})
    x = np.arange(len(tasks))
    width = 0.8 / len(all_methods) if all_methods else 0.8
    lower_is_better = metric in {"ece", "brier_score"}

    for i, method in enumerate(all_methods):
        values = [task_results[t].get(method, {}).get(metric, 0) or 0 for t in tasks]
        offset = (i - len(all_methods) / 2) * width + width / 2
        bars = ax.bar(x + offset, values, width, label=method, alpha=0.7)
        for bar, val in zip(bars, values):
            good = val < 0.1 if lower_is_better else val > 0.7
            ok = val < 0.3 if lower_is_better else val > 0.5
            bar.set_color("green" if good else ("orange" if ok else "red"))

    ax.set_xlabel("Task")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Task Comparison — {metric.replace('_', ' ').title()}")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if standalone:
        _save_and_close(fig, save_path)
    return ax


def distribution(
    uncertainties: list[float],
    method_name: str = "",
    save_path: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Histogram of uncertainty values with mean/median annotations.

    Args:
        uncertainties: Per-sample uncertainty scores.
        method_name: Label shown in the plot title.
        save_path: If given, save the figure to this path.
        ax: Axes to draw into. When ``None`` a new figure is created.

    Returns:
        The :class:`~matplotlib.axes.Axes` that was drawn into.
    """
    standalone = ax is None
    fig, ax = (plt.subplots(figsize=_DEFAULT_FIG_SIZE) if standalone else (ax.figure, ax))

    u = np.array(uncertainties, dtype=float)
    ax.hist(u, bins=30, alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(u), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(u):.3f}")
    ax.axvline(np.median(u), color="green", linestyle="--", linewidth=2,
               label=f"Median: {np.median(u):.3f}")
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Uncertainty Distribution{': ' + method_name if method_name else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if standalone:
        _save_and_close(fig, save_path)
    return ax


def token_heatmap(
    token_uncertainties: list[list[float]],
    tokens: list[list[str]],
    method_name: str = "",
    sample_indices: list[int] | None = None,
    max_samples: int = 10,
    save_path: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Heatmap of token-level uncertainty across samples.

    Args:
        token_uncertainties: Per-sample list of per-token uncertainty values.
        tokens: Per-sample list of token strings (same shape as
            ``token_uncertainties``).
        method_name: Label shown in the plot title.
        sample_indices: Which samples to include. Defaults to the first
            ``max_samples``.
        max_samples: Upper limit when ``sample_indices`` is ``None``.
        save_path: If given, save the figure to this path.
        ax: Axes to draw into. When ``None`` a new figure is created.

    Returns:
        The :class:`~matplotlib.axes.Axes` that was drawn into.
    """
    if sample_indices is None:
        sample_indices = list(range(min(max_samples, len(token_uncertainties))))

    max_len = max(len(token_uncertainties[i]) for i in sample_indices)
    data = [
        token_uncertainties[i] + [0.0] * (max_len - len(token_uncertainties[i]))
        for i in sample_indices
    ]

    standalone = ax is None
    fig, ax = (
        plt.subplots(figsize=(max(12, max_len * 0.5), len(sample_indices) * 0.8))
        if standalone
        else (ax.figure, ax)
    )

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Sample")
    ax.set_title(f"Token-Level Uncertainty{': ' + method_name if method_name else ''}")
    ax.set_yticks(range(len(sample_indices)))
    ax.set_yticklabels([f"Sample {i}" for i in sample_indices])
    plt.colorbar(im, ax=ax, label="Uncertainty")

    if standalone:
        _save_and_close(fig, save_path)
    return ax

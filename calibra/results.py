"""BenchmarkResult dataclass returned by Benchmark.run()."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Holds all outputs from a single Benchmark.run() call.

    Attributes:
        task: Task type (``"qa"``, ``"math"``, ``"summarization"``, ``"open-ended"``).
        model_name: HuggingFace identifier of the evaluated model.
        dataset: Dataset name or ``"custom"`` for user-provided data.
        methods: Uncertainty methods that were evaluated.
        uncertainties: Per-method list of per-sample uncertainty scores.
        errors: Per-sample error scores (0.0 = correct / high quality,
            1.0 = incorrect / low quality).
        metrics: Per-method metric dict (``auroc``, ``ece``, correlations, …).
        token_level: Optional per-method token-level data
            (``{"uncertainties": [...], "tokens": [...]}``).
        performance: Optional latency and memory stats.
    """

    task: str
    model_name: str
    dataset: str
    methods: list[str]
    uncertainties: dict[str, list[float]]
    errors: list[float]
    metrics: dict[str, dict]
    token_level: dict[str, dict] | None = field(default=None)
    performance: dict | None = field(default=None)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def auroc(self) -> dict[str, float]:
        """AUROC score per method."""
        return {m: self.metrics[m].get("auroc", float("nan")) for m in self.methods}

    @property
    def ece(self) -> dict[str, float]:
        """ECE score per method."""
        return {m: self.metrics[m].get("ece", float("nan")) for m in self.methods}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict representation."""
        def _clean(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        return _clean({
            "task": self.task,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "methods": self.methods,
            "uncertainties": self.uncertainties,
            "errors": self.errors,
            "metrics": self.metrics,
            "token_level": self.token_level,
            "performance": self.performance,
        })

    def to_json(self, path: str) -> None:
        """Write results to a JSON file.

        Args:
            path: Destination file path.
        """
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

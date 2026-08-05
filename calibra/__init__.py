"""
calibra — uncertainty quantification for language model outputs.

Quick start::

    from calibra import Estimator, Benchmark, metrics, viz

    est = Estimator.from_pretrained("Qwen/Qwen3-8B")
    result = est.estimate("What is the capital of France?", methods=["entropy"])

    bench = Benchmark(est)
    results = bench.run(task="qa", dataset="squad", max_samples=100)
    # or with your own data:
    # results = bench.run(task="qa", dataset=[{"input": "...", "target": "..."}])

    metrics.auroc(results.uncertainties["entropy"], results.errors)
    viz.roc_curve(results, method="entropy", save_path="roc.png")
"""

from calibra.estimator import Estimator
from calibra.benchmark import Benchmark
from calibra.results import BenchmarkResult
from calibra import metrics
from calibra import viz
from calibra.datasets import load_builtin, validate_custom

__version__ = "0.1.0"
__all__ = [
    "Estimator",
    "Benchmark",
    "BenchmarkResult",
    "metrics",
    "viz",
    "load_builtin",
    "validate_custom",
]

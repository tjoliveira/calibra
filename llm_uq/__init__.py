"""
llm-uq — uncertainty quantification for language model outputs.

PyPI distribution: ``pip install llm-uq``
Import name:       ``import llm_uq``

Six uncertainty estimation methods (entropy, max_probability,
sequence_probability, self_consistency, self_reflection, bsdetector)
evaluated across QA, math, summarisation, and open-ended generation tasks.

Quick start::

    from llm_uq import Estimator, Benchmark, metrics, viz

    est = Estimator.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    # Single-prompt uncertainty
    scores = est.estimate("What is the capital of France?", methods=["entropy"])
    # {"entropy": 0.43}

    # Benchmark on your own data
    bench = Benchmark(est)
    results = bench.run(
        task="qa",
        dataset=[{"input": "Who wrote Hamlet?", "target": "Shakespeare"}],
    )

    # Metrics and visualisation
    print(results.auroc)
    viz.roc_curve(results.uncertainties["entropy"], results.errors)

Full documentation: https://github.com/tjoliveira/llm-uq/blob/main/docs/usage.md
Interactive notebook: https://github.com/tjoliveira/llm-uq/blob/main/quickstart.ipynb
"""

from llm_uq.estimator import Estimator
from llm_uq.benchmark import Benchmark
from llm_uq.results import BenchmarkResult
from llm_uq import metrics
from llm_uq import viz
from llm_uq.datasets import load_builtin, validate_custom

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

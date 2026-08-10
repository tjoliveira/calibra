<p align="center">
  <img src="assets/logo.svg" alt="calibra" width="480"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/llm-uq/"><img src="https://img.shields.io/pypi/v/llm-uq?label=PyPI&color=6366f1" alt="PyPI"/></a>
  <a href="https://pypi.org/project/llm-uq/"><img src="https://img.shields.io/pypi/pyversions/llm-uq?color=818cf8" alt="Python versions"/></a>
  <a href="https://github.com/tjoliveira/calibra"><img src="https://img.shields.io/badge/github-tjoliveira%2Fcalibra-blue?logo=github" alt="GitHub"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"/></a>
</p>

<p align="center">
  <b>Uncertainty quantification for language model outputs.</b><br/>
  Six methods. Four tasks. Rigorous metrics. One import.
</p>

---

**calibra** tells you when your language model doesn't know what it's talking about.

It implements six uncertainty estimation methods — spanning token probability, semantic consistency, and self-reflection — and evaluates them across QA, math, summarisation, and open-ended generation tasks. You get AUROC, ECE, and Spearman correlation so you can measure, not just guess, how well uncertainty tracks actual errors.

## Installation

```bash
pip install llm-uq
```

```bash
# With 4-bit / 8-bit quantization support (requires CUDA):
pip install "llm-uq[quantization]"
```

> The PyPI distribution is `llm-uq`; the import name is `calibra`.

## Quickstart

```python
from calibra import Estimator, Benchmark, metrics, viz

# Load any HuggingFace causal LM
est = Estimator.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

# Single-prompt uncertainty — one forward pass
scores = est.estimate(
    "What is the boiling point of water?",
    methods=["entropy", "max_probability", "sequence_probability"],
)
# {"entropy": 0.43, "max_probability": 0.21, "sequence_probability": 1.87}

# Full benchmark on your own data
bench = Benchmark(est)
results = bench.run(
    task="qa",
    dataset=[
        {"input": "Who wrote Hamlet?",        "target": "Shakespeare"},
        {"input": "What year did WWII end?",   "target": "1945"},
    ],
    methods=["entropy", "self_consistency"],
)

# Metrics
print(results.auroc)   # {"entropy": 0.71, "self_consistency": 0.78}
print(results.ece)     # {"entropy": 0.08, "self_consistency": 0.05}

# Visualisation
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
viz.roc_curve(results.uncertainties["entropy"], results.errors, ax=axes[0])
viz.calibration_curve(results.uncertainties["entropy"], results.errors, ax=axes[1])
plt.show()
```

## Uncertainty Methods

| Method | Cost | Family | Notes |
|---|---|---|---|
| `entropy` | 1× | Probability | Mean token entropy over greedy decode |
| `max_probability` | 1× | Probability | Mean `1 − max(softmax)` per token |
| `sequence_probability` | 1× | Probability | Negative mean log-probability |
| `self_consistency` | N× | Sampling | Semantic spread across N stochastic samples |
| `self_reflection` | 1× | Sampling | Model rates its own answer (A/B/C prompt) |
| `bsdetector` | N+1× | Sampling | Weighted combination of the two above |

`self_consistency` and `bsdetector` require a sentence-transformers model (default: `all-MiniLM-L6-v2`) to measure semantic similarity across samples.

## Supported Tasks

| Task | Built-in dataset | Custom data |
|---|---|---|
| `qa` | SQuAD | ✓ |
| `math` | GSM8K | ✓ |
| `summarization` | CNN/DailyMail | ✓ |
| `open-ended` | WritingPrompts | ✓ |

Pass a dataset name string for built-in data, or a list of `{"input": ..., "target": ...}` dicts for your own:

```python
# Built-in
results = bench.run(task="qa", dataset="squad", max_samples=100)

# Custom
results = bench.run(task="math", dataset=[
    {"input": "If x + 3 = 7, what is x?", "target": "4"},
])
```

## Metrics

```python
from calibra import metrics

u = results.uncertainties["entropy"]
e = results.errors

metrics.auroc(u, e)          # float — area under the ROC curve
metrics.ece(u, e)            # (float, dict) — expected calibration error + bin stats
metrics.correlations(u, e)   # {"pearson": (r, p), "spearman": (r, p)}
metrics.uncertainty_stats(u) # {"mean", "std", "min", "max", "median", "q25", "q75"}
metrics.summarize(u, e)      # MetricSummary dataclass with all of the above
```

## Visualisation

All plot functions accept an optional `ax=` argument for embedding into existing figures.

```python
from calibra import viz

viz.roc_curve(uncertainties, errors, method_name="entropy", save_path="roc.png")
viz.calibration_curve(uncertainties, errors, method_name="entropy")
viz.distribution(uncertainties, method_name="entropy")
viz.correlation(uncertainties, quality_scores, method_name="entropy")
viz.method_comparison(results.metrics, metric="auroc")
viz.task_comparison(task_results, metric="auroc")
viz.token_heatmap(token_uncertainties, tokens)
```

## Quantization

Run large models on consumer hardware:

```python
# 4-bit (requires bitsandbytes + CUDA)
est = Estimator.from_pretrained("Qwen/Qwen3-8B", load_in_4bit=True)

# 8-bit
est = Estimator.from_pretrained("Qwen/Qwen3-8B", load_in_8bit=True)
```

## Token-level uncertainty

```python
detail = est.score_all("What is the capital of France?", max_new_tokens=30)

detail["entropy"]["sentence"]  # float — mean over tokens
detail["entropy"]["tokens"]    # list[float] — per-token entropy
```

## Documentation

Full usage guide: [`docs/usage.md`](docs/usage.md)

Topics covered:
- All six uncertainty methods and their trade-offs
- Built-in vs. custom datasets
- Metrics in depth (AUROC, ECE, Spearman ρ)
- Visualisation composability with `ax=`
- Quantization

## Research

The `experiments/` directory contains the configuration and runner used in the accompanying study ([`REPORT.md`](REPORT.md)). Pre-computed results for Qwen3-8B, Qwen3-4B, and Ministral-8B are in `results/`.

## Development

```bash
git clone https://github.com/tjoliveira/calibra
cd calibra
pip install -e ".[dev]"
pytest tests/
```

## License

MIT

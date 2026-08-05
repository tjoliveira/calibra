# calibra — Usage Guide

**calibra** is a Python library for estimating and evaluating uncertainty in language model outputs. It supports six uncertainty methods, four built-in HuggingFace datasets, and arbitrary user-provided datasets.

---

## Installation

```bash
pip install calibra
```

For 4-bit / 8-bit quantization (requires CUDA):

```bash
pip install "calibra[quantization]"
```

For development (tests, docs):

```bash
pip install "calibra[dev,docs]"
```

---

## Quickstart

```python
from calibra import Estimator, Benchmark, metrics, viz

# 1. Load a HuggingFace model
est = Estimator.from_pretrained("Qwen/Qwen3-8B")

# 2. Estimate uncertainty for a single prompt
scores = est.estimate(
    "What is the boiling point of water?",
    methods=["entropy", "self_reflection"],
)
# {"entropy": 0.432, "self_reflection": 0.0}

# 3. Benchmark across a dataset
bench = Benchmark(est)
results = bench.run(task="qa", dataset="squad", max_samples=100)

# 4. Inspect metrics
print(results.auroc)   # {"entropy": 0.71, "self_reflection": 0.65, ...}
print(results.ece)     # {"entropy": 0.08, ...}

# 5. Visualise
viz.roc_curve(
    results.uncertainties["entropy"],
    results.errors,
    method_name="entropy",
    save_path="roc_entropy.png",
)
```

---

## Using Built-in Datasets

```python
from calibra import Benchmark, Estimator

est = Estimator.from_pretrained("Qwen/Qwen3-4B")
bench = Benchmark(est)

# Task strings and their default HuggingFace datasets:
#   "qa"            → SQuAD  (rajpurkar/squad)
#   "math"          → GSM8K  (openai/gsm8k)
#   "summarization" → CNN/DailyMail  (abisee/cnn_dailymail)
#   "open-ended"    → WritingPrompts  (euclaise/writingprompts)

results = bench.run(task="math", dataset="gsm8k", max_samples=50)
```

You can also specify a custom HuggingFace dataset identifier:

```python
results = bench.run(task="qa", dataset="my-org/my-qa-dataset", max_samples=200)
```

---

## Using Custom Datasets

Any Python list of dicts is accepted, as long as each item has an `"input"` key (the prompt) and a `"target"` key (the ground-truth answer or reference). An optional `"id"` field is supported; missing IDs are assigned automatically.

```python
from calibra import Benchmark, Estimator, validate_custom

est = Estimator.from_pretrained("Qwen/Qwen3-8B")
bench = Benchmark(est)

my_data = [
    {"input": "What year did WWII end?",      "target": "1945"},
    {"input": "Who wrote Hamlet?",            "target": "Shakespeare"},
    {"input": "What is the speed of light?",  "target": "299792458 m/s"},
]

# Optional: validate before passing (raises ValueError on bad structure)
my_data = validate_custom(my_data, task="qa")

results = bench.run(task="qa", dataset=my_data)
print(results.auroc)
```

### Custom dataset keys by task

| Task | Required keys | Notes |
|---|---|---|
| `"qa"` | `"input"`, `"target"` | Question / short answer |
| `"math"` | `"input"`, `"target"` | Problem / numerical answer |
| `"summarization"` | `"input"`, `"target"` | Article / reference summary |
| `"open-ended"` | `"input"`, `"target"` | Prompt / reference response |

---

## Uncertainty Methods

| Method | Family | Extra cost | Needs semantic model? | Description |
|---|---|---|---|---|
| `entropy` | Probability | 0 × | No | Mean token entropy from one greedy-decode pass |
| `max_probability` | Probability | 0 × | No | Mean of `1 − max(softmax)` per token |
| `sequence_probability` | Probability | 0 × | No | Negative mean log-probability of generated tokens |
| `self_consistency` | Sampling | N × | **Yes** | Mean semantic distance of N samples to greedy output |
| `self_reflection` | Sampling | 1 × | No | Model rates its own answer: A → 0.0, B → 1.0, C → 0.5 |
| `bsdetector` | Sampling | N+1 × | **Yes** | Weighted combination of self_consistency + self_reflection |

Probability-based methods share a single forward pass. Sampling-based methods require additional generation calls (controlled by `num_samples`).

The semantic model defaults to `"all-MiniLM-L6-v2"` (sentence-transformers). Pass `semantic_model=None` to disable `self_consistency` and `bsdetector`.

```python
# Disable semantic model (faster, no self_consistency / bsdetector)
est = Estimator.from_pretrained("Qwen/Qwen3-8B", semantic_model=None)

# Custom semantic model
est = Estimator.from_pretrained(
    "Qwen/Qwen3-8B",
    semantic_model="paraphrase-multilingual-mpnet-base-v2",
)
```

---

## Metrics

| Metric | Function | When to use |
|---|---|---|
| AUROC | `metrics.auroc(u, e)` | Binary tasks (QA, Math) — does uncertainty predict errors? |
| ECE | `metrics.ece(u, e)` | All tasks — is uncertainty well-calibrated? |
| Spearman ρ | `metrics.correlations(u, e)["spearman"]` | Continuous tasks (Summarization, Open-ended) |
| Summary | `metrics.summarize(u, e)` | All metrics at once as a `MetricSummary` dataclass |

```python
from calibra import metrics

u = results.uncertainties["entropy"]
e = results.errors

auroc_score = metrics.auroc(u, e)
ece_score, _ = metrics.ece(u, e)
corr = metrics.correlations(u, e)["spearman"][0]

summary = metrics.summarize(u, e)
print(summary.auroc, summary.ece, summary.spearman)
```

---

## Visualisation

All plot functions accept an optional `ax` parameter, making them composable into multi-panel figures:

```python
import matplotlib.pyplot as plt
from calibra import viz

# Single plot saved to disk
viz.roc_curve(u, errors, method_name="entropy", save_path="roc.png")
viz.distribution(u, method_name="entropy", save_path="dist.png")

# Multi-panel composition
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
viz.roc_curve(u, errors, method_name="entropy", ax=axes[0])
viz.calibration_curve(u, errors, method_name="entropy", ece=0.07, ax=axes[1])
fig.savefig("combined.png", dpi=300)

# Compare all methods
viz.method_comparison(results.metrics, metric="auroc", save_path="comparison.png")
```

Apply the default calibra visual theme once per session:

```python
viz.set_style()
```

---

## Quantization

```python
# 4-bit (smallest memory footprint, requires CUDA + bitsandbytes)
est = Estimator.from_pretrained("Qwen/Qwen3-8B", load_in_4bit=True)

# 8-bit
est = Estimator.from_pretrained("Qwen/Qwen3-8B", load_in_8bit=True)
```

Install bitsandbytes:

```bash
pip install "calibra[quantization]"
```

---

## Reproducing the Research

The `experiments/` directory contains the original experiment runner and configuration files used in the accompanying report (`REPORT.md`). Pre-computed results (JSON files + figures) are stored under `results/`.

```bash
# Re-run experiments (requires GPU, ~1-2 hours per model)
python experiments/run_experiments.py --config experiments/configs/qwen3_8b.yaml
```

Results tables can be re-generated from existing JSON files:

```bash
python scripts/extract_results_table.py
python scripts/extract_latency_quality_tradeoff.py
```

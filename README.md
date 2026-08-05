# calibra

A Python library for uncertainty quantification in language model outputs.

calibra implements six uncertainty estimation methods and evaluates them across multiple task types, giving you rigorous metrics (AUROC, ECE, Spearman correlation) to understand when your model knows it doesn't know.

## Installation

```bash
pip install calibra
# or, for quantization support:
pip install "calibra[quantization]"
```

## Quickstart

```python
from calibra import Estimator, Benchmark, metrics, viz

# Load any HuggingFace causal language model
est = Estimator.from_pretrained("Qwen/Qwen3-8B")

# Single-prompt uncertainty estimation
scores = est.estimate(
    "What is the boiling point of water?",
    methods=["entropy", "self_reflection"],
)
# {"entropy": 0.43, "self_reflection": 0.0}

# Full benchmark on a built-in dataset
bench = Benchmark(est)
results = bench.run(task="qa", dataset="squad", max_samples=100)

# Benchmark on your own data
my_data = [
    {"input": "Who wrote Hamlet?", "target": "Shakespeare"},
    {"input": "What year did WWII end?", "target": "1945"},
]
results = bench.run(task="qa", dataset=my_data)

# Metrics
print(results.auroc)   # {"entropy": 0.71, ...}
print(results.ece)     # {"entropy": 0.08, ...}

# Visualisation
viz.roc_curve(
    results.uncertainties["entropy"],
    results.errors,
    method_name="entropy",
    save_path="roc.png",
)
```

## Uncertainty Methods

| Method | Cost | Notes |
|---|---|---|
| `entropy` | 1× | Mean token entropy (greedy decode) |
| `max_probability` | 1× | Mean `1 − max(softmax)` per token |
| `sequence_probability` | 1× | Negative mean log-probability |
| `self_consistency` | N× | Semantic spread across N samples |
| `self_reflection` | 1× | Model rates its own answer |
| `bsdetector` | N+1× | self_consistency + self_reflection |

`self_consistency` and `bsdetector` require a sentence-transformers model (default: `all-MiniLM-L6-v2`).

## Documentation

Full usage guide: [`docs/usage.md`](docs/usage.md)

Topics covered:
- Built-in datasets (SQuAD, GSM8K, CNN/DailyMail, WritingPrompts)
- Custom dataset protocol (`"input"` / `"target"` keys)
- All six uncertainty methods and their trade-offs
- Metrics (AUROC, ECE, Spearman ρ)
- Visualisation with `ax=` composability
- Quantization (4-bit / 8-bit)

## Research

The `experiments/` directory contains the configuration and runner used in the accompanying study (`REPORT.md`). Pre-computed results for three models (Qwen3-8B, Qwen3-4B, Ministral-8B) are in `results/`.

## Development

```bash
git clone https://github.com/tiago/calibra
cd calibra
pip install -e ".[dev]"
pytest tests/
```

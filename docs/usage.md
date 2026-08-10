# calibra — Usage Guide

**calibra** is a Python library for estimating and evaluating uncertainty in language model outputs. It implements six uncertainty estimation methods across four task types, with rigorous evaluation metrics and built-in visualisation.

---

## Installation

```bash
pip install llm-uq
```

> The PyPI distribution name is `llm-uq`. The import name is `calibra`.

For 4-bit / 8-bit quantization (requires CUDA and a compatible GPU):

```bash
pip install "llm-uq[quantization]"
```

For development (tests, linting, docs):

```bash
git clone https://github.com/tjoliveira/llm-uq
cd calibra
pip install -e ".[dev,docs]"
```

---

## Loading a model

`Estimator.from_pretrained` wraps HuggingFace `AutoModelForCausalLM` and returns a ready-to-use estimator.

```python
from llm_uq import Estimator

est = Estimator.from_pretrained(
    "Qwen/Qwen3-8B",
    trust_remote_code=True,           # required for some Qwen checkpoints
    semantic_model="all-MiniLM-L6-v2",# sentence-transformers model for SC / BSDetector
    device=None,                      # auto: "cuda" if available, else "cpu"
)
```

If you already have a loaded model and tokenizer (e.g. from your own training pipeline), pass them directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_uq import Estimator

model = AutoModelForCausalLM.from_pretrained("my-org/my-model")
tokenizer = AutoTokenizer.from_pretrained("my-org/my-model")
est = Estimator(model, tokenizer, semantic_model="all-MiniLM-L6-v2")
```

---

## Single-prompt uncertainty estimation

### `estimate()` — sentence-level scores

```python
scores = est.estimate(
    "What is the boiling point of water?",
    methods=["entropy", "max_probability", "sequence_probability",
             "self_consistency", "self_reflection", "bsdetector"],
    max_new_tokens=100,   # max tokens to generate
    temperature=1.0,      # sampling temperature (sampling-based methods)
    num_samples=5,        # stochastic samples for self_consistency / bsdetector
)
# Returns: {"entropy": 0.43, "max_probability": 0.21, ...}
```

Omit `methods` to run all six. Unknown method names raise `ValueError`.

### `score_all()` — sentence + token-level scores

```python
detail = est.score_all(
    "Who invented the telephone?",
    max_new_tokens=50,
    num_samples=5,
)

# Probability-based methods return {"sentence": float, "tokens": list[float]}
sentence_entropy = detail["entropy"]["sentence"]
token_entropies  = detail["entropy"]["tokens"]     # one value per generated token

# Sampling-based methods return a plain float
sc_score  = detail["self_consistency"]
bsd_score = detail["bsdetector"]
```

### Comparing prompts

```python
factual    = est.estimate("What is 2 + 2?",                   methods=["entropy"])
speculative = est.estimate("What will AI look like in 2100?", methods=["entropy"])

print(f"Factual entropy:     {factual['entropy']:.4f}")
print(f"Speculative entropy: {speculative['entropy']:.4f}")
# Speculative prompt typically yields higher entropy.
```

---

## Benchmarking across a dataset

`Benchmark.run()` evaluates all requested methods on every sample and computes AUROC, ECE, and correlations automatically.

### Built-in datasets

```python
from llm_uq import Estimator, Benchmark

est = Estimator.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
bench = Benchmark(est)

# Available built-in datasets:
#   "qa"            → SQuAD       (rajpurkar/squad)
#   "math"          → GSM8K       (openai/gsm8k)
#   "summarization" → CNN/DailyMail (abisee/cnn_dailymail)
#   "open-ended"    → WritingPrompts (euclaise/writingprompts)

results = bench.run(
    task="qa",
    dataset="squad",
    max_samples=100,       # optional: limit to N samples
    methods=["entropy", "max_probability", "self_consistency"],
    max_new_tokens=100,
    num_samples=5,
    seed=42,               # shuffling seed for built-in datasets
)
```

### Custom datasets

Pass a list of dicts with `"input"` and `"target"` keys:

```python
from llm_uq import validate_custom

my_data = [
    {"input": "What year did WWII end?",      "target": "1945"},
    {"input": "Who wrote Hamlet?",            "target": "Shakespeare"},
    {"input": "What is the speed of light?",  "target": "299792458 m/s"},
]

# Optional validation step — raises ValueError if structure is wrong
my_data = validate_custom(my_data, task="qa")

results = bench.run(task="qa", dataset=my_data)
```

The `"id"` field is optional; missing IDs are assigned sequential integers.

### Custom dataset keys by task

| Task | `"input"` | `"target"` |
|---|---|---|
| `"qa"` | Question text | Expected answer (short string) |
| `"math"` | Problem statement | Numerical answer |
| `"summarization"` | Full article text | Reference summary |
| `"open-ended"` | Writing prompt | Reference response |

### Inspecting results

```python
# Aggregated metrics
print(results.auroc)    # {"entropy": 0.71, "self_consistency": 0.78, ...}
print(results.ece)      # {"entropy": 0.08, ...}

# Per-sample data
results.uncertainties   # {"entropy": [0.43, 0.12, ...], ...}
results.errors          # [0.0, 1.0, 0.0, ...]  (1.0 = error)

# Full metric dict per method
results.metrics["entropy"]
# {
#   "auroc": 0.71, "ece": 0.08,
#   "correlation_pearson": 0.42, "correlation_spearman": 0.39,
#   "error_rate": 0.28, "total_samples": 100,
#   "uncertainty_stats": {"mean": 0.43, "std": 0.12, ...},
#   "avg_latency_s": 0.82
# }

# Token-level data (probability-based methods only)
results.token_level["entropy"]["uncertainties"]  # list of per-sample token lists
results.token_level["entropy"]["tokens"]         # list of per-sample token strings

# Save to JSON
results.to_json("results/squad_qwen3_8b.json")
```

---

## Uncertainty methods in depth

### Entropy

Computes the Shannon entropy over the vocabulary at each generation step:

$$H(t) = -\sum_{v \in \mathcal{V}} p(v \mid \mathbf{x}, \mathbf{y}_{<t}) \log p(v \mid \mathbf{x}, \mathbf{y}_{<t})$$

Sentence-level score:

$$\mathcal{U}_{\text{entropy}} = \frac{1}{T}\sum_{t=1}^{T} H(t)$$

**Interpretation:** High entropy means probability mass is spread across many tokens — the model is genuinely unsure what to say next. Low entropy means one token dominates.

**Cost:** 1 forward pass. No semantic model required.

---

### Max Probability

$$\mathcal{U}_{\text{max\_prob}} = \frac{1}{T}\sum_{t=1}^{T} \Bigl(1 - \max_{v \in \mathcal{V}} p(v \mid \mathbf{x}, \mathbf{y}_{<t})\Bigr)$$

**Interpretation:** Focuses on the single most likely token rather than the full distribution. Complementary to entropy — useful when the distribution is bimodal (one dominant token and a single strong alternative).

**Cost:** 1 forward pass. No semantic model required.

---

### Sequence Probability

The negative mean log-likelihood of the generated sequence:

$$\mathcal{U}_{\text{seq\_prob}} = -\frac{1}{T}\sum_{t=1}^{T} \log p(y_t \mid \mathbf{x}, \mathbf{y}_{<t})$$

**Interpretation:** Per-token perplexity in natural-log space. A value of 0 means the model assigned probability 1 to every token it generated. Tends to scale with sequence length.

**Cost:** 1 forward pass. No semantic model required.

---

### Self-Consistency

Generate $N$ stochastic samples and measure their mean semantic distance from the greedy reference output:

$$\mathcal{U}_{\text{SC}} = \frac{1}{N}\sum_{i=1}^{N} \Bigl(1 - \cos\bigl(\mathbf{e}_{\text{ref}},\, \mathbf{e}_i\bigr)\Bigr)$$

where $\mathbf{e}_{\text{ref}} = \text{embed}(y_{\text{greedy}})$ and $\mathbf{e}_i = \text{embed}(s_i)$.

**Interpretation:** If the model always says semantically the same thing regardless of temperature, uncertainty is 0. Large variation in meaning across samples signals uncertainty.

**Cost:** $N$ additional forward passes. Requires a sentence-transformers semantic model.

---

### Self-Reflection

Present the model's own answer back to it via a structured prompt and ask it to choose A, B, or C:

$$\mathcal{U}_{\text{SR}} = \begin{cases} 0.0 & \text{A — "I believe my answer is correct"} \\ 0.5 & \text{C — "I am not sure"} \\ 1.0 & \text{B — "I believe my answer is incorrect"} \end{cases}$$

**Interpretation:** A lightweight introspective signal. Models that frequently choose B or C on correct answers are poorly calibrated. Works best on instruction-tuned models.

**Cost:** 1 additional forward pass (1 token generated). No semantic model required.

---

### BSDetector

A weighted linear combination of self-consistency and self-reflection:

$$\mathcal{U}_{\text{BSD}} = \beta \cdot \mathcal{U}_{\text{SC}} + (1 - \beta) \cdot \mathcal{U}_{\text{SR}}, \quad \beta = 0.7$$

**Interpretation:** Combines the distributional signal from self-consistency with the introspective signal from self-reflection. The default $\beta = 0.7$ gives more weight to self-consistency.

**Cost:** $N+1$ forward passes. Requires a semantic model.

```python
# Adjust beta via the low-level method:
score = est.bsdetector_score(
    prompt="Who invented the telephone?",
    answer="Alexander Graham Bell",
    num_samples=5,
    beta=0.5,   # equal weight to both components
)
```

---

## Metrics in depth

### AUROC

```python
auroc = metrics.auroc(uncertainties, errors)
```

Treats uncertainty as a binary classifier score for predicting errors. A value of 0.5 is random; 1.0 is perfect. Returns `NaN` if all samples are in the same class.

### ECE

```python
ece_score, bin_stats = metrics.ece(uncertainties, errors, n_bins=10)
```

`bin_stats` contains `bin_boundaries`, `bin_accuracies`, `bin_confidences`, and `bin_counts` — everything needed to draw a reliability diagram. Used by `viz.calibration_curve()`.

### Correlations

```python
corr = metrics.correlations(uncertainties, errors, method="all")
# {"pearson": (r, p_value), "spearman": (rho, p_value)}

corr = metrics.correlations(uncertainties, errors, method="spearman")
# {"spearman": (rho, p_value)}
```

Appropriate for continuous error scales (ROUGE-L for summarisation; Prometheus scores for open-ended generation).

### Uncertainty statistics

```python
stats = metrics.uncertainty_stats(uncertainties)
# {"mean": 0.43, "std": 0.12, "min": 0.02, "max": 1.87, "median": 0.38, "q25": 0.21, "q75": 0.61}
```

### `MetricSummary` dataclass

```python
summary = metrics.summarize(uncertainties, errors)

summary.auroc             # float
summary.ece               # float
summary.pearson           # float (NaN for binary tasks)
summary.spearman          # float (NaN for binary tasks)
summary.error_rate        # float — mean error across samples
summary.total_samples     # int
summary.uncertainty_stats # dict — same as metrics.uncertainty_stats()
```

---

## Visualisation in depth

### ROC curve

```python
viz.roc_curve(
    uncertainties, errors,
    method_name="entropy",  # appears in the plot title
    save_path="roc.png",    # optional; PNG at 300 dpi
    ax=None,                # pass a Matplotlib Axes to compose
)
```

### Calibration / reliability diagram

```python
viz.calibration_curve(
    uncertainties, errors,
    method_name="entropy",
    n_bins=10,
    ece=summary.ece,        # optional pre-computed value to annotate
    save_path="cal.png",
    ax=None,
)
```

### Uncertainty distribution

```python
viz.distribution(
    uncertainties,
    method_name="entropy",
    save_path="dist.png",
    ax=None,
)
```

### Uncertainty vs. quality scatter

```python
viz.correlation(
    uncertainties, quality_scores,
    method_name="entropy",
    save_path="corr.png",
    ax=None,
)
```

### Method comparison bar chart

```python
viz.method_comparison(
    results.metrics,     # dict[method_name, metric_dict]
    metric="auroc",      # or "ece", "correlation_spearman", ...
    save_path="compare.png",
    ax=None,
)
```

### Cross-task grouped bar chart

```python
viz.task_comparison(
    {
        "qa":   qa_results.metrics,
        "math": math_results.metrics,
    },
    metric="auroc",
    save_path="tasks.png",
    ax=None,
)
```

### Token-level heatmap

```python
viz.token_heatmap(
    token_uncertainties=results.token_level["entropy"]["uncertainties"],
    tokens=results.token_level["entropy"]["tokens"],
    method_name="entropy",
    max_samples=10,       # number of rows to show
    save_path="heatmap.png",
    ax=None,
)
```

### Composing multi-panel figures

```python
import matplotlib.pyplot as plt
from llm_uq import viz

viz.set_style()   # apply calibra's seaborn whitegrid theme

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
viz.roc_curve(u, e,          ax=axes[0, 0], method_name="entropy")
viz.calibration_curve(u, e,  ax=axes[0, 1], method_name="entropy")
viz.distribution(u,          ax=axes[1, 0], method_name="entropy")
viz.correlation(u, scores,   ax=axes[1, 1], method_name="entropy")
fig.suptitle("Entropy — Full Diagnostic", fontsize=14)
plt.tight_layout()
fig.savefig("full_diagnostic.png", dpi=300)
```

---

## Quantization

For GPU-constrained environments, enable quantization via `bitsandbytes`:

```bash
pip install "llm-uq[quantization]"
```

```python
# 4-bit — smallest VRAM footprint (~4 GB for a 7B model)
est = Estimator.from_pretrained("Qwen/Qwen3-8B", load_in_4bit=True, trust_remote_code=True)

# 8-bit — better quality/size trade-off (~8 GB for a 7B model)
est = Estimator.from_pretrained("Qwen/Qwen3-8B", load_in_8bit=True, trust_remote_code=True)
```

Quantization is silently ignored on CPU — a warning is logged and the model loads at full precision.

---

## Reproducing the Research

The `experiments/` directory contains the configuration files and runner used in the accompanying report ([`REPORT.md`](../REPORT.md)).

```bash
# Run experiments for a specific model (GPU required, ~1–2 hours)
python experiments/run_experiments.py --config experiments/configs/qwen3_8b.yaml

# Re-generate result tables from saved JSON files
python scripts/extract_results_table.py
python scripts/extract_latency_quality_tradeoff.py
```

Pre-computed results for Qwen3-8B, Qwen3-4B, and Ministral-8B are stored under `results/`.

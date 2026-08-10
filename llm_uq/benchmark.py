"""Benchmark: run uncertainty estimation across a dataset and collect metrics."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from tqdm import tqdm

from llm_uq.estimator import Estimator
from llm_uq.results import BenchmarkResult
from llm_uq.datasets import load_builtin, validate_custom
from llm_uq import metrics as _metrics
from llm_uq._scoring import score_qa, score_math, score_summarization, score_open_ended

logger = logging.getLogger(__name__)

_ALL_METHODS = [
    "entropy",
    "max_probability",
    "sequence_probability",
    "self_consistency",
    "self_reflection",
    "bsdetector",
]

# For brief-answer tasks, cap generation length so the model doesn't ramble.
_SHORT_ANSWER_TASKS = {"qa", "math"}
_SHORT_ANSWER_MAX_TOKENS = 6


class Benchmark:
    """Evaluate uncertainty methods across a dataset.

    Example::

        from llm_uq import Estimator, Benchmark

        est = Estimator.from_pretrained("Qwen/Qwen3-8B")
        bench = Benchmark(est)

        # Built-in dataset:
        results = bench.run(task="qa", dataset="squad", max_samples=100)

        # Custom dataset:
        data = [{"input": "What year did WWII end?", "target": "1945"}]
        results = bench.run(task="qa", dataset=data)
    """

    def __init__(
        self,
        estimator: Estimator,
        judge_model_name: str = "prometheus-eval/prometheus-7b-v2.0",
    ) -> None:
        """Initialise with an :class:`~llm_uq.Estimator`.

        The Prometheus 2 judge model is loaded lazily — only when
        ``run(task="open-ended")`` is first called.

        Args:
            estimator: A ready-to-use :class:`~llm_uq.Estimator` instance.
            judge_model_name: HuggingFace identifier for the Prometheus 2 judge.
        """
        self.estimator = estimator
        self._judge_model_name = judge_model_name
        self._judge_model = None
        self._judge_tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        dataset: str | list[dict[str, Any]],
        max_samples: int | None = None,
        methods: list[str] | None = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        num_samples: int = 5,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Run the full benchmark on a task.

        Args:
            task: Task type — one of ``"qa"``, ``"math"``,
                ``"summarization"``, ``"open-ended"``.
            dataset: Either a built-in dataset name (``"squad"``, ``"gsm8k"``,
                ``"cnn_dailymail"``, ``"writingprompts"``) or a list of dicts
                with ``"input"`` and ``"target"`` keys.
            max_samples: Limit evaluation to this many samples.
            methods: Uncertainty methods to run. Defaults to all six.
            max_new_tokens: Max tokens generated per sample.
            temperature: Sampling temperature for sampling-based methods.
            num_samples: Number of samples for ``self_consistency`` /
                ``bsdetector``.
            seed: Random seed for shuffling built-in datasets.

        Returns:
            A :class:`~llm_uq.BenchmarkResult` with uncertainties, errors,
            and computed metrics.
        """
        if methods is None:
            methods = list(_ALL_METHODS)

        if not 1 <= num_samples <= 100:
            raise ValueError(f"num_samples must be between 1 and 100, got {num_samples}.")

        # Resolve dataset
        if isinstance(dataset, str):
            dataset_label = dataset
            data = load_builtin(task, dataset=dataset, max_samples=max_samples, seed=seed)
        else:
            dataset_label = "custom"
            data = validate_custom(dataset, task)
            if max_samples:
                data = data[:max_samples]

        if task == "open-ended" and (
            "self_reflection" in methods or "bsdetector" in methods
        ):
            self._ensure_judge_loaded()

        effective_max_tokens = (
            min(max_new_tokens, _SHORT_ANSWER_MAX_TOKENS)
            if task in _SHORT_ANSWER_TASKS
            else max_new_tokens
        )
        if task in _SHORT_ANSWER_TASKS and max_new_tokens > _SHORT_ANSWER_MAX_TOKENS:
            logger.info(
                "Capping max_new_tokens to %d for %s task.", _SHORT_ANSWER_MAX_TOKENS, task
            )

        logger.info("Starting benchmark: task=%s, n=%d, methods=%s", task, len(data), methods)

        all_uncertainties: dict[str, list[float | None]] = {m: [] for m in methods}
        all_errors: list[float] = []
        token_level: dict[str, dict] = {}
        perf_latencies: list[float] = []
        method_perf: dict[str, list[float]] = {m: [] for m in methods}

        for sample in tqdm(data, desc=f"Benchmark [{task}]"):
            prompt = self._build_prompt(sample, task)

            # Single greedy-decode forward pass shared by probability-based methods
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            _, outputs = self.estimator._generate_with_scores(prompt, effective_max_tokens)
            latency = max(time.perf_counter() - t0, 0.0)
            perf_latencies.append(latency)

            prediction = self.estimator._extract_text(outputs)
            correctness = self._score_correctness(sample, prediction, task)
            all_errors.append(1.0 - correctness)

            for method in methods:
                t_m = time.perf_counter()
                try:
                    val = self._run_method(
                        method, prompt, outputs, prediction, task,
                        effective_max_tokens, temperature, num_samples,
                    )
                    # entropy / max_probability return dicts — extract sentence level
                    if isinstance(val, dict):
                        # Collect token-level data
                        if method not in token_level:
                            token_level[method] = {"uncertainties": [], "tokens": []}
                        token_level[method]["uncertainties"].append(val.get("tokens", []))
                        token_texts = self._extract_token_texts(outputs)
                        token_level[method]["tokens"].append(token_texts)
                        all_uncertainties[method].append(val["sentence"])
                    else:
                        all_uncertainties[method].append(float(val))
                except Exception as exc:
                    logger.warning("Method %s failed on sample %s: %s",
                                   method, sample.get("id", "?"), type(exc).__name__)
                    all_uncertainties[method].append(None)
                method_perf[method].append(max(time.perf_counter() - t_m, 0.0))

        # Compute metrics per method
        computed_metrics: dict[str, dict] = {}
        final_uncertainties: dict[str, list[float]] = {}

        for method in methods:
            raw = all_uncertainties[method]
            valid = [(u, e) for u, e in zip(raw, all_errors) if u is not None]
            if not valid:
                logger.warning("Method %s: no valid uncertainties.", method)
                computed_metrics[method] = {}
                final_uncertainties[method] = []
                continue

            u_vals, e_vals = zip(*valid)
            u_list = list(u_vals)
            e_list = list(e_vals)
            final_uncertainties[method] = u_list

            summary = _metrics.summarize(u_list, e_list)
            computed_metrics[method] = {
                "auroc": summary.auroc,
                "ece": summary.ece,
                "correlation_pearson": summary.pearson,
                "correlation_spearman": summary.spearman,
                "error_rate": summary.error_rate,
                "total_samples": summary.total_samples,
                "uncertainty_stats": summary.uncertainty_stats,
                "avg_latency_s": (
                    sum(method_perf[method]) / len(method_perf[method])
                    if method_perf[method] else None
                ),
            }

        model_name = getattr(self.estimator.model, "name_or_path", "unknown")

        return BenchmarkResult(
            task=task,
            model_name=model_name,
            dataset=dataset_label,
            methods=methods,
            uncertainties=final_uncertainties,
            errors=all_errors,
            metrics=computed_metrics,
            token_level=token_level or None,
            performance={
                "avg_latency_s": (
                    sum(perf_latencies) / len(perf_latencies) if perf_latencies else None
                )
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, sample: dict[str, Any], task: str) -> str:
        inp = sample["input"]
        if task == "qa":
            return (
                f"Question:\n<input>\n{inp}\n</input>\n\n"
                "Provide ONLY the answer with no explanation, no punctuation, "
                "and no additional text. Just the answer.\nAnswer:"
            )
        if task == "math":
            return (
                f"Problem:\n<input>\n{inp}\n</input>\n\n"
                "Provide ONLY the final numerical answer. "
                "No explanation, no text, just the number.\nAnswer:"
            )
        if task == "summarization":
            return f"Article:\n<input>\n{inp[:500]}\n</input>\n\nSummary:"
        if task == "open-ended":
            return f"<input>\n{inp}\n</input>"
        raise ValueError(f"Unknown task: {task!r}")

    def _score_correctness(
        self, sample: dict[str, Any], prediction: str, task: str
    ) -> float:
        target = sample["target"]
        if task == "qa":
            return score_qa(target, prediction)
        if task == "math":
            return score_math(target, prediction)
        if task == "summarization":
            return score_summarization(target, prediction)
        if task == "open-ended":
            if self._judge_model is None:
                logger.warning(
                    "Judge model not loaded — returning 1.0 for open-ended correctness."
                )
                return 1.0
            return score_open_ended(sample, prediction, self._judge_model, self._judge_tokenizer)
        raise ValueError(f"Unknown task: {task!r}")

    def _run_method(
        self,
        method: str,
        prompt: str,
        outputs,
        prediction: str,
        task: str,
        max_new_tokens: int,
        temperature: float,
        num_samples: int,
    ):
        est = self.estimator
        if method == "entropy":
            return est.entropy_score(outputs)
        if method == "max_probability":
            return est.max_prob_score(outputs)
        if method == "sequence_probability":
            return est.sequence_prob_score(outputs)
        if method == "self_consistency":
            return est.self_consistency_score(
                prediction, prompt, num_samples, max_new_tokens, temperature
            )
        if method == "self_reflection":
            return est.self_reflection_score(prompt, prediction, task_type=task)
        if method == "bsdetector":
            return est.bsdetector_score(
                prompt, prediction, num_samples, max_new_tokens, temperature
            )
        raise ValueError(f"Unknown method: {method!r}")

    def _extract_token_texts(self, outputs) -> list[str]:
        input_len = outputs.sequences.shape[1] - len(outputs.scores)
        eos_id = self.estimator.tokenizer.eos_token_id
        tokens = []
        for i in range(len(outputs.scores)):
            tok_id = int(outputs.sequences[0, input_len + i].item())
            if eos_id is not None and tok_id == int(eos_id):
                continue
            tokens.append(self.estimator.tokenizer.convert_ids_to_tokens(tok_id))
        return tokens

    def _ensure_judge_loaded(self) -> None:
        if self._judge_model is not None:
            return
        from llm_uq._loader import _load_model
        logger.info("Loading Prometheus 2 judge: %s", self._judge_model_name)
        self._judge_model, self._judge_tokenizer = _load_model(self._judge_model_name)
        logger.info("Prometheus 2 judge ready.")

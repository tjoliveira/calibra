"""Dataset loading and validation for llm-uq benchmarks.

Built-in datasets are loaded from HuggingFace Hub. Custom datasets are
accepted as plain Python lists of dicts with ``"input"`` and ``"target"``
keys.

Example — built-in::

    from llm_uq.datasets import load_builtin
    data = load_builtin("qa", dataset="squad", max_samples=50)

Example — custom::

    from llm_uq.datasets import validate_custom
    data = validate_custom([
        {"input": "What year did WWII end?", "target": "1945"},
    ], task="qa")
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Canonical keys per task expected by Benchmark
_REQUIRED_KEYS = {"input", "target"}

# Short-name aliases → full HuggingFace repo ids
_DATASET_ALIASES: dict[str, str] = {
    "squad": "rajpurkar/squad",
    "gsm8k": "openai/gsm8k",
    "cnn_dailymail": "abisee/cnn_dailymail",
    "writingprompts": "euclaise/writingprompts",
}

# HuggingFace dataset defaults per task
_TASK_DEFAULTS: dict[str, dict] = {
    "qa": {
        "hf_name": "rajpurkar/squad",
        "hf_kwargs": {},
        "input_field": "question",
        "target_field": "answers",
    },
    "math": {
        "hf_name": "openai/gsm8k",
        "hf_kwargs": {"name": "main"},
        "input_field": "question",
        "target_field": "answer",
    },
    "summarization": {
        "hf_name": "abisee/cnn_dailymail",
        "hf_kwargs": {"name": "3.0.0"},
        "input_field": "article",
        "target_field": "highlights",
    },
    "open-ended": {
        "hf_name": "euclaise/writingprompts",
        "hf_kwargs": {},
        "input_field": "prompt",
        "target_field": "story",
    },
}


def load_builtin(
    task: str,
    dataset: str | None = None,
    split: str = "test",
    max_samples: int | None = None,
    seed: int = 42,
    trust_remote_code: bool = False,
) -> list[dict[str, Any]]:
    """Load a built-in HuggingFace dataset and normalise it.

    Args:
        task: One of ``"qa"``, ``"math"``, ``"summarization"``,
            ``"open-ended"``.
        dataset: HuggingFace dataset identifier. Defaults to the standard
            dataset for each task (SQuAD, GSM8K, CNN/DailyMail,
            WritingPrompts).
        split: Dataset split to load (e.g. ``"test"``, ``"validation"``).
        max_samples: Truncate to this many samples after shuffling.
        seed: Random seed for shuffling.
        trust_remote_code: Allow the dataset repo to execute arbitrary code
            on load. Disabled by default.

    Returns:
        List of dicts with keys ``"id"``, ``"input"``, and ``"target"``.

    Raises:
        ValueError: If ``task`` is not recognised.
    """
    if task not in _TASK_DEFAULTS:
        raise ValueError(
            f"Unknown task {task!r}. Choose from {list(_TASK_DEFAULTS)}."
        )

    from datasets import load_dataset  # deferred import — not required at import time

    cfg = _TASK_DEFAULTS[task]
    if dataset is None:
        hf_name = cfg["hf_name"]
        hf_kwargs = cfg["hf_kwargs"]
    else:
        hf_name = _DATASET_ALIASES.get(dataset, dataset)
        hf_kwargs = cfg["hf_kwargs"] if hf_name == cfg["hf_name"] else {}

    if trust_remote_code:
        logger.warning(
            "trust_remote_code=True: the dataset repo '%s' may execute arbitrary "
            "Python code on your machine. Only use this with repos you fully trust.",
            hf_name,
        )
    logger.info("Loading dataset: %s (split=%s)", hf_name, split)
    ds = load_dataset(hf_name, **hf_kwargs, split=split, trust_remote_code=trust_remote_code)

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    input_field = cfg["input_field"]
    target_field = cfg["target_field"]

    samples: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        raw_target = row.get(target_field, "")
        # SQuAD answers are a dict with an "answers" list
        if isinstance(raw_target, dict) and "text" in raw_target:
            target = raw_target["text"][0] if raw_target["text"] else ""
        elif isinstance(raw_target, list):
            target = raw_target[0] if raw_target else ""
        else:
            target = str(raw_target)

        samples.append({
            "id": row.get("id", str(i)),
            "input": str(row.get(input_field, "")),
            "target": target,
        })

    return samples


def validate_custom(
    data: list[dict[str, Any]],
    task: str,
) -> list[dict[str, Any]]:
    """Validate and normalise a user-provided dataset.

    Each element must be a dict with at least ``"input"`` and ``"target"``
    keys. Missing ``"id"`` fields are assigned sequential integers.

    Args:
        data: List of sample dicts.
        task: Task name (used only for error messages).

    Returns:
        The validated list, with ``"id"`` fields guaranteed.

    Raises:
        ValueError: If ``data`` is empty, not a list, or any element is
            missing ``"input"`` or ``"target"``.
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            "Custom dataset must be a non-empty list of dicts. Got "
            f"{type(data).__name__!r} with length {len(data) if isinstance(data, list) else 'N/A'}."
        )

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dict: {type(item).__name__!r}")
        missing = _REQUIRED_KEYS - item.keys()
        if missing:
            raise ValueError(
                f"Item {i} is missing required keys {missing}. "
                f"Each sample for task={task!r} must have 'input' and 'target'."
            )

    # Assign sequential ids where absent
    return [
        {**item, "id": item.get("id", str(i))}
        for i, item in enumerate(data)
    ]

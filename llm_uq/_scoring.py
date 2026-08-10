"""Internal answer-correctness scoring functions used by Benchmark.

All public functions in this module receive normalised sample dicts (with
``"input"`` and ``"target"`` keys) and a prediction string, and return a
correctness score in [0.0, 1.0].
"""

from __future__ import annotations

import logging
import re
import string
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation (from evaluator_improved_example.py, properly integrated)
# ---------------------------------------------------------------------------

_ROMAN_NUMERALS = {
    "I": 1, "V": 5, "X": 10, "L": 50,
    "C": 100, "D": 500, "M": 1000,
}

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}


def _roman_to_arabic(roman: str) -> str:
    roman = roman.upper().strip()
    if not all(c in _ROMAN_NUMERALS for c in roman):
        return roman
    total, prev = 0, 0
    for ch in reversed(roman):
        val = _ROMAN_NUMERALS[ch]
        total = total - val if val < prev else total + val
        prev = val
    return str(total)


def _word_to_number(word: str) -> str:
    return _NUMBER_WORDS.get(word.lower().strip(), word)


def _normalise_text(text: str) -> str:
    """Lower-case, strip punctuation, convert Roman numerals and number words."""
    cleaned = text.lower().strip().translate(str.maketrans("", "", string.punctuation))

    def _replace_roman(m: re.Match) -> str:
        return _roman_to_arabic(m.group(1))

    cleaned = re.sub(r"\b([IVXLCDM]+)\b", _replace_roman, cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(_word_to_number(w) for w in cleaned.split())
    return cleaned


def _contains_at_word_boundary(haystack: str, needle: str) -> bool:
    """Return True if ``needle`` appears in ``haystack`` as a whole word."""
    pattern = r"\b" + re.escape(needle) + r"\b"
    return bool(re.search(pattern, haystack, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------


def score_qa(ground_truth: str, prediction: str) -> float:
    """Binary QA correctness using word-boundary-aware substring match.

    Handles Roman numerals (e.g. ``"Super Bowl V"`` ≠ ``"Super Bowl VII"``)
    and number words (e.g. ``"one"`` == ``"1"``).

    Args:
        ground_truth: Reference answer string.
        prediction: Model-generated answer string.

    Returns:
        1.0 if ground truth is found in prediction, else 0.0.
    """
    gt = _normalise_text(ground_truth)
    pred = _normalise_text(prediction)
    return 1.0 if _contains_at_word_boundary(pred, gt) else 0.0


def score_math(ground_truth: str, prediction: str) -> float:
    """Binary math correctness — identical logic to :func:`score_qa`.

    Args:
        ground_truth: Reference numerical answer.
        prediction: Model-generated response.

    Returns:
        1.0 if ground truth is found in prediction, else 0.0.
    """
    return score_qa(ground_truth, prediction)


def score_summarization(ground_truth: str, prediction: str) -> float:
    """ROUGE-L F-measure between ground truth and predicted summary.

    Args:
        ground_truth: Reference summary.
        prediction: Generated summary.

    Returns:
        ROUGE-L F-measure in [0, 1]. Falls back to word-overlap ratio on
        failure.
    """
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(ground_truth, prediction)["rougeL"].fmeasure)
    except Exception as exc:
        logger.warning("ROUGE-L failed (%s); using word-overlap fallback.", exc)
        gt_words = set(ground_truth.lower().split())
        pred_words = set(prediction.lower().split())
        return len(gt_words & pred_words) / max(len(gt_words), 1)


def score_open_ended(
    sample: dict[str, Any],
    prediction: str,
    judge_model,
    judge_tokenizer,
) -> float:
    """Evaluate open-ended output quality using Prometheus 2.

    Args:
        sample: Sample dict with at least ``"input"`` and ``"target"`` keys.
        prediction: Model-generated response.
        judge_model: Loaded Prometheus 2 model.
        judge_tokenizer: Prometheus 2 tokenizer.

    Returns:
        Normalised quality score in [0.0, 1.0].

    Raises:
        ValueError: If ``"target"`` is absent from ``sample``.
    """
    import torch
    from llm_uq._compat import format_mistral_prompt

    instruction = sample.get("input", "")
    reference = sample.get("target", "")
    if not reference:
        raise ValueError(
            f"Sample {sample.get('id', '?')} is missing 'target' — required for Prometheus 2."
        )

    evaluation_prompt = (
        "###Task Description:\n"
        "An instruction (might include an Input inside it), a response to evaluate, "
        "a reference answer that gets a score of 5, and a score rubric representing "
        "a evaluation criteria are given.\n"
        "1. Write a detailed feedback that assess the quality of the response strictly "
        "based on the given score rubric, not evaluating in general.\n"
        "2. After writing a feedback, write a score that is an integer between 1 and 5. "
        "You should refer to the score rubric.\n"
        '3. The output format should look as follows: '
        '"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"\n'
        "4. Please do not generate any other opening, closing, and explanations.\n\n"
        f"###The instruction to evaluate:\n<instruction>\n{instruction}\n</instruction>\n\n"
        f"###Response to evaluate:\n<response>\n{prediction}\n</response>\n\n"
        f"###Reference Answer (Score 5):\n<reference>\n{reference}\n</reference>\n\n"
        "###Score Rubrics:\n"
        "[Quality of the response]\n"
        "Score 1: Completely irrelevant, incorrect, or fails to address the instruction.\n"
        "Score 2: Partially relevant but contains significant errors or omissions.\n"
        "Score 3: Mostly relevant and correct but lacks completeness or quality.\n"
        "Score 4: Relevant, correct, and complete with minor issues.\n"
        "Score 5: Highly relevant, correct, complete, and well-written.\n\n"
        "###Feedback: \n"
    )

    system_msg = (
        "You are a fair judge assistant tasked with providing clear, objective feedback "
        "based on specific criteria, ensuring each assessment reflects the absolute "
        "standards set for performance."
    )
    formatted = format_mistral_prompt(system_msg, evaluation_prompt)

    inputs = judge_tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=2048
    ).to(judge_model.device)

    with torch.no_grad():
        out = judge_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=judge_tokenizer.eos_token_id,
        )

    eval_text = judge_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    raw_score = _parse_prometheus_score(eval_text)
    return max(0.0, min(1.0, raw_score / 5.0))


def _parse_prometheus_score(text: str) -> float:
    """Extract the integer score (1-5) from Prometheus 2 output."""
    patterns = [
        r"\[RESULT\]\s*(\d+)",
        r"\[RESULT\]\s*\(?\s*(\d+)\s*\)?",
        r"(?:[Ss]core|[Rr]ating):\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            s = int(m.group(1))
            if 1 <= s <= 5:
                return float(s)

    for word in reversed(text.split()[-5:]):
        if word.isdigit() and 1 <= int(word) <= 5:
            return float(word)

    nums = re.findall(r"\b([1-5])\b", text)
    if nums:
        return float(nums[-1])

    logger.warning("Could not parse Prometheus score from judge output. Defaulting to 1.")
    return 1.0

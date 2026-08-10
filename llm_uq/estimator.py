"""Core uncertainty estimation for language model outputs."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from llm_uq._loader import _load_model

logger = logging.getLogger(__name__)

_ALL_METHODS = [
    "entropy",
    "max_probability",
    "sequence_probability",
    "self_consistency",
    "self_reflection",
    "bsdetector",
]


class Estimator:
    """Estimates output uncertainty for a HuggingFace causal language model.

    Supports six methods spanning two families:

    * **Probability-based** (one forward pass, greedy decode):
      ``entropy``, ``max_probability``, ``sequence_probability``
    * **Sampling-based** (N additional forward passes):
      ``self_consistency``, ``self_reflection``, ``bsdetector``

    ``self_consistency`` and ``bsdetector`` require a ``semantic_model``
    (a sentence-transformers identifier) to measure semantic agreement.

    Example::

        est = Estimator.from_pretrained("Qwen/Qwen3-8B")
        scores = est.estimate("What is the boiling point of water?",
                              methods=["entropy", "self_reflection"])
        # {"entropy": 1.23, "self_reflection": 0.0}
    """

    def __init__(
        self,
        model,
        tokenizer,
        semantic_model: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialise with an already-loaded model and tokenizer.

        Args:
            model: A HuggingFace ``AutoModelForCausalLM`` instance.
            tokenizer: Matching tokenizer.
            semantic_model: Sentence-transformers model name for
                ``self_consistency`` / ``bsdetector``.
            device: Device string. Inferred from the model when ``None``.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or str(next(model.parameters()).device)

        self._semantic_model = None
        if semantic_model:
            logger.info("Loading semantic model: %s", semantic_model)
            from sentence_transformers import SentenceTransformer
            self._semantic_model = SentenceTransformer(semantic_model)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str | None = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        semantic_model: str = "all-MiniLM-L6-v2",
        trust_remote_code: bool = False,
    ) -> "Estimator":
        """Load a HuggingFace model and return a ready-to-use Estimator.

        Args:
            model_name: HuggingFace model identifier (e.g. ``"Qwen/Qwen3-8B"``).
            device: Target device. Auto-detected when ``None``.
            load_in_8bit: Enable 8-bit quantization (requires CUDA +
                ``llm-uq[quantization]``).
            load_in_4bit: Enable 4-bit quantization (requires CUDA +
                ``llm-uq[quantization]``).
            semantic_model: Sentence-transformers model for
                ``self_consistency`` / ``bsdetector``. Pass ``None`` to
                disable those methods.
            trust_remote_code: Allow the model repo to execute arbitrary code
                on load. Disabled by default — only enable for repos you fully
                trust (e.g. some Qwen models require this).

        Returns:
            A fully initialised :class:`Estimator`.
        """
        model, tokenizer = _load_model(
            model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        )
        return cls(model, tokenizer, semantic_model=semantic_model, device=device)

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def estimate(
        self,
        prompt: str,
        methods: list[str] | None = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        num_samples: int = 5,
    ) -> dict[str, float]:
        """Estimate uncertainty for a single prompt.

        Generates a response using greedy decoding and computes the
        requested uncertainty methods. Probability-based methods share one
        forward pass; sampling-based methods require additional generation.

        Args:
            prompt: Input text.
            methods: Methods to compute. Defaults to all six methods.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature for sampling-based methods.
            num_samples: Number of samples for ``self_consistency`` /
                ``bsdetector``.

        Returns:
            Dict mapping method name to uncertainty score (higher = more
            uncertain). ``entropy`` and ``max_probability`` expose their
            sentence-level score under the method name; token-level values
            are accessible via :meth:`entropy_score` / :meth:`max_prob_score`
            directly.

        Raises:
            ValueError: If an unrecognised method name is requested.
        """
        if methods is None:
            methods = list(_ALL_METHODS)

        unknown = set(methods) - set(_ALL_METHODS)
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}. Choose from {_ALL_METHODS}.")

        if not 1 <= num_samples <= 100:
            raise ValueError(f"num_samples must be between 1 and 100, got {num_samples}.")

        _, outputs = self._generate_with_scores(prompt, max_new_tokens)
        reference = self._extract_text(outputs)

        result: dict[str, float] = {}

        if "entropy" in methods:
            result["entropy"] = self.entropy_score(outputs)["sentence"]
        if "max_probability" in methods:
            result["max_probability"] = self.max_prob_score(outputs)["sentence"]
        if "sequence_probability" in methods:
            result["sequence_probability"] = self.sequence_prob_score(outputs)

        if "self_consistency" in methods:
            try:
                result["self_consistency"] = self.self_consistency_score(
                    reference, prompt, num_samples, max_new_tokens, temperature
                )
            except ValueError as exc:
                logger.warning("self_consistency skipped: %s", exc)
                result["self_consistency"] = float("nan")

        if "self_reflection" in methods:
            try:
                result["self_reflection"] = self.self_reflection_score(prompt, reference)
            except Exception as exc:
                logger.warning("self_reflection skipped: %s", exc)
                result["self_reflection"] = float("nan")

        if "bsdetector" in methods:
            try:
                result["bsdetector"] = self.bsdetector_score(
                    prompt, reference, num_samples, max_new_tokens, temperature
                )
            except Exception as exc:
                logger.warning("bsdetector skipped: %s", exc)
                result["bsdetector"] = float("nan")

        return result

    def score_all(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        num_samples: int = 5,
    ) -> dict:
        """Compute all six uncertainty methods and return token-level data too.

        Returns a dict where probability-based methods map to
        ``{"sentence": float, "tokens": list[float]}`` and sampling-based
        methods map to a plain ``float``.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            num_samples: Number of samples for sampling-based methods.

        Returns:
            Dict mapping method names to uncertainty values.
        """
        _, outputs = self._generate_with_scores(prompt, max_new_tokens)
        reference = self._extract_text(outputs)

        result: dict = {}
        result["entropy"] = self.entropy_score(outputs)
        result["max_probability"] = self.max_prob_score(outputs)
        result["sequence_probability"] = self.sequence_prob_score(outputs)

        for method, fn in [
            ("self_consistency", lambda: self.self_consistency_score(
                reference, prompt, num_samples, max_new_tokens, temperature)),
            ("self_reflection", lambda: self.self_reflection_score(prompt, reference)),
            ("bsdetector", lambda: self.bsdetector_score(
                prompt, reference, num_samples, max_new_tokens, temperature)),
        ]:
            try:
                result[method] = fn()
            except Exception as exc:
                logger.warning("%s skipped: %s", method, exc)
                result[method] = None

        return result

    # ------------------------------------------------------------------
    # Probability-based methods
    # ------------------------------------------------------------------

    def entropy_score(self, outputs) -> dict[str, float | list[float]]:
        """Token-entropy uncertainty from a greedy-decode output.

        Args:
            outputs: Return value of :meth:`_generate_with_scores`.

        Returns:
            Dict with ``"sentence"`` (mean entropy) and ``"tokens"``
            (per-token entropy list).
        """
        input_len = outputs.sequences.shape[1] - len(outputs.scores)
        eos_id = self.tokenizer.eos_token_id
        token_entropies: list[float] = []

        for i, step_scores in enumerate(outputs.scores):
            tok_id = int(outputs.sequences[0, input_len + i].item())
            if eos_id is not None and tok_id == int(eos_id):
                continue
            log_p = F.log_softmax(step_scores, dim=-1)
            p = F.softmax(step_scores, dim=-1)
            h = -torch.sum(p * log_p, dim=-1)
            token_entropies.append(float(h.cpu().item()))

        return {
            "sentence": float(np.mean(token_entropies)) if token_entropies else 0.0,
            "tokens": token_entropies,
        }

    def max_prob_score(self, outputs) -> dict[str, float | list[float]]:
        """Max-probability uncertainty (1 − max token probability).

        Args:
            outputs: Return value of :meth:`_generate_with_scores`.

        Returns:
            Dict with ``"sentence"`` and ``"tokens"`` keys.
        """
        input_len = outputs.sequences.shape[1] - len(outputs.scores)
        eos_id = self.tokenizer.eos_token_id
        token_uncertainties: list[float] = []

        for i, step_scores in enumerate(outputs.scores):
            tok_id = int(outputs.sequences[0, input_len + i].item())
            if eos_id is not None and tok_id == int(eos_id):
                continue
            p = F.softmax(step_scores, dim=-1)
            uncertainty = float((1.0 - torch.max(p, dim=-1)[0]).cpu().item())
            token_uncertainties.append(uncertainty)

        return {
            "sentence": float(np.mean(token_uncertainties)) if token_uncertainties else 0.0,
            "tokens": token_uncertainties,
        }

    def sequence_prob_score(self, outputs) -> float:
        """Negative mean log-probability of the generated sequence.

        Args:
            outputs: Return value of :meth:`_generate_with_scores`.

        Returns:
            Uncertainty score ≥ 0 (higher = more uncertain).
        """
        input_len = outputs.sequences.shape[1] - len(outputs.scores)
        eos_id = self.tokenizer.eos_token_id
        log_probs: list[float] = []

        for i, step_scores in enumerate(outputs.scores):
            tok_id = outputs.sequences[0, input_len + i]
            if eos_id is not None and int(tok_id.item()) == int(eos_id):
                continue
            log_p = F.log_softmax(step_scores, dim=-1)
            log_probs.append(float(log_p[0, tok_id].item()))

        if not log_probs:
            logger.warning("sequence_probability: no valid tokens — returning 100.0")
            return 100.0

        return float(np.clip(-np.mean(log_probs), 0.0, 1000.0))

    # ------------------------------------------------------------------
    # Sampling-based methods
    # ------------------------------------------------------------------

    def self_consistency_score(
        self,
        reference_output: str,
        prompt: str,
        num_samples: int = 5,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> float:
        """Mean semantic distance between sampled outputs and a reference.

        Args:
            reference_output: Greedy-decoded reference answer.
            prompt: Original input prompt.
            num_samples: Number of stochastic samples to generate.
            max_new_tokens: Max tokens per sample.
            temperature: Sampling temperature.

        Returns:
            Mean cosine distance in [0, 2] (0 = identical, higher = more uncertain).

        Raises:
            ValueError: If no semantic model was provided at construction.
        """
        if self._semantic_model is None:
            raise ValueError(
                "A semantic_model is required for self_consistency. "
                "Pass semantic_model=<model_name> to Estimator.from_pretrained()."
            )

        samples = self._sample_outputs(prompt, num_samples, max_new_tokens, temperature)
        all_texts = [reference_output] + samples
        embeddings = self._semantic_model.encode(all_texts)
        ref_emb = embeddings[0]
        distances = [
            1.0 - float(np.dot(ref_emb, s) / (np.linalg.norm(ref_emb) * np.linalg.norm(s) + 1e-10))
            for s in embeddings[1:]
        ]
        return float(np.mean(distances)) if distances else 0.0

    def self_reflection_score(
        self,
        prompt: str,
        answer: str,
        task_type: str | None = None,
    ) -> float:
        """Ask the model to evaluate its own answer confidence.

        The model is presented with a structured (A/B/C) reflection prompt.
        Response ``A`` (correct) → 0.0, ``B`` (incorrect) → 1.0, ``C``
        (unsure) → 0.5.

        Args:
            prompt: Original question or instruction.
            answer: The model's previously generated response.
            task_type: Optional task hint (``"summarization"`` adjusts the
                prompt to focus on quality rather than correctness).

        Returns:
            Uncertainty score in {0.0, 0.5, 1.0}.

        Raises:
            ValueError: If the model response cannot be parsed as A, B, or C.
        """
        if task_type == "summarization":
            task_desc = (
                "Task: Review the summary above and determine your confidence in its quality.\n\n"
                "Instructions:\n"
                "1. Evaluate whether the summary is high quality, low quality, or if you are uncertain.\n"
                "2. Consider: completeness, accuracy, coherence, and relevance to the article.\n"
                "3. Choose exactly ONE of the following options:\n"
                "   (A) High quality — confident the summary is good\n"
                "   (B) Low quality — the summary is poor\n"
                "   (C) I am not sure — uncertain about summary quality\n"
                "4. IMPORTANT: Respond with EXACTLY ONE CHARACTER: A, B, or C.\n"
            )
        else:
            task_desc = (
                "Task: Review the proposed answer above and determine your confidence in its correctness.\n\n"
                "Instructions:\n"
                "1. Carefully evaluate whether the proposed answer is correct, incorrect, or uncertain.\n"
                "2. Choose exactly ONE of the following options:\n"
                "   (A) Correct — confident the answer is correct\n"
                "   (B) Incorrect — the answer is wrong\n"
                "   (C) I am not sure — uncertain about the answer\n"
                "3. IMPORTANT: Respond with EXACTLY ONE CHARACTER: A, B, or C.\n"
            )

        reflection_prompt = (
            f"<original_prompt>\n{prompt}\n</original_prompt>\n\n"
            f"<proposed_answer>\n{answer}\n</proposed_answer>\n\n"
            f"{task_desc}Answer:\n"
        )
        inputs = self.tokenizer(reflection_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=16,  # models like Qwen3 emit <think> tokens before the answer
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Search the whole response — the letter may follow whitespace or punctuation
        for ch in response.upper():
            if ch in ("A", "B", "C"):
                return {"A": 0.0, "B": 1.0, "C": 0.5}[ch]

        logger.warning("self_reflection: could not parse response %r — defaulting to 0.5", response)
        return 0.5

    def bsdetector_score(
        self,
        prompt: str,
        answer: str,
        num_samples: int = 5,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        beta: float = 0.7,
    ) -> float:
        """BSDetector: weighted combination of self-consistency and self-reflection.

        Args:
            prompt: Original question or instruction.
            answer: The model's previously generated response.
            num_samples: Number of stochastic samples for self-consistency.
            max_new_tokens: Max tokens per sample.
            temperature: Sampling temperature.
            beta: Weight for the self-consistency component (1 − beta goes to
                self-reflection). Default: 0.7.

        Returns:
            Combined uncertainty in [0, 1].

        Raises:
            ValueError: If both component methods fail.
        """
        consistency: float | None = None
        reflection: float | None = None

        try:
            consistency = self.self_consistency_score(
                answer, prompt, num_samples, max_new_tokens, temperature
            )
        except ValueError as exc:
            logger.warning("bsdetector: self_consistency failed — %s", exc)

        try:
            reflection = self.self_reflection_score(prompt, answer)
        except Exception as exc:
            logger.warning("bsdetector: self_reflection failed — %s", exc)

        if consistency is not None and reflection is not None:
            return beta * consistency + (1 - beta) * reflection
        if consistency is not None:
            return consistency
        if reflection is not None:
            return reflection

        raise ValueError(
            "bsdetector: both self_consistency and self_reflection failed. "
            "Ensure a semantic_model is available."
        )

    # ------------------------------------------------------------------
    # Internal generation helpers
    # ------------------------------------------------------------------

    def _generate_with_scores(self, prompt: str, max_new_tokens: int = 100):
        """Greedy decode with per-step logit scores.

        Returns:
            ``(inputs, outputs)`` where ``outputs`` has ``.scores`` and
            ``.sequences``.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return inputs, outputs

    def _extract_text(self, outputs) -> str:
        input_len = outputs.sequences.shape[1] - len(outputs.scores)
        return self.tokenizer.decode(
            outputs.sequences[0][input_len:], skip_special_tokens=True
        ).strip()

    def _sample_outputs(
        self,
        prompt: str,
        num_samples: int = 5,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> list[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = []
        with torch.no_grad():
            for _ in range(num_samples):
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                text = self.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                outputs.append(text)
        return outputs

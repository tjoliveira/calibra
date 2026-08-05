"""Tests for calibra.Estimator using the fake model/tokenizer fixtures."""

import math
import pytest
import torch

from calibra.estimator import Estimator


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_pretrained_uses_load_model(mocker, fake_model, fake_tokenizer):
    mocker.patch("calibra.estimator._load_model", return_value=(fake_model, fake_tokenizer))
    est = Estimator.from_pretrained("fake/model", semantic_model=None)
    assert isinstance(est, Estimator)


def test_constructor_sets_device(fake_model, fake_tokenizer):
    est = Estimator(fake_model, fake_tokenizer, device="cpu")
    assert est.device == "cpu"


# ---------------------------------------------------------------------------
# estimate() — single prompt high-level API
# ---------------------------------------------------------------------------


def test_estimate_returns_requested_methods(estimator_fixture):
    result = estimator_fixture.estimate("Hello", methods=["entropy", "max_probability"])
    assert "entropy" in result
    assert "max_probability" in result
    assert "self_consistency" not in result


def test_estimate_all_methods_no_semantic_model(estimator_fixture):
    # self_consistency and bsdetector should return nan when no semantic model
    result = estimator_fixture.estimate("Hello")
    assert "entropy" in result
    assert math.isnan(result["self_consistency"])


def test_estimate_unknown_method_raises(estimator_fixture):
    with pytest.raises(ValueError, match="Unknown methods"):
        estimator_fixture.estimate("Hello", methods=["nonexistent"])


# ---------------------------------------------------------------------------
# Probability-based scores
# ---------------------------------------------------------------------------


def test_entropy_score_returns_sentence_and_tokens(estimator_fixture, fake_outputs):
    result = estimator_fixture.entropy_score(fake_outputs)
    assert "sentence" in result
    assert "tokens" in result
    assert isinstance(result["sentence"], float)
    assert isinstance(result["tokens"], list)


def test_entropy_sentence_is_mean_of_tokens(estimator_fixture, fake_outputs):
    result = estimator_fixture.entropy_score(fake_outputs)
    if result["tokens"]:
        import numpy as np
        assert result["sentence"] == pytest.approx(float(np.mean(result["tokens"])), abs=1e-6)


def test_max_prob_score_in_range(estimator_fixture, fake_outputs):
    result = estimator_fixture.max_prob_score(fake_outputs)
    # uncertainty = 1 - max(softmax), so in [0, 1]
    assert 0.0 <= result["sentence"] <= 1.0
    for t in result["tokens"]:
        assert 0.0 <= t <= 1.0


def test_sequence_prob_score_non_negative(estimator_fixture, fake_outputs):
    score = estimator_fixture.sequence_prob_score(fake_outputs)
    assert score >= 0.0


# ---------------------------------------------------------------------------
# Sampling-based scores
# ---------------------------------------------------------------------------


def test_self_consistency_requires_semantic_model(estimator_fixture):
    with pytest.raises(ValueError, match="semantic_model"):
        estimator_fixture.self_consistency_score("ref", "prompt")


def test_self_reflection_parses_a(estimator_fixture, fake_tokenizer, mocker):
    # Make the model generate token for 'A'
    a_id = ord("A")

    def fake_generate(**kwargs):
        n = kwargs.get("input_ids").shape[1]
        return torch.tensor([[0] * n + [a_id]])

    mocker.patch.object(estimator_fixture.model, "generate", side_effect=fake_generate)
    score = estimator_fixture.self_reflection_score("Q", "answer")
    assert score == 0.0


def test_self_reflection_parses_b(estimator_fixture, mocker):
    b_id = ord("B")

    def fake_generate(**kwargs):
        n = kwargs.get("input_ids").shape[1]
        return torch.tensor([[0] * n + [b_id]])

    mocker.patch.object(estimator_fixture.model, "generate", side_effect=fake_generate)
    score = estimator_fixture.self_reflection_score("Q", "answer")
    assert score == 1.0


def test_self_reflection_parses_c(estimator_fixture, mocker):
    c_id = ord("C")

    def fake_generate(**kwargs):
        n = kwargs.get("input_ids").shape[1]
        return torch.tensor([[0] * n + [c_id]])

    mocker.patch.object(estimator_fixture.model, "generate", side_effect=fake_generate)
    score = estimator_fixture.self_reflection_score("Q", "answer")
    assert score == 0.5


def test_self_reflection_raises_on_unparseable(estimator_fixture, mocker):
    # Generate EOS (0) — tokenizer.decode will return empty string
    def fake_generate(**kwargs):
        n = kwargs.get("input_ids").shape[1]
        return torch.tensor([[0] * n + [0]])

    mocker.patch.object(estimator_fixture.model, "generate", side_effect=fake_generate)
    with pytest.raises(ValueError, match="parse"):
        estimator_fixture.self_reflection_score("Q", "answer")


# ---------------------------------------------------------------------------
# score_all
# ---------------------------------------------------------------------------


def test_score_all_returns_all_method_keys(estimator_fixture):
    result = estimator_fixture.score_all("Hello")
    for method in ("entropy", "max_probability", "sequence_probability",
                   "self_consistency", "self_reflection", "bsdetector"):
        assert method in result

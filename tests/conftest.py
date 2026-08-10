"""Shared fixtures for llm_uq tests."""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal fake model / tokenizer (no download required)
# ---------------------------------------------------------------------------


class _FakeBatch(dict):
    """A dict subclass that supports .to() so tests don't need real tensors."""

    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token_id = 0
    pad_token = "<pad>"
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None):
        # Encode as character ordinals, length=5, always
        ids = [min(ord(c), 127) for c in text[:5]] + [0] * max(0, 5 - len(text))
        input_ids = torch.tensor([ids])
        return _FakeBatch(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )

    def decode(self, ids, skip_special_tokens=True):
        chars = []
        for i in ids:
            i = int(i)
            if i == 0 and skip_special_tokens:
                continue
            if 32 <= i < 127:
                chars.append(chr(i))
        return "".join(chars)

    def convert_ids_to_tokens(self, idx: int) -> str:
        return f"tok{idx}"

    def from_pretrained(self, *a, **kw):
        return self


class _FakeOutput:
    """Mimics transformers GenerateOutput with scores and sequences."""

    def __init__(self, n_tokens: int = 4, vocab_size: int = 200):
        # sequences: [1, 5 + n_tokens]
        gen = torch.randint(1, vocab_size, (1, n_tokens))
        self.sequences = torch.cat([torch.zeros(1, 5, dtype=torch.long), gen], dim=1)
        # scores: list of n_tokens tensors, each [1, vocab_size]
        self.scores = [torch.randn(1, vocab_size) for _ in range(n_tokens)]


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name_or_path = "fake-model"
        # dummy parameter so next(model.parameters()).device works
        self._p = nn.Parameter(torch.zeros(1))

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        n = kwargs.get("max_new_tokens", 4)
        vocab = 200
        return_dict = kwargs.get("return_dict_in_generate", False)
        gen = torch.randint(1, vocab, (1, n))
        seqs = torch.cat([input_ids, gen], dim=1)

        if return_dict and kwargs.get("output_scores", False):
            return _FakeOutput(n, vocab)

        # Return plain tensor for sampling calls
        return seqs

    def forward(self, *a, **kw):
        pass


@pytest.fixture
def fake_tokenizer():
    return _FakeTokenizer()


@pytest.fixture
def fake_model():
    m = _FakeModel()
    m.eval()
    return m


@pytest.fixture
def fake_outputs():
    return _FakeOutput(n_tokens=4, vocab_size=200)


@pytest.fixture
def estimator_fixture(fake_model, fake_tokenizer):
    from llm_uq.estimator import Estimator
    return Estimator(fake_model, fake_tokenizer, semantic_model=None, device="cpu")

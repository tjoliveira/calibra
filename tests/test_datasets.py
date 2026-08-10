"""Tests for llm_uq.datasets."""

import pytest

from llm_uq.datasets import validate_custom, load_builtin


# ---------------------------------------------------------------------------
# validate_custom
# ---------------------------------------------------------------------------


def test_validate_custom_accepts_valid_list():
    data = [{"input": "Q", "target": "A"}]
    result = validate_custom(data, task="qa")
    assert len(result) == 1
    assert result[0]["input"] == "Q"
    assert result[0]["target"] == "A"


def test_validate_custom_assigns_ids():
    data = [{"input": "Q", "target": "A"}, {"input": "Q2", "target": "A2"}]
    result = validate_custom(data, task="qa")
    assert result[0]["id"] == "0"
    assert result[1]["id"] == "1"


def test_validate_custom_preserves_existing_id():
    data = [{"input": "Q", "target": "A", "id": "my-id"}]
    result = validate_custom(data, task="qa")
    assert result[0]["id"] == "my-id"


def test_validate_custom_rejects_empty_list():
    with pytest.raises(ValueError, match="non-empty"):
        validate_custom([], task="qa")


def test_validate_custom_rejects_non_list():
    with pytest.raises(ValueError):
        validate_custom({"input": "Q", "target": "A"}, task="qa")  # type: ignore


def test_validate_custom_rejects_missing_input_key():
    with pytest.raises(ValueError, match="input"):
        validate_custom([{"target": "A"}], task="qa")


def test_validate_custom_rejects_missing_target_key():
    with pytest.raises(ValueError, match="target"):
        validate_custom([{"input": "Q"}], task="qa")


def test_validate_custom_rejects_non_dict_items():
    with pytest.raises(ValueError):
        validate_custom(["Q"], task="qa")  # type: ignore


# ---------------------------------------------------------------------------
# load_builtin
# ---------------------------------------------------------------------------


def test_load_builtin_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        load_builtin("unknown_task")


def _inject_fake_hf_datasets(mocker, rows):
    """Inject a fake ``datasets`` module so load_builtin can be tested without HF."""
    import sys
    import unittest.mock as umock

    class _FakeDS(list):
        def shuffle(self, seed=None):
            return self

        def select(self, indices):
            return [self[i] for i in indices]

        def __len__(self):
            return list.__len__(self)

    fake_ds = _FakeDS(rows)
    fake_mod = umock.MagicMock()
    fake_mod.load_dataset.return_value = fake_ds
    mocker.patch.dict(sys.modules, {"datasets": fake_mod})
    return fake_mod


def test_load_builtin_qa_calls_hf(mocker):
    rows = [
        {"question": "Q1", "answers": {"text": ["A1"]}, "id": "1"},
        {"question": "Q2", "answers": {"text": ["A2"]}, "id": "2"},
    ]
    _inject_fake_hf_datasets(mocker, rows)
    result = load_builtin("qa", max_samples=2)
    assert len(result) == 2
    assert result[0]["input"] == "Q1"
    assert result[0]["target"] == "A1"
    assert "id" in result[0]


def test_load_builtin_truncates_to_max_samples(mocker):
    rows = [{"question": f"Q{i}", "answers": {"text": [f"A{i}"]}, "id": str(i)} for i in range(10)]

    import sys
    import unittest.mock as umock

    class _FakeDS(list):
        def shuffle(self, seed=None):
            return self

        def select(self, indices):
            return [self[i] for i in indices]

        def __len__(self):
            return list.__len__(self)

    fake_mod = umock.MagicMock()
    fake_mod.load_dataset.return_value = _FakeDS(rows)
    mocker.patch.dict(sys.modules, {"datasets": fake_mod})
    result = load_builtin("qa", max_samples=3)
    assert len(result) == 3

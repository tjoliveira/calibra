"""Tests for llm_uq.Benchmark."""

import pytest

from llm_uq.benchmark import Benchmark
from llm_uq.results import BenchmarkResult


def _make_bench(estimator_fixture):
    return Benchmark(estimator_fixture)


def _patch_run_method(mocker, bench, uncertainty=0.5):
    """Patch _run_method to return a fixed float for all methods."""
    mocker.patch.object(bench, "_run_method", return_value=uncertainty)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_judge_not_loaded_at_init(estimator_fixture):
    bench = Benchmark(estimator_fixture)
    assert bench._judge_model is None


# ---------------------------------------------------------------------------
# run() with custom dataset
# ---------------------------------------------------------------------------


def test_run_with_custom_list_dataset(mocker, estimator_fixture):
    bench = _make_bench(estimator_fixture)
    _patch_run_method(mocker, bench)
    data = [{"input": "Q", "target": "A", "id": "1"}]
    result = bench.run(task="qa", dataset=data, methods=["entropy"])
    assert isinstance(result, BenchmarkResult)
    assert result.dataset == "custom"
    assert result.task == "qa"


def test_run_validates_custom_data(estimator_fixture):
    bench = _make_bench(estimator_fixture)
    with pytest.raises(ValueError):
        bench.run(task="qa", dataset=[{"input": "Q"}])  # missing "target"


def test_run_returns_benchmark_result(mocker, estimator_fixture):
    bench = _make_bench(estimator_fixture)
    _patch_run_method(mocker, bench)
    data = [{"input": "Q1", "target": "A1"}, {"input": "Q2", "target": "A2"}]
    result = bench.run(task="qa", dataset=data, methods=["entropy", "max_probability"])
    assert isinstance(result, BenchmarkResult)
    assert "entropy" in result.methods
    assert "max_probability" in result.methods


def test_run_with_builtin_string_dataset(mocker, estimator_fixture):
    fake_data = [
        {"id": "1", "input": "What year?", "target": "1945"},
        {"id": "2", "input": "Which city?", "target": "Paris"},
    ]
    mocker.patch("llm_uq.benchmark.load_builtin", return_value=fake_data)
    bench = _make_bench(estimator_fixture)
    _patch_run_method(mocker, bench)
    result = bench.run(task="qa", dataset="squad", methods=["entropy"])
    assert result.dataset == "squad"


# ---------------------------------------------------------------------------
# BenchmarkResult properties
# ---------------------------------------------------------------------------


def test_result_auroc_property(mocker, estimator_fixture):
    bench = _make_bench(estimator_fixture)
    _patch_run_method(mocker, bench)
    data = [{"input": "Q", "target": "A"}]
    result = bench.run(task="qa", dataset=data, methods=["entropy"])
    auroc_dict = result.auroc
    assert "entropy" in auroc_dict
    assert isinstance(auroc_dict["entropy"], float)


def test_result_to_dict_is_json_serialisable(mocker, estimator_fixture):
    import json
    bench = _make_bench(estimator_fixture)
    _patch_run_method(mocker, bench)
    data = [{"input": "Q", "target": "A"}]
    result = bench.run(task="qa", dataset=data, methods=["entropy"])
    d = result.to_dict()
    # Should not raise
    json.dumps(d)


def test_result_to_json_writes_file(mocker, tmp_path, estimator_fixture):
    bench = _make_bench(estimator_fixture)
    _patch_run_method(mocker, bench)
    data = [{"input": "Q", "target": "A"}]
    result = bench.run(task="qa", dataset=data, methods=["entropy"])
    out = tmp_path / "out.json"
    result.to_json(str(out))
    assert out.exists()


# ---------------------------------------------------------------------------
# Scoring regression
# ---------------------------------------------------------------------------


def test_improved_qa_scoring_roman_numerals(mocker, estimator_fixture):
    """Super Bowl V should NOT match Super Bowl VII."""
    from llm_uq._scoring import score_qa
    assert score_qa("Super Bowl V", "Super Bowl VII") == 0.0
    assert score_qa("Super Bowl V", "The winner is Super Bowl V") == 1.0


def test_prometheus_not_loaded_when_not_needed(mocker, estimator_fixture):
    bench = _make_bench(estimator_fixture)
    load_spy = mocker.patch.object(bench, "_ensure_judge_loaded")
    mocker.patch("llm_uq.benchmark.load_builtin",
                 return_value=[{"input": "Q", "target": "A", "id": "1"}])
    mocker.patch.object(bench, "_run_method", return_value=0.5)
    bench.run(task="qa", dataset="squad", methods=["entropy"])
    load_spy.assert_not_called()

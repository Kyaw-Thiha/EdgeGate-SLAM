import pytest
import torch
import numpy as np
from edgegate.data.synthetic_generator import generate
from edgegate.solvers.pypose_solver import PyPoseSolver
from edgegate.training.evaluate import (
    evaluate_one_graph,
    evaluate_one_graph_classical,
    accumulate_metrics,
)


class DummyGNN(torch.nn.Module):
    """Return edge_label as confidence (perfect oracle) for testing."""
    def forward(self, data):
        if hasattr(data, "edge_label") and data.edge_label is not None:
            labels = data.edge_label
        else:
            labels = torch.ones(data.edge_index.shape[1])
        return labels.clone()


def _make_graph(seed: int = 0):
    return generate(
        num_poses=30,
        num_loop_closures=6,
        outlier_rate=30,
        outlier_structure="random",
        seed=seed,
        proximity_threshold=5.0,
    )


class TestEvaluateOneGraph:

    def test_returns_expected_keys(self):
        model = DummyGNN()
        solver = PyPoseSolver()
        g = _make_graph(0)
        result = evaluate_one_graph(model, solver, g)
        assert set(result.keys()) == {"tp", "fp", "fn", "ate"}

    def test_perfect_predictions_zero_fp_fn(self):
        """DummyGNN returns edge_label as confidence → perfect predictions."""
        model = DummyGNN()
        solver = PyPoseSolver()
        g = _make_graph(0)
        result = evaluate_one_graph(model, solver, g)
        assert result["tp"] >= 0
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_ate_is_float_when_gt_available(self):
        model = DummyGNN()
        solver = PyPoseSolver()
        g = _make_graph(1)
        result = evaluate_one_graph(model, solver, g)
        assert result["ate"] is not None
        assert isinstance(result["ate"], float)
        assert result["ate"] >= 0.0

    def test_graph_without_edge_labels(self):
        """Graph with None edge_label should not crash (tp=fp=fn=0)."""
        model = DummyGNN()
        solver = PyPoseSolver()
        g = _make_graph(2)
        g.edge_label = None
        result = evaluate_one_graph(model, solver, g)
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 0


class TestEvaluateClassical:

    def test_returns_ate(self):
        solver = PyPoseSolver()
        g = _make_graph(3)
        w = torch.ones(g.edge_index.shape[1])
        result = evaluate_one_graph_classical(solver, g, w)
        assert "ate" in result
        assert isinstance(result["ate"], float)
        assert result["ate"] >= 0.0

    def test_no_gt_gives_none_ate(self):
        solver = PyPoseSolver()
        g = _make_graph(4)
        g.gt_node_poses = None
        w = torch.ones(g.edge_index.shape[1])
        result = evaluate_one_graph_classical(solver, g, w)
        assert result["ate"] is None


class TestAccumulateMetrics:

    def test_aggregate_perfect_results(self):
        results = [
            {"tp": 2, "fp": 0, "fn": 0, "ate": 0.1},
            {"tp": 3, "fp": 0, "fn": 0, "ate": 0.2},
        ]
        agg = accumulate_metrics(results)
        assert agg["tp"] == 5
        assert agg["fp"] == 0
        assert agg["fn"] == 0
        assert agg["precision"] == 1.0
        assert agg["recall"] == 1.0
        assert agg["f1"] == 1.0
        assert agg["ate"] == pytest.approx(0.15)

    def test_aggregate_mixed_results(self):
        results = [
            {"tp": 1, "fp": 0, "fn": 1, "ate": None},
            {"tp": 0, "fp": 1, "fn": 1, "ate": 0.5},
        ]
        agg = accumulate_metrics(results)
        assert agg["tp"] == 1
        assert agg["fp"] == 1
        assert agg["fn"] == 2
        assert agg["precision"] == pytest.approx(0.5)
        assert agg["recall"] == pytest.approx(1 / 3)
        assert agg["f1"] == pytest.approx(0.4)
        assert agg["ate"] == pytest.approx(0.5)

    def test_jensen_inequality_avoided(self):
        """Accumulated F1 must NOT equal average of per-graph F1 scores."""
        # These are counts, not F1 — accumulate_metrics computes F1 from totals
        results = [
            {"tp": 1, "fp": 0, "fn": 1, "ate": None},  # per-graph: P=1.0, R=0.5 → F1=0.667
            {"tp": 0, "fp": 1, "fn": 1, "ate": None},  # per-graph: P=0.0, R=0.0 → F1=0.0
        ]
        agg = accumulate_metrics(results)
        # Accumulated: tp=1, fp=1, fn=2 → P=0.5, R=0.333 → F1=0.4
        assert agg["f1"] == pytest.approx(0.4)
        # Per-graph F1 average would be (0.667 + 0.0) / 2 = 0.333
        assert agg["f1"] != pytest.approx(0.333, abs=0.05)

    def test_no_labels_gives_none_f1(self):
        results = [{"tp": 0, "fp": 0, "fn": 0, "ate": 0.1}]
        agg = accumulate_metrics(results)
        assert agg["f1"] is None
        assert agg["precision"] is None
        assert agg["recall"] is None

    def test_no_ate_gives_none_ate(self):
        results = [{"tp": 1, "fp": 0, "fn": 0, "ate": None}]
        agg = accumulate_metrics(results)
        assert agg["ate"] is None

    def test_handles_mixed_keys_in_results(self):
        """Classical results (no tp/fp/fn keys) should be handled gracefully."""
        results = [{"ate": 0.3}, {"ate": 0.5}]
        agg = accumulate_metrics(results)
        assert agg["tp"] == 0
        assert agg["fp"] == 0
        assert agg["fn"] == 0
        assert agg["f1"] is None
        assert agg["ate"] == pytest.approx(0.4)


import pytest

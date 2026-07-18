from __future__ import annotations
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock

from edgegate.data.synthetic_generator import generate
from edgegate.training.evaluate import (
    evaluate_one_graph,
    evaluate_one_graph_classical,
    accumulate_metrics,
)
from edgegate.models.edgegate_gnn import EdgeGateGNN
from edgegate.solvers.pypose_solver import PyPoseSolver


def _make_model():
    return EdgeGateGNN(hidden_dim=16, num_layers=2, dropout=0.0)


def _make_graph():
    # 50 poses, segment_length=5 ensures enough turns for proximal pairs
    return generate(
        num_poses=50, num_loop_closures=4, outlier_rate=30,
        outlier_structure="random", seed=7, segment_length=5,
        proximity_threshold=5.0,
    )


def test_evaluate_one_graph_returns_new_keys():
    model = _make_model()
    solver = PyPoseSolver()
    graph = _make_graph()
    r = evaluate_one_graph(model, solver, graph)
    assert "final_cost" in r
    assert "num_iterations" in r
    assert "converged" in r
    assert "solve_time_s" in r
    assert "solver_failed" in r
    assert "lc_confidence" in r
    assert "lc_labels" in r


def test_evaluate_one_graph_lc_arrays_match_labels():
    model = _make_model()
    solver = PyPoseSolver()
    graph = _make_graph()
    r = evaluate_one_graph(model, solver, graph)
    assert r["lc_confidence"].shape == r["lc_labels"].shape
    assert len(r["lc_confidence"]) > 0


def test_evaluate_one_graph_solve_time_positive():
    model = _make_model()
    solver = PyPoseSolver()
    graph = _make_graph()
    r = evaluate_one_graph(model, solver, graph)
    assert r["solve_time_s"] >= 0.0
    assert not r["solver_failed"]
    assert r["final_cost"] is not None
    assert r["final_cost"] >= 0.0


def test_evaluate_one_graph_solver_failed_on_exception():
    """solver_failed=True and cost=None when solver raises RuntimeError."""
    model = _make_model()
    bad_solver = MagicMock()
    bad_solver.solve.side_effect = RuntimeError("Indeterminant linear system")
    graph = _make_graph()
    r = evaluate_one_graph(model, bad_solver, graph)
    assert r["solver_failed"] is True
    assert r["final_cost"] is None
    assert r["ate"] is None


def test_evaluate_one_graph_classical_returns_new_keys():
    solver = PyPoseSolver()
    graph = _make_graph()
    w = torch.ones(graph.edge_index.shape[1])
    r = evaluate_one_graph_classical(solver, graph, w)
    assert "final_cost" in r
    assert "solve_time_s" in r
    assert "solver_failed" in r


def test_accumulate_metrics_includes_new_keys():
    model = _make_model()
    solver = PyPoseSolver()
    graph = _make_graph()
    r1 = evaluate_one_graph(model, solver, graph)
    r2 = evaluate_one_graph(model, solver, graph)
    agg = accumulate_metrics([r1, r2])
    assert "final_cost" in agg
    assert "solve_time_s" in agg
    assert "converged_count" in agg
    assert "failed_count" in agg
    assert "num_iterations_mean" in agg
    assert agg["failed_count"] == 0


def test_accumulate_metrics_failed_count():
    """failed_count increments for each solver_failed=True result."""
    r_ok = {"tp": 1, "fp": 0, "fn": 0, "ate": 0.1, "final_cost": 5.0,
            "solve_time_s": 0.1, "num_iterations": 10, "converged": True,
            "solver_failed": False}
    r_fail = {"tp": 0, "fp": 0, "fn": 0, "ate": None, "final_cost": None,
              "solve_time_s": 0.05, "num_iterations": None, "converged": None,
              "solver_failed": True}
    agg = accumulate_metrics([r_ok, r_fail])
    assert agg["failed_count"] == 1
    assert agg["converged_count"] == 1

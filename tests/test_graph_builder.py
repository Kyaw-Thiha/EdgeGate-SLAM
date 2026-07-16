import numpy as np
import pytest
import torch

from edgegate.data.types import PoseGraph
from edgegate.data.graph_builder import to_pyg
from edgegate.data.se2_utils import compose, inverse_compose, angle_wrap
from edgegate.data.synthetic_generator import generate, ODOM_INFO, LC_INFO

# Shared defaults — num_poses large enough and proximity_threshold loose enough
# that any seed reliably produces sufficient proximal pairs.
_NUM_POSES = 200
_NUM_LC = 6
_OUTLIER_RATE = 33  # 33% of 6 = 2 outliers, 4 inliers
_E_ODOM = _NUM_POSES - 1
_E_TOTAL = _E_ODOM + _NUM_LC


def _gen(**kwargs):
    defaults = dict(
        num_poses=_NUM_POSES,
        num_loop_closures=_NUM_LC,
        outlier_rate=_OUTLIER_RATE,
        outlier_structure="random",
        seed=0,
        proximity_threshold=5.0,  # loose enough that any seed produces revisits
    )
    defaults.update(kwargs)
    return generate(**defaults)


# ── se2_utils ─────────────────────────────────────────────────────────────────

def test_compose_heading_east():
    pose = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(compose(pose, [1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-10)


def test_compose_heading_north():
    pose = np.array([0.0, 0.0, np.pi / 2])
    np.testing.assert_allclose(compose(pose, [1.0, 0.0, 0.0]), [0.0, 1.0, np.pi / 2], atol=1e-10)


def test_inverse_compose_heading_east():
    np.testing.assert_allclose(
        inverse_compose([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-10
    )


def test_inverse_compose_heading_north():
    # Robot at origin heading North; target 1 unit north → (1, 0, 0) in local frame
    np.testing.assert_allclose(
        inverse_compose([0.0, 0.0, np.pi / 2], [0.0, 1.0, np.pi / 2]),
        [1.0, 0.0, 0.0],
        atol=1e-10,
    )


def test_se2_round_trip():
    pi = np.array([1.0, 2.0, 0.3])
    pj = np.array([3.0, 4.0, 0.7])
    np.testing.assert_allclose(compose(pi, inverse_compose(pi, pj)), pj, atol=1e-10)


def test_angle_wrap_non_boundary():
    # Inputs that don't land on the ±π boundary have unambiguous expected values
    np.testing.assert_allclose(angle_wrap(2.5 * np.pi),  0.5 * np.pi, atol=1e-10)
    np.testing.assert_allclose(angle_wrap(-1.5 * np.pi), 0.5 * np.pi, atol=1e-10)
    np.testing.assert_allclose(angle_wrap(0.5), 0.5, atol=1e-10)


def test_angle_wrap_boundary():
    # 3π maps to the ±π boundary; either sign is a valid representation
    assert np.isclose(abs(angle_wrap(3 * np.pi)), np.pi)


# ── to_pyg ────────────────────────────────────────────────────────────────────

def _minimal_graph(with_label: bool = True) -> PoseGraph:
    return PoseGraph(
        node_init=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        edge_index=np.array([[0, 1, 0], [1, 2, 2]], dtype=np.int64),
        edge_measurement=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        edge_info=np.array([[500.0, 0.0, 0.0, 500.0, 0.0, 100.0]] * 3),
        edge_type=np.array([0, 0, 1], dtype=np.int64),
        edge_label=np.array([1.0, 1.0, 1.0], dtype=np.float32) if with_label else None,
    )


def test_to_pyg_shapes():
    data = to_pyg(_minimal_graph())
    assert data.x.shape == (3, 3)
    assert data.edge_index.shape == (2, 3)
    assert data.edge_attr.shape == (3, 6)
    assert data.edge_type.shape == (3,)
    assert data.edge_label.shape == (3,)


def test_to_pyg_edge_attr_ordering():
    # Non-trivial off-diagonals: only indices [0,3,5] (Ixx,Iyy,Iθθ) should appear
    graph = PoseGraph(
        node_init=np.zeros((2, 3)),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        edge_measurement=np.array([[1.0, 2.0, 3.0]]),
        edge_info=np.array([[500.0, 10.0, 20.0, 400.0, 30.0, 100.0]]),
        edge_type=np.array([0], dtype=np.int64),
    )
    data = to_pyg(graph)
    # Expected: [dx, dy, dθ, Ixx, Iyy, Iθθ] = [1, 2, 3, 500, 400, 100]
    torch.testing.assert_close(
        data.edge_attr, torch.tensor([[1.0, 2.0, 3.0, 500.0, 400.0, 100.0]])
    )


def test_to_pyg_dtypes():
    data = to_pyg(_minimal_graph())
    assert data.x.dtype == torch.float32
    assert data.edge_index.dtype == torch.int64
    assert data.edge_attr.dtype == torch.float32
    assert data.edge_type.dtype == torch.int64


def test_to_pyg_label_absent_when_none():
    data = to_pyg(_minimal_graph(with_label=False))
    assert "edge_label" not in data


def test_to_pyg_label_present_when_set():
    data = to_pyg(_minimal_graph(with_label=True))
    assert "edge_label" in data
    assert data.edge_label.dtype == torch.float32


# ── synthetic_generator ───────────────────────────────────────────────────────

def test_generate_shapes():
    g = _gen()
    assert g.node_init.shape == (_NUM_POSES, 3)
    assert g.edge_index.shape == (2, _E_TOTAL)
    assert g.edge_measurement.shape == (_E_TOTAL, 3)
    assert g.edge_info.shape == (_E_TOTAL, 6)
    assert g.edge_type.shape == (_E_TOTAL,)
    assert g.edge_label.shape == (_E_TOTAL,)


def test_generate_edge_type_split():
    g = _gen()
    assert np.all(g.edge_type[:_E_ODOM] == 0)
    assert np.all(g.edge_type[_E_ODOM:] == 1)


def test_generate_odom_edges_consecutive():
    g = _gen()
    gaps = g.edge_index[1, :_E_ODOM] - g.edge_index[0, :_E_ODOM]
    assert np.all(gaps == 1)


def test_generate_lc_edges_respect_min_gap():
    g = _gen(min_gap=5)
    lc_gaps = g.edge_index[1, _E_ODOM:] - g.edge_index[0, _E_ODOM:]
    assert np.all(lc_gaps >= 5)


def test_generate_odom_labels_inlier():
    g = _gen()
    assert np.all(g.edge_label[:_E_ODOM] == 1.0)


def test_generate_outlier_count():
    # 33% of 6 LC edges = 2 outliers, 4 inliers
    g = _gen()
    lc_labels = g.edge_label[_E_ODOM:]
    assert int((lc_labels == 0.0).sum()) == 2
    assert int((lc_labels == 1.0).sum()) == 4


def test_generate_reproducible():
    g1 = _gen(seed=42)
    g2 = _gen(seed=42)
    np.testing.assert_array_equal(g1.node_init, g2.node_init)
    np.testing.assert_array_equal(g1.edge_index, g2.edge_index)
    np.testing.assert_array_equal(g1.edge_measurement, g2.edge_measurement)


def test_generate_different_seeds():
    g1 = _gen(seed=0)
    g2 = _gen(seed=1)
    assert not np.array_equal(g1.node_init, g2.node_init)


def test_generate_zero_outliers():
    g = _gen(outlier_rate=0)
    assert np.all(g.edge_label == 1.0)


def test_generate_all_outliers():
    g = _gen(num_loop_closures=4, outlier_rate=100)
    lc_labels = g.edge_label[_E_ODOM : _E_ODOM + 4]
    assert np.all(lc_labels == 0.0)


def test_generate_info_matrices():
    g = _gen()
    assert np.all(g.edge_info[:_E_ODOM] == ODOM_INFO[np.newaxis])
    assert np.all(g.edge_info[_E_ODOM:] == LC_INFO[np.newaxis])


def test_generate_clustered_structure():
    g = _gen(outlier_structure="clustered", outlier_rate=50, num_loop_closures=4)
    lc_labels = g.edge_label[_E_ODOM : _E_ODOM + 4]
    assert int((lc_labels == 0.0).sum()) == 2


def test_generate_uniform_measurement():
    # Smoke test: runs without error
    g = _gen(outlier_measurement="uniform", outlier_rate=50, num_loop_closures=4)
    assert g.edge_label is not None


def test_generate_to_pyg_pipeline():
    g = _gen()
    data = to_pyg(g)
    assert data.x.shape == (_NUM_POSES, 3)
    assert data.edge_attr.shape == (_E_TOTAL, 6)
    assert "edge_label" in data

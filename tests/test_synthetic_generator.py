"""Tests for synthetic generator parameter extensions: info_scale, lc_ratio."""
import numpy as np
from edgegate.data.synthetic_generator import generate, LC_INFO


def test_info_scale_default_preserves_behavior():
    """Default info_scale=1.0 produces the same LC info as before."""
    g = generate(
        num_poses=100, num_loop_closures=10,
        outlier_rate=30, outlier_structure="random", seed=42,
    )
    lc_mask = g.edge_type == 1
    lc_info = g.edge_info[lc_mask]
    assert lc_info.shape[0] == 10
    np.testing.assert_array_equal(lc_info[0], LC_INFO)


def test_info_scale_multiplies_lc_info():
    """info_scale=F multiplies LC info by F, odometry info unchanged."""
    g = generate(
        num_poses=100, num_loop_closures=10,
        outlier_rate=30, outlier_structure="random", seed=42,
        info_scale=0.5,
    )
    lc_mask = g.edge_type == 1
    odom_mask = g.edge_type == 0
    np.testing.assert_array_almost_equal(
        g.edge_info[lc_mask][0], LC_INFO * 0.5
    )
    np.testing.assert_array_equal(
        g.edge_info[odom_mask][0],
        [500.0, 0.0, 0.0, 500.0, 0.0, 100.0],
    )


def test_info_scale_large_value():
    """info_scale=100 produces correctly scaled LC info."""
    g = generate(
        num_poses=100, num_loop_closures=10,
        outlier_rate=30, outlier_structure="random", seed=42,
        info_scale=100.0,
    )
    lc_mask = g.edge_type == 1
    np.testing.assert_array_almost_equal(
        g.edge_info[lc_mask][0], LC_INFO * 100.0
    )


def test_lc_ratio_default_preserves_num_loop_closures():
    """Default lc_ratio=None uses num_loop_closures directly."""
    g = generate(
        num_poses=100, num_loop_closures=20,
        outlier_rate=30, outlier_structure="random", seed=42,
    )
    lc_mask = g.edge_type == 1
    assert lc_mask.sum() == 20


def test_lc_ratio_overrides_num_loop_closures():
    """lc_ratio=N produces num_poses // N loop closures,
    ignoring the explicit num_loop_closures value."""
    g = generate(
        num_poses=100, num_loop_closures=999,
        outlier_rate=30, outlier_structure="random", seed=42,
        lc_ratio=5,
    )
    lc_mask = g.edge_type == 1
    assert lc_mask.sum() == 100 // 5


def test_lc_ratio_dense():
    """lc_ratio=3 -> ~1 LC per 3 poses, high outlier rate to stay within grid capacity."""
    g = generate(
        num_poses=100, num_loop_closures=10,
        outlier_rate=70, outlier_structure="random", seed=42,
        lc_ratio=3,
    )
    lc_mask = g.edge_type == 1
    assert lc_mask.sum() == 100 // 3


def test_lc_ratio_40_to_1():
    """lc_ratio=40 -> sparse LCs."""
    g = generate(
        num_poses=100, num_loop_closures=10,
        outlier_rate=30, outlier_structure="random", seed=42,
        lc_ratio=40,
    )
    lc_mask = g.edge_type == 1
    assert lc_mask.sum() == max(1, 100 // 40)


def test_lc_ratio_and_info_scale_compose():
    """lc_ratio and info_scale work together."""
    g = generate(
        num_poses=100, num_loop_closures=10,
        outlier_rate=30, outlier_structure="random", seed=42,
        lc_ratio=5, info_scale=0.5,
    )
    lc_mask = g.edge_type == 1
    assert lc_mask.sum() == 100 // 5
    np.testing.assert_array_almost_equal(
        g.edge_info[lc_mask][0], LC_INFO * 0.5
    )

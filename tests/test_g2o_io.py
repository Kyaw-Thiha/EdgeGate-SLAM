import textwrap
import numpy as np
import pytest
from edgegate.data.g2o_io import load_g2o, save_g2o

# Minimal 4-node graph:
#   odometry chain:  0→1→2→3  (consecutive IDs → type 0)
#   loop-closure:    0→3       (non-consecutive → type 1)
FIXTURE = textwrap.dedent("""\
    VERTEX_SE2 0 0.0 0.0 0.0
    VERTEX_SE2 1 1.0 0.0 0.0
    VERTEX_SE2 2 2.0 0.0 0.0
    VERTEX_SE2 3 2.0 1.0 1.5707963
    EDGE_SE2 0 1 1.0 0.0 0.0 500.0 0.0 0.0 500.0 0.0 100.0
    EDGE_SE2 1 2 1.0 0.0 0.0 500.0 0.0 0.0 500.0 0.0 100.0
    EDGE_SE2 2 3 0.0 1.0 1.5707963 500.0 0.0 0.0 500.0 0.0 100.0
    EDGE_SE2 0 3 2.0 1.0 1.5707963 100.0 0.0 0.0 100.0 0.0 50.0
""")


@pytest.fixture
def g2o_file(tmp_path):
    p = tmp_path / "test.g2o"
    p.write_text(FIXTURE)
    return p


def test_load_shapes(g2o_file):
    g = load_g2o(g2o_file)
    assert g.node_init.shape == (4, 3)
    assert g.edge_index.shape == (2, 4)
    assert g.edge_measurement.shape == (4, 3)
    assert g.edge_info.shape == (4, 6)
    assert g.edge_type.shape == (4,)


def test_load_node_values(g2o_file):
    g = load_g2o(g2o_file)
    np.testing.assert_allclose(g.node_init[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(g.node_init[3], [2.0, 1.0, 1.5707963])


def test_load_edge_measurement(g2o_file):
    g = load_g2o(g2o_file)
    np.testing.assert_allclose(g.edge_measurement[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(g.edge_measurement[3], [2.0, 1.0, 1.5707963])


def test_load_edge_info(g2o_file):
    g = load_g2o(g2o_file)
    np.testing.assert_allclose(g.edge_info[0], [500.0, 0.0, 0.0, 500.0, 0.0, 100.0])
    np.testing.assert_allclose(g.edge_info[3], [100.0, 0.0, 0.0, 100.0, 0.0, 50.0])


def test_edge_type_inference(g2o_file):
    g = load_g2o(g2o_file)
    # consecutive edges → odometry
    assert g.edge_type[0] == 0  # 0→1
    assert g.edge_type[1] == 0  # 1→2
    assert g.edge_type[2] == 0  # 2→3
    # non-consecutive → loop-closure
    assert g.edge_type[3] == 1  # 0→3


def test_no_labels(g2o_file):
    g = load_g2o(g2o_file)
    assert g.edge_label is None


def test_manifold(g2o_file):
    g = load_g2o(g2o_file)
    assert g.manifold == "SE2"


def test_round_trip(g2o_file, tmp_path):
    g = load_g2o(g2o_file)
    out = tmp_path / "out.g2o"
    save_g2o(g, out)
    g2 = load_g2o(out)
    np.testing.assert_allclose(g2.node_init, g.node_init)
    np.testing.assert_array_equal(g2.edge_index, g.edge_index)
    np.testing.assert_allclose(g2.edge_measurement, g.edge_measurement)
    np.testing.assert_allclose(g2.edge_info, g.edge_info)
    np.testing.assert_array_equal(g2.edge_type, g.edge_type)


def test_save_custom_poses(g2o_file, tmp_path):
    g = load_g2o(g2o_file)
    shifted = g.node_init + 0.1
    out = tmp_path / "shifted.g2o"
    save_g2o(g, out, poses=shifted)
    g2 = load_g2o(out)
    np.testing.assert_allclose(g2.node_init, shifted, atol=1e-8)


def test_unordered_vertices(tmp_path):
    # Vertices listed out of order — should still produce correct sorted node_init
    content = textwrap.dedent("""\
        VERTEX_SE2 2 2.0 0.0 0.0
        VERTEX_SE2 0 0.0 0.0 0.0
        VERTEX_SE2 1 1.0 0.0 0.0
        EDGE_SE2 0 1 1.0 0.0 0.0 500.0 0.0 0.0 500.0 0.0 100.0
        EDGE_SE2 1 2 1.0 0.0 0.0 500.0 0.0 0.0 500.0 0.0 100.0
    """)
    p = tmp_path / "unordered.g2o"
    p.write_text(content)
    g = load_g2o(p)
    assert g.node_init.shape == (3, 3)
    np.testing.assert_allclose(g.node_init[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(g.node_init[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(g.node_init[2], [2.0, 0.0, 0.0])


def test_comments_and_blank_lines_ignored(tmp_path):
    content = textwrap.dedent("""\
        # This is a comment
        VERTEX_SE2 0 0.0 0.0 0.0

        VERTEX_SE2 1 1.0 0.0 0.0
        # another comment
        EDGE_SE2 0 1 1.0 0.0 0.0 500.0 0.0 0.0 500.0 0.0 100.0
    """)
    p = tmp_path / "comments.g2o"
    p.write_text(content)
    g = load_g2o(p)
    assert g.node_init.shape == (2, 3)
    assert g.edge_index.shape == (2, 1)

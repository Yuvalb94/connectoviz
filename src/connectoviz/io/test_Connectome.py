import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

# Import Connectome and functions for testing
# from Connectome import Connectome  # <-- for flat-folder lazy testing
from connectoviz.core.connectome import Connectome  # <-- uncomment when back in package

# import Connectome as connectome_module  # for flat-folder patching

# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def dummy_con_mat():
    """Fixture: Dummy 4x4 connectivity matrix."""
    return np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]])


@pytest.fixture
def dummy_atlas():
    """Fixture: Dummy atlas DataFrame."""
    return pd.DataFrame({"node_index": [0, 1, 2, 3], "label": ["A", "B", "C", "D"]})


@pytest.fixture
def dummy_metadata():
    """Fixture: Dummy node metadata DataFrame."""
    return pd.DataFrame(
        {
            "node_index": [0, 1, 2, 3],
            "label": ["A", "B", "C", "D"],  # Add this!
            "node_name": ["A", "B", "C", "D"],
            "group": ["G1", "G2", "G1", "G2"],
            "Layer1": [1, 0, 1, 0],
            "Layer2": [0, 1, 0, 1],
        }
    )


@pytest.fixture
def layout_dict_basic():
    """Fixture: Basic layout dictionary for reordering."""
    return {
        "hemi": True,
        "other": True,
        "grouping": "group",
        "display_node_name": True,
        "display_group_name": False,
    }


@pytest.fixture
def layers_list():
    """Fixture: Dummy layers list."""
    return ["Layer1", "Layer2"]


# --------------------------
# Tests for Connectome class
# --------------------------


# @patch("Connectome.check_metadata")  # <-- flat-folder
@patch("connectoviz.io.parsers.check_metadata")  # <-- uncomment for package
# the same for merge_metadata
# @patch("Connectome.merge_metadata")  # <-- flat-folder
@patch("connectoviz.io.parsers.merge_metadata")  # <-- uncomment for package
def test_connectome_initialization(
    mock_check_metadata, mock_merge_metadata, dummy_con_mat, dummy_atlas, dummy_metadata
):
    """
    Test Connectome initialization with dummy data.
    """
    mock_check_metadata.return_value = dummy_metadata
    # Make sure mock_merge_metadata accepts 2 args and returns dummy_metadata
    mock_merge_metadata.side_effect = lambda *args, **kwargs: dummy_metadata

    conn = Connectome(dummy_con_mat, dummy_atlas, dummy_metadata, label_col="node_name")
    assert isinstance(conn.con_mat, np.ndarray)
    assert isinstance(conn.node_metadata, pd.DataFrame)
    assert conn.con_mat.shape[0] == conn.node_metadata.shape[0]


# -------- Patching handle_layout --------


# @patch("Connectome.handle_layout")  # <-- flat-folder
@patch(
    "connectoviz.utils.handle_layout_prefrences.handle_layout"
)  # <-- uncomment for package
def test_reorder_nodes_calls_handle_layout(
    mock_handle_layout, dummy_con_mat, dummy_atlas, dummy_metadata, layout_dict_basic
):
    """
    Test reorder_nodes: ensures handle_layout is called and updates merged_metadata.
    """
    mock_handle_layout.return_value = ({"All": dummy_metadata}, True, False, True)

    conn = Connectome(dummy_con_mat, dummy_atlas, dummy_metadata, label_col="node_name")
    conn.reorder_nodes(layout_dict_basic)

    mock_handle_layout.assert_called_once_with(dummy_metadata, layout_dict_basic)
    assert isinstance(conn.merged_metadata, dict)
    assert "All" in conn.merged_metadata


# -------- Patching handle_layers --------


# @patch("Connectome.handle_layers")  # <-- flat-folder
@patch(
    "connectoviz.utils.handle_layout_prefrences.handle_layers"
)  # <-- uncomment for package
def test_apply_layers_calls_handle_layers(
    mock_handle_layers, dummy_con_mat, dummy_atlas, dummy_metadata, layers_list
):
    """
    Test apply_layers: ensures handle_layers is called and updates merged_metadata.
    """
    mock_handle_layers.return_value = dummy_metadata[
        ["node_index", "node_name"] + layers_list
    ]

    conn = Connectome(dummy_con_mat, dummy_atlas, dummy_metadata, label_col="node_name")
    conn.merged_metadata = dummy_metadata  # Pretend reorder_nodes ran already
    conn.apply_layers(layers_list)

    mock_handle_layers.assert_called_once_with(dummy_metadata, layers_list)
    assert isinstance(conn.merged_metadata, pd.DataFrame)
    assert all(col in conn.merged_metadata.columns for col in layers_list)


# -------- Other tests --------


def test_to_numpy_returns_copy(dummy_con_mat, dummy_atlas, dummy_metadata):
    """
    Test to_numpy: should return a copy of the connectivity matrix.
    """
    conn = Connectome(dummy_con_mat, dummy_atlas, dummy_metadata, label_col="node_name")
    result = conn.to_numpy()
    assert np.array_equal(result, dummy_con_mat)
    assert result is not conn.con_mat  # Ensure it's a copy


def test_to_dataframe_returns_dataframe(dummy_con_mat, dummy_atlas, dummy_metadata):
    """
    Test to_dataframe: should return a DataFrame representation of the connectivity matrix.
    """
    conn = Connectome(dummy_con_mat, dummy_atlas, dummy_metadata, label_col="node_name")
    df = conn.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == dummy_con_mat.shape
    assert list(df.index) == list(dummy_metadata.index)


def test_apply_mask_applies_correctly(dummy_con_mat, dummy_atlas, dummy_metadata):
    """
    Test apply_mask: should apply a mask to the connectivity matrix.
    """
    conn = Connectome(dummy_con_mat, dummy_atlas, dummy_metadata, label_col="node_name")
    mask = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    conn.apply_mask(mask)
    expected = dummy_con_mat * mask
    assert np.array_equal(conn.con_mat, expected)

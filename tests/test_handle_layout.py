import pytest
import pandas as pd
import warnings
from connectoviz.utils.handle_layout_prefrences import (
    handle_hemisphere,
    handle_layout,
    reordering_all,
    handle_layers,
)
import connectoviz.utils.handle_layout_prefrences as handler

# --------------------------
# Mock check_layout_dict & check_layout_list_Multilayer for tests
# --------------------------


# Force them to always return True during testing
def mock_check_layout_dict(layout_dict, comb_mat):
    return True


def mock_check_layout_list_Multilayer(layers_list, comb_mat):
    return True


# Patch in the handler module


handler.check_layout_dict = mock_check_layout_dict
handler.check_layout_list_Multilayer = mock_check_layout_list_Multilayer

# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def sample_metadata():
    """Fixture: Example metadata DataFrame with node_index, node_name, and grouping columns."""
    return pd.DataFrame(
        {
            "node_index": [1, 2, 3, 4],
            "node_name": ["L_A", "R_B", "C", "L_D"],
            "grouping": ["G1", "G2", "G1", "G2"],
        }
    )


@pytest.fixture
def layout_dict_basic():
    """Fixture: Basic layout_dict with hemi enabled and grouping defined."""
    return {
        "hemi": True,
        "other": True,
        "grouping": "grouping",
        "display_node_name": True,
        "display_group_name": False,
    }


@pytest.fixture
def layers_list():
    """Fixture: Example list of layers."""
    return ["Layer1", "Layer2"]


# --------------------------
# Tests for handle_hemisphere
# --------------------------


def test_handle_hemisphere_with_hemi_column(sample_metadata):
    """
    Test handle_hemisphere: when a 'hemi' column already exists in metadata,
    it should be used as-is and return a DataFrame.
    """
    df = sample_metadata.copy()
    df["hemi"] = ["left", "right", "other", "left"]
    result = handle_hemisphere(df, {"other": True})
    assert isinstance(result, pd.DataFrame)
    assert "hemi" in result.columns
    assert set(result["hemi"].unique()) <= {"left", "right", "other"}


def test_handle_hemisphere_with_node_name(sample_metadata):
    """
    Test handle_hemisphere: if no 'hemi' column exists, infer hemisphere from node_name
    prefixes/suffixes and split metadata into left/right/other.
    """
    result = handle_hemisphere(sample_metadata, {"other": True})
    assert isinstance(result, tuple)
    right_df, left_df, other_df = result
    assert "hemi" in right_df.columns
    assert len(left_df) == 2  # L_A, L_D
    assert len(right_df) == 1  # R_B
    assert len(other_df) == 1  # C (doesn't match L_ or R_)


def test_handle_hemisphere_no_node_index_raises(sample_metadata):
    """
    Test handle_hemisphere: if 'node_index' column is missing, raises ValueError.
    """
    df = sample_metadata.drop(columns="node_index")
    with pytest.raises(ValueError, match="must contain a 'node_index' column"):
        handle_hemisphere(df, {"other": True})


# --------------------------
# Tests for handle_layout
# --------------------------


def test_handle_layout_returns_dict_and_bools(sample_metadata, layout_dict_basic):
    """
    Test handle_layout: should return a dict of DataFrames and the three boolean flags
    (display_node_name, display_group_name, hemi).
    """
    result = handle_layout(sample_metadata, layout_dict_basic)
    reordered_dict, display_node, display_group, hemi = result
    assert isinstance(reordered_dict, dict)
    assert "L" in reordered_dict and "R" in reordered_dict
    assert display_node is True
    assert display_group is False
    assert hemi is True


def test_handle_layout_invalid_layout_dict(sample_metadata):
    """
    Test handle_layout: if check_layout_dict fails (invalid layout_dict), raises ValueError.
    """
    # Use the original check_layout_dict instead of the mock
    import connectoviz.io.parsers as real_parser

    handler.check_layout_dict = real_parser.check_layout_dict

    bad_layout = {"wrong_key": True}
    with pytest.raises(ValueError):
        handle_layout(sample_metadata, bad_layout)


# --------------------------
# Tests for reordering_all
# --------------------------


def test_reordering_all_returns_expected(sample_metadata, layout_dict_basic):
    """
    Test reordering_all: should return a reordered combined metadata DataFrame
    and the three boolean flags.
    """
    reordered_df, display_node, display_group, hemi = reordering_all(
        sample_metadata, layout_dict_basic
    )
    assert isinstance(reordered_df, pd.DataFrame)
    assert set(reordered_df.columns) >= {"node_index", "node_name", "hemi"}
    assert display_node is True
    assert display_group is False
    assert hemi is True


def test_reordering_all_warns_on_none_layout(sample_metadata):
    """
    Test reordering_all: if layout_dict is None, raises UserWarning and returns
    original metadata with all flags False.
    """
    with warnings.catch_warnings(record=True) as w:
        reordered_df, node_flag, group_flag, hemi_flag = reordering_all(
            sample_metadata, None
        )
        assert any("Using default layout preferences" in str(w_.message) for w_ in w)
        pd.testing.assert_frame_equal(reordered_df, sample_metadata)
        assert node_flag is False
        assert group_flag is False
        assert hemi_flag is False


# --------------------------
# Tests for handle_layers
# --------------------------


def test_handle_layers_filters_columns(sample_metadata, layers_list):
    """
    Test handle_layers: should return a filtered DataFrame with only
    node_index, node_name, and specified layers.
    """
    comb_meta = sample_metadata.copy()
    comb_meta["Layer1"] = [1, 0, 1, 0]
    comb_meta["Layer2"] = [0, 1, 0, 1]
    filtered_df = handle_layers(comb_meta, layers_list)
    expected_columns = {"node_index", "node_name", "Layer1", "Layer2"}
    assert set(filtered_df.columns) == expected_columns


def test_handle_layers_invalid_layers_raises(sample_metadata):
    """
    Test handle_layers: if a layer is not in the DataFrame, raises ValueError.
    """
    with pytest.raises(ValueError, match="not in the combined metadata DataFrame"):
        handle_layers(sample_metadata, ["NonExistentLayer"])

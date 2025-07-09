import numpy as np
import pandas as pd
import pytest

# Import functions directly from parsers.py
from connectoviz.io.parsers import (
    parse_matrix,
    check_mask,
    masking,
    check_mapping,
    merge_metadata,
)


# ============================
# Tests for parse_matrix
# ============================


def test_parse_matrix_numpy_array():
    """
    Test parse_matrix with a NumPy array input.
    It should return the same array unchanged.
    """
    arr = np.array([[1, 2], [3, 4]])
    result = parse_matrix(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


def test_parse_matrix_pandas_dataframe():
    """
    Test parse_matrix with a Pandas DataFrame input.
    It should convert the DataFrame to a NumPy array.
    """
    df = pd.DataFrame([[1, 2], [3, 4]])
    result = parse_matrix(df)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, df.values)


def test_parse_matrix_list():
    """
    Test parse_matrix with a Python list of lists.
    It should convert it to a NumPy array.
    """
    data = [[1, 2], [3, 4]]
    result = parse_matrix(data)
    np.testing.assert_array_equal(result, np.array(data))


def test_parse_matrix_invalid_type():
    """
    Test parse_matrix with an unsupported input type (integer).
    Should raise TypeError.
    """
    with pytest.raises(TypeError):
        parse_matrix(12345)


# ============================
# Tests for check_mask
# ============================


def test_check_mask_valid():
    """
    Test check_mask with a valid 2D binary mask.
    Should return True.
    """
    mask = np.array([[1, 0], [0, 1]])
    assert check_mask(mask)


def test_check_mask_invalid_values():
    """
    Test check_mask with non-binary values in the mask.
    Should raise ValueError.
    """
    mask = np.array([[1, 2], [0, 1]])
    with pytest.raises(ValueError, match="Mask must contain only 0s and 1s"):
        check_mask(mask)


def test_check_mask_invalid_dim():
    """
    Test check_mask with a 1D array instead of 2D.
    Should raise ValueError.
    """
    mask = np.array([1, 0, 1])
    with pytest.raises(ValueError, match="Mask must be a 2D array"):
        check_mask(mask)


def test_check_mask_none():
    """
    Test check_mask with None as input.
    Should return False (mask not applied).
    """
    assert check_mask(None) is False


# ============================
# Tests for masking
# ============================


def test_masking_applies_mask():
    """
    Test masking when a valid binary mask is provided.
    Should apply the mask to the matrix.
    """
    mask = np.array([[1, 0], [0, 1]])
    matrix = np.array([[5, 6], [7, 8]])
    masked = masking(mask, matrix)
    expected = np.array([[5, 0], [0, 8]])
    np.testing.assert_array_equal(masked, expected)


def test_masking_no_mask_warns():
    """
    Test masking when mask is None.
    Should return the original matrix and raise a UserWarning.
    """
    matrix = np.array([[5, 6], [7, 8]])
    with pytest.warns(UserWarning, match="No mask provided"):
        result = masking(None, matrix)
        np.testing.assert_array_equal(result, matrix)


# ============================
# Tests for check_mapping
# ============================


def test_check_mapping_dict():
    """
    Test check_mapping with a valid dictionary input.
    Should return the same dictionary.
    """
    mapping = {"A": "Region1", "B": "Region2"}
    result = check_mapping(mapping=mapping)
    assert result == mapping


def test_check_mapping_dataframe():
    """
    Test check_mapping with a DataFrame containing node-to-label mapping.
    Should convert DataFrame to a dictionary.
    """
    df = pd.DataFrame({"Node": ["A", "B"], "Label": ["Region1", "Region2"]})
    result = check_mapping(mapping=df)
    expected = {"A": "Region1", "B": "Region2"}
    assert result == expected


def test_check_mapping_lists():
    """
    Test check_mapping with two lists: node_vec and label_vec.
    Should create and return a mapping dictionary.
    """
    nodes = ["A", "B"]
    labels = ["Region1", "Region2"]
    result = check_mapping(node_vec=nodes, label_vec=labels)
    assert result == dict(zip(nodes, labels))


def test_check_mapping_invalid_length():
    """
    Test check_mapping with node_vec and label_vec of different lengths.
    Should raise ValueError.
    """
    nodes = ["A"]
    labels = ["Region1", "Region2"]
    with pytest.raises(ValueError, match="must have the same length"):
        check_mapping(node_vec=nodes, label_vec=labels)


# ============================
# Tests for merge_metadata
# ============================


def test_merge_metadata_success():
    """
    Test merge_metadata with two DataFrames of matching rows and indices.
    Should merge them into one DataFrame.
    """
    df1 = pd.DataFrame({"A": [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({"B": [3, 4]}, index=[0, 1])
    merged = merge_metadata(df1, df2)
    expected = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=[0, 1])
    pd.testing.assert_frame_equal(merged, expected)


def test_merge_metadata_row_mismatch():
    """
    Test merge_metadata with DataFrames having different row counts.
    Should raise ValueError.
    """
    df1 = pd.DataFrame({"A": [1, 2]})
    df2 = pd.DataFrame({"B": [3, 4, 5]})
    with pytest.raises(ValueError, match="must have the same number of rows"):
        merge_metadata(df1, df2)


def test_merge_metadata_non_dataframe():
    """
    Test merge_metadata with a non-DataFrame input.
    Should raise TypeError.
    """
    with pytest.raises(TypeError, match="All inputs must be Pandas DataFrames"):
        merge_metadata(pd.DataFrame({"A": [1, 2]}), "not a dataframe")

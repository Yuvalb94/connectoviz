# connectoviz/io/parsers.py

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.io
import h5py
from pathlib import Path
from typing import Any, Union, Dict, List, Optional
import nibabel as nib
import warnings


# Function to parse a connectivity matrix from various formats
def parse_matrix(data: Any) -> np.ndarray:
    """
    Parse a connectivity matrix from various supported formats.

    Parameters
    ----------
    data : Any
        Input matrix or file path. Supported types include:
        - numpy.ndarray
        - pandas.DataFrame
        - scipy sparse matrix
        - list
        - str path to .npy, .npz, .mat, .h5, .csv, .tsv, .nii, .nii.gz, or .txt

    Returns
    -------
    np.ndarray
        Matrix converted to a NumPy array.

    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.DataFrame):
        # Convert DataFrame to NumPy array
        if data.index.name is not None:
            data = data.reset_index(drop=True)
        return data.to_numpy()

    elif sparse.issparse(data):
        return data.toarray()
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, str):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Matrix file not found: {data}")
        if data.endswith(".npy") or data.endswith(".npz"):
            return np.load(data)
        elif data.endswith(".mat"):
            mat = scipy.io.loadmat(data)
            # Filter out MATLAB metadata keys
            valid_keys = [k for k in mat.keys() if not k.startswith("__")]
            if not valid_keys:
                raise ValueError("No valid matrix key found in .mat file.")
            return mat[valid_keys[0]]

        elif data.endswith(".h5"):
            with h5py.File(data, "r") as f:
                keys = list(f.keys())
                if not keys:
                    raise ValueError("Empty HDF5 file.")
                return f[keys[0]][()]

        elif data.endswith(".csv"):
            return pd.read_csv(data, index_col=0).to_numpy()
        elif data.endswith(".tsv"):
            return pd.read_csv(data, sep="\t", index_col=0).to_numpy()
        elif data.endswith(".nii") or data.endswith(".nii.gz"):
            img = nib.load(data)
            return img.get_fdata()
        elif data.endswith(".txt"):
            return np.loadtxt(data)

    raise TypeError("Unsupported matrix input format.")


"""
Note:
    This file and project assumes metadata and atlas are already provided as pandas.DataFrame objects.
    File parsing and external loading must be handled before calling this function.
"""
##act from assumption that metadata is a DataFrame
# #handling all metadata formats that can be passed to the parser as input
# def parse_metadata(meta: Union[pd.DataFrame, Dict[str, List], str]) -> pd.DataFrame:
#     """Accept various formats and convert to Pandas DataFrame.
#     The expected formats for the metadata input are:
#     pandas.DataFrame, dict (with string keys and list values),
#     or str (file path to .csv,.tsv ,.json or .txt).
#     Returns a Pandas DataFrame representation of the metadata
#     or raises an error if the input format is unsupported.
#     """
#     if isinstance(meta, pd.DataFrame):
#         return meta
#     elif isinstance(meta, dict):
#         return pd.DataFrame(meta)
#     elif isinstance(meta, str):
#         meta = Path(meta)
#         if not meta.exists():
#             raise FileNotFoundError(f"metadata file not found: {meta}")
#         if meta.endswith(".csv"):
#             return pd.read_csv(meta, index_col=0)
#         elif meta.endswith(".tsv"):
#             return pd.read_csv(meta, sep="\t", index_col=0)
#         elif meta.endswith(".xlsx"):
#             return pd.read_excel(meta, index_col=0)
#         elif meta.endswith(".json"):
#             return pd.read_json(meta, orient='index')
#         elif meta.endswith(".txt"):
#             return pd.read_csv(meta, sep="\t", index_col=0)
#     raise TypeError("Unsupported metadata input format.")


# check mask
def check_mask(mask: Union[np.ndarray, None]) -> bool:
    """
    Check if the mask is a valid binary 2D NumPy array.

    Parameters
    ----------
    mask : np.ndarray or None
        The binary mask to validate (only 0s and 1s).

    Returns
    -------
    bool
        True if valid mask, False if None, raises otherwise.
    """

    if isinstance(mask, np.ndarray):
        # Check if the mask is a binary mask (0s and 1s)
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("Mask must contain only 0s and 1s .")
        # Check if the mask is only 2D
        if mask.ndim != 2:
            raise ValueError("Mask must be a 2D array.")
        return True
    elif mask is None:
        return False
    else:
        raise TypeError("Unsupported mask input format.")


def masking(mask: Union[np.ndarray, None], con_mat: np.ndarray) -> Optional[np.ndarray]:
    """
    Apply a binary mask to a connectivity matrix.

    Parameters
    ----------
    mask : np.ndarray or None
        A binary (0/1) mask of the same shape as con_mat.
    con_mat : np.ndarray
        Connectivity matrix to mask.

    Returns
    -------
    np.ndarray
        Masked connectivity matrix. Returns original if mask is None.

    Raises
    ------
    ValueError
        If the mask shape doesn't match the matrix.
    """

    mask_valid = check_mask(mask)
    if mask_valid:
        # check if mask in same shape as
        assert isinstance(mask, np.ndarray)  #  This to satisfie mypy
        if mask.shape != con_mat.shape:
            raise ValueError("Mask shape must match the connectivity matrix shape.")
        # Apply the mask to the connectivity matrix
        masked_matrix = con_mat * mask
        return masked_matrix
    else:
        # raise warning that no mask is provided
        warnings.warn(
            "No mask provided. Returning the original connectivity matrix.", UserWarning
        )
        return con_mat


def check_namings(col_names: List, df: pd.DataFrame) -> bool:
    """
    Check if the given column names exist in the DataFrame.

    Parameters
    ----------
    col_names : list of str
        List of column names to check.
    df : pd.DataFrame
        DataFrame in which to look for the column names.

    Returns
    -------
    bool
        True if all column names are found.


    """
    # Check if the column name is a string
    for col_name in col_names:
        if not isinstance(col_name, str):
            raise TypeError("Column name must be a string.")
        if col_name in df.columns:
            continue
        else:
            raise ValueError(
                f"Column '{col_name}' not found in DataFrame. Available columns: {df.columns.tolist()}"
            )
    return True


# fuction that checks if mapping exists and in the right format
def check_mapping(
    mapping: Optional[Union[pd.DataFrame, Dict[str, str]]] = None,
    node_vec: Optional[List[str]] = None,
    label_vec: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Processes mapping information either from a DataFrame/dict or from two separate lists.

    Parameters:
    - mapping: A DataFrame with two columns or a dictionary mapping node to label.
    - node_vec: A list of node names (used only if mapping is None).
    - label_vec: A list of labels corresponding to node_vec.

    Returns:
    - A dictionary mapping nodes to labels.

    Raises:
    - ValueError: if inputs are invalid or inconsistent.
    """
    if mapping is None:
        if node_vec is None or label_vec is None:
            raise ValueError(
                "If mapping is None, both node_vec and label_vec must be provided."
            )
        # check if both are in same length
        if len(node_vec) != len(label_vec):
            raise ValueError("node_vec and label_vec must have the same length.")
        # convert all values in label_vec to string
        label_vec = [str(label) for label in label_vec]
        # create a dictionary from node_vec and label_vec
        return dict(zip(node_vec, label_vec))

    if isinstance(mapping, pd.DataFrame):
        # if empty DataFrame, raise error
        if mapping.empty:
            raise ValueError("Mapping DataFrame cannot be empty.")

        if mapping.shape[1] != 2:
            raise ValueError("DataFrame mapping must have exactly two columns.")
        return dict(zip(mapping.iloc[:, 0], mapping.iloc[:, 1]))

    if isinstance(mapping, dict):
        return mapping

    raise TypeError("Mapping must be a DataFrame, dictionary, or None.")


def check_col_existence(cols_name: list) -> bool:
    """
    Validate that at least one matching column was found.

    Parameters
    ----------
    cols_name : list of str
        List of column names that matched a condition (e.g., containing "label").

    Returns
    -------
    bool
        Always returns True if the input is valid.


    """
    if cols_name is None:
        raise ValueError("No column with 'label' found in the atlas DataFrame.")
    # elif len(cols_name) > 1:
    #     raise ValueError(f"Multiple columns fitting found: {cols_name}. Please specify a the relevant col.")
    return True


###changing atlas and metedata to accept only DataFrame.
## add explanation in documantion abot it and that label_col is the column of his choosing to label the nodes
# by default (if not filled by user)  it is the first column that contains the word 'label' in it


# match mapping to atlas
def compare_mapping(
    mapping: Dict[str, str], atlas: pd.DataFrame
) -> tuple[Optional[str], str]:
    """
    Validate and match the mapping dictionary against the atlas DataFrame.

    Parameters
    ----------
    mapping : dict
        Dictionary mapping node index to label.
    atlas : pd.DataFrame
        Atlas containing node label and index columns.

    Returns
    -------
    list of str
        Inferred [index_col, label_col] if mapping matches atlas content.

    Raises
    ------
    TypeError, ValueError
        If mapping format or values do not align with atlas.
    """

    if not isinstance(mapping, dict):
        raise TypeError("Mapping must be a dictionary.")

    values = list(mapping.values())

    label_candidates = [
        col
        for col in atlas.columns
        if all(value in atlas[col].values for value in values)
    ]
    if not label_candidates:
        raise ValueError(
            "No matching column found in the atlas DataFrame for the mapping values."
        )
    label_col = label_candidates[0]

    index_candidates = [col for col in atlas.columns if "index" in col.lower()]
    index_col: Optional[str] = index_candidates[0] if index_candidates else None

    return index_col, label_col


def atlas_check(
    atlas: pd.DataFrame,
    index_col: Optional[Union[str, None]] = None,
    label_col: Optional[Union[str, None]] = None,
    mapping: Optional[dict] = None,
) -> tuple[bool, Optional[str], str]:
    """
    Validate atlas format and infer index and label columns if not provided.

    Parameters
    ----------
    atlas : pd.DataFrame
        Atlas DataFrame with node information.
    index_col : str or None, optional
        Column to use as index (matching node indices).
    label_col : str or None, optional
        Column to use as label (e.g., region name).
    mapping : dict, optional
        Dictionary mapping index to label. Overrides column inference.

    Returns
    -------
    bool
    True if atlas is valid.
    list[str or None]
        [index_col, label_col] values to be used.


    """
    # Check if atlas is a DataFrame
    if not isinstance(atlas, pd.DataFrame):
        raise TypeError("Atlas must be a Pandas DataFrame.")
    # Check if atlas is empty
    if atlas.empty:
        raise ValueError("Atlas DataFrame cannot be empty.")

    # If mapping is provided
    if mapping is not None:
        # Check mapping values against atlas
        index_col_r, label_col_r = compare_mapping(mapping, atlas)
        return True, index_col_r, label_col_r
    # in case no mapping is provided
    else:
        if index_col is None:
            # If no index_col is provided, check if any column contains 'index'
            index_cols = [col for col in atlas.columns if "index" in col.lower()]
            if check_col_existence(index_cols):
                index_col = index_cols[0]
            # Check if the atlas has a column with the word 'label' in it  and the same for index_col
        if label_col is None:
            # If no label_col is provided, check if any column contains 'label'
            label_cols = [col for col in atlas.columns if "label" in col.lower()]
            if check_col_existence(label_cols):
                label_col = label_cols[0]

        elif index_col is not None and label_col is not None:
            # If naming  is provided, check if it exists in the DataFrame
            if not check_namings([label_col, index_col], atlas):
                raise ValueError(
                    f"Specified label_col '{label_col}' not found in the atlas DataFrame."
                )
        # Ensure the label_col is a string
        if not isinstance(label_col, str):
            raise TypeError("label_col must be a string representing the column name.")
        return True, index_col, label_col


def check_metadata(
    metadata: pd.DataFrame,
    atlas: pd.DataFrame,
    index_col: Optional[Union[str, None]] = None,
    label_col: Optional[Union[str, None]] = None,
    mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """check if the metadata is in the correct format

    Parameters
    ----------
    metadata : pd.DataFrame
        _metadata DataFrame to be checked and validated of info about the nodes
    atlas : pd.DataFrame
        atlas DataFrame to be checked and validated of info about the nodes
    index_col : Optional[Union[str, None]], optional
        optional- index_col the user gives to fit to index in atlas , by default None
    label_col : Optional[Union[str, None]], optional
        optional- name of column in atlas the user wnats to label the nodes with, by default None
    mapping : Optional[dict], optional
        a dict that map index to label. overides use of label and index col, by default None

    Returns
    -------
    pd.DataFrame
        Returns the metadata DataFrame if all checks pass.


    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("Metadata must be a Pandas DataFrame.")
    # mapping overtakes the index_col and label_col
    if mapping is not None:
        # check if all values are strings
        if not all(isinstance(v, str) for v in mapping.values()):
            raise TypeError("All values in mapping must be strings.")
        # check mapping values against atlas
        _, index_col, label_col = atlas_check(
            atlas, index_col=None, label_col=None, mapping=mapping
        )

    # If index_col is provided, check if it exists in the DataFrame
    # check atlas for index_col and label_col
    _, index_col, label_col = atlas_check(
        atlas, index_col=index_col, label_col=label_col
    )
    # check if label_col exists in the metadata DataFrame
    if label_col not in metadata.columns:
        raise ValueError(
            f"Specified label_col '{label_col}' not found in the metadata DataFrame. Available columns: {metadata.columns.tolist()}"
        )
    # check if index_col exists in the metadata DataFrame
    if index_col is not None and index_col not in metadata.columns:
        # raise warning as index_col is optional and the naming may not fit the atlas naming
        raise ValueError(
            f"Warning: Specified index_col '{index_col}' not found in the metadata DataFrame. naming may not fit the atlas naming."
        )

    # ensurin label_col values match the atlas label_col values
    if not metadata[label_col].isin(atlas[label_col]).all():
        raise ValueError(
            f"Metadata '{label_col}' values do not match atlas '{label_col}' values. "
            f"Metadata values: {metadata[label_col].unique()}, "
            f"Atlas values: {atlas[label_col].unique()}"
        )
    # if all checks pass, return the metadata DataFrame
    return metadata


# func to merge all relevant metadata into a single DataFrame
def merge_metadata(*metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Merge multiple metadata DataFrames along columns with index alignment.

    Parameters
    ----------
    metadata : tuple of pd.DataFrame
        One or more DataFrames to merge. Must have the same index and number of rows.

    Returns
    -------
    pd.DataFrame
        Merged metadata DataFrame.


    """
    if not metadata:
        raise ValueError("At least one metadata DataFrame must be provided.")
    if not all(isinstance(m, pd.DataFrame) for m in metadata):
        # raise error
        raise TypeError("All inputs must be Pandas DataFrames.")

    # assuming metadatas[0] is the atlas and the rest are metadata DataFrames
    # check if all in same number of rows
    row_counts = [m.shape[0] for m in metadata]
    if len(set(row_counts)) != 1:
        raise ValueError("All metadata DataFrames must have the same number of rows.")
    # check if all have the same index
    indices = [m.index for m in metadata]
    if not all(idx.equals(indices[0]) for idx in indices):
        raise ValueError("All metadata DataFrames must have the same index.")

    # Concatenate DataFrames along columns, aligning by index
    merged = pd.concat(metadata, axis=1, join="outer")
    if merged.empty:
        raise ValueError("Merged DataFrame is empty. Check input metadata.")
    # if names of columns arent unique,raise error with name of the first column that is not unique
    if not merged.columns.is_unique:
        non_unique_cols = merged.columns[merged.columns.duplicated()].unique()
        # raise warning with the non-unique columns and inform of adding suffixes

        warnings.warn(
            f"Non-unique column names found: {non_unique_cols.tolist()}. Adding suffixes to make them unique.",
            UserWarning,
        )  # add suffixes to non-unique columns
        merged.columns = (
            pd.Series(merged.columns)
            .astype(str)
            .where(
                ~pd.Series(merged.columns).duplicated(),
                pd.Series(merged.columns).astype(str)
                + "_"
                + pd.Series(merged.columns).duplicated().cumsum().astype(str),
            )
            .values
        )

        # check again if names are unique
        if not merged.columns.is_unique:
            non_unique_cols = merged.columns[merged.columns.duplicated()].unique()
            # raise error with the non-unique columns
            raise ValueError(
                f"Non-unique column names found: {non_unique_cols.tolist()}"
            )

    return merged


# checking user prefrences validity


def check_layout_dict(layout_dict: Dict[str, Any], comb_mat: pd.DataFrame) -> bool:
    """
    Check if the provided layout_dict is a valid dictionary.

    Parameters
    ----------
    layout_dict : Dict[str, Any]
        Dictionary of layout preferences.
    comb_mat : pd.DataFrame
        Combined metadata with all relevant info except con matrix.

    Returns
    -------
    bool
        True if valid, raises ValueError otherwise.
    """
    if not isinstance(layout_dict, dict):
        raise TypeError("layout_dict must be a dictionary.")
    if not layout_dict:
        # raise error if empty
        raise ValueError(
            "layout_dict cannot be empty. Please provide valid preferences."
        )

    # Check required keys
    required_keys = [
        "hemi",
        "other",
        "grouping",
        "display_node_name",
        "display_group_name",
    ]
    missing_keys = [key for key in required_keys if key not in layout_dict]
    if missing_keys:
        raise ValueError(f"Missing required keys in layout_dict: {missing_keys}")

    # check thaere are no additional keys that are not in the required keys
    extra_keys = [key for key in layout_dict if key not in required_keys]
    if extra_keys:
        raise ValueError(
            f"Extra keys found in layout_dict that are not required: {extra_keys}"
        )

    # check if all keys are strings
    if not all(isinstance(key, str) for key in layout_dict.keys()):
        raise TypeError("All keys in layout_dict must be strings.")

    # check if all values are booleans or strings
    if not all(
        isinstance(value, (bool, str, type(None))) for value in layout_dict.values()
    ):
        raise TypeError("All values in layout_dict must be booleans or strings.")

    # if all checks pass, return True
    return True


def check_layout_list_Multilayer(
    layers_list: list[str,], comb_mat: pd.DataFrame
) -> bool:
    """
    Check if the provided layers_list is a valid list of dictionaries.

    Parameters
    ----------
    layers_list : List[str,]
        List of features that will be used for  representing layers.
    comb_mat : pd.DataFrame
        Combined metadata with all relevant info except con matrix.


    Returns
    -------
    bool
        True if valid, raises ValueError otherwise.
    """
    if not isinstance(layers_list, list):
        raise TypeError("layers_list must be a list.")
    if not layers_list:
        # raise warning and return False
        warnings.warn("layers_list is empty. No layers will be applied.", UserWarning)
        return False
    # check if all elements in layers_list are strings
    if not all(isinstance(layer, str) for layer in layers_list):
        raise TypeError("All elements in layers_list must be strings.")
    # check if all layers are in the combined metadata DataFrame  as columns
    for layer in layers_list:
        if layer not in comb_mat.columns:
            raise ValueError(
                f"Layer '{layer}' not found in the combined metadata DataFrame. Available columns: {comb_mat.columns.tolist()}"
            )
    # if all checks pass, return True
    return True

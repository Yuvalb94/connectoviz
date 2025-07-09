# handling layout prefrences input
from typing import Dict, Any, Union, Optional
import pandas as pd
from connectoviz.io.parsers import check_layout_list_Multilayer, check_layout_dict
import warnings


def handle_hemisphere(
    combined_metadata: pd.DataFrame, layout_dict: Dict[str, Any]
) -> Union[
    pd.DataFrame,
    tuple[pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
]:
    """
    Handle the hemisphere preferences and return the combined metadata DataFrame with the hemisphere column.

    Parameters
    ----------
    combined_metadata : pd.DataFrame
        The combined metadata DataFrame.
    layout_dict : Dict[str, Any]
        The layout preferences dictionary.

    Returns
    -------
    pd.DataFrame
        The combined metadata DataFrame with the hemi column added.
    """
    # f
    if "node_index" not in combined_metadata.columns:
        raise ValueError(
            "The combined metadata DataFrame must contain a 'node_index' column."
        )
    nodal_name = layout_dict.get("node_name", "node_name")
    if nodal_name not in combined_metadata.columns:
        nodal_name = "node_name"
    if nodal_name not in combined_metadata.columns:
        raise ValueError(
            f"The combined metadata DataFrame must contain a '{nodal_name}' column."
        )
    # check if there is 'hemispheric' or hemi column
    if "hemispheric" in combined_metadata.columns:
        # set as hemi coloumn
        combined_metadata["hemi"] = combined_metadata["hemispheric"]
    # check it its called 'hemisphere' or 'Hemi'
    elif "hemisphere" in combined_metadata.columns:
        # set as hemi coloumn
        combined_metadata["hemi"] = combined_metadata["hemisphere"]
    elif "Hemi" in combined_metadata.columns:
        # set as hemi coloumn
        combined_metadata["hemi"] = combined_metadata["Hemi"]
    else:
        combined_metadata["hemi"] = combined_metadata[nodal_name].apply(
            lambda x: (
                "left"
                if x.startswith("L_") or x.endswith("_L")
                else "right" if x.startswith("R_") or x.endswith("_R") else "other"
            )
        )
    # fill empty values with 'other'
    combined_metadata["hemi"] = combined_metadata["hemi"].fillna("other")

    # remove columns that arent node_index, node_name, hemi, and group_name
    # go over all rows and change the  hemi coloumn values to 'left', 'right', or 'other'
    for row in combined_metadata.itertuples():
        # if left right or other -keep it
        if row.hemi not in ["left", "right", "other"]:
            # if value in row and in hemi column starts with L or ends with _L - set to left
            if combined_metadata.at[row.Index, "hemi"].startswith("L"):
                combined_metadata.at[row.Index, "hemi"] = "left"
            # if it starts with R or ends with _R - set to right
            elif combined_metadata.at[row.Index, "hemi"].startswith("R"):
                combined_metadata.at[row.Index, "hemi"] = "right"
            else:
                combined_metadata.at[row.Index, "hemi"] = "other"
    # divide to 3 matrices bsaed on hemi column
    right_metadata = combined_metadata[combined_metadata["hemi"] == "right"]
    left_metadata = combined_metadata[combined_metadata["hemi"] == "left"]
    other_metadata = combined_metadata[combined_metadata["hemi"] == "other"]

    if not layout_dict.get("other", False):
        # return just the right and left metadata
        return (right_metadata, left_metadata)
    elif layout_dict["other"]:
        # return all three metadata DataFrames as a tuple
        return (right_metadata, left_metadata, other_metadata)

    return combined_metadata


def handle_layout(
    combined_metadata: pd.DataFrame, layout_dict: Dict[str, Any]
) -> tuple[Dict[str, pd.DataFrame], bool, bool, bool]:
    """
    Handle the layout preferences and return the combined metadata DataFrame with the other features in a dict.

    Parameters
    ----------
    combined_metadata : pd.DataFrame
        The combined metadata DataFrame.
    layout_dict : Dict[str, Any]
        The layout preferences dictionary.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing the combined metadata DataFrame with the other features.
    display_node_name : bool
        A boolean indicating whether to display the node names.
    display_group_name : bool
        A boolean indicating whether to display the group names.
    hemi : bool
        A boolean indicating whether to handle the hemisphere preferences.
    """
    # check if grouping is in the layout_dict
    try:
        check_layout_dict(layout_dict, combined_metadata)
    except Exception:
        raise ValueError("layout_dict is not valid")

    grouping = layout_dict.get("grouping", None)
    hemi = layout_dict.get("hemi", True)
    other = layout_dict.get("other", False)
    nodal_name = layout_dict.get("node_name", "node_name")

    display_group_name = layout_dict.get("display_group_name", False)

    display_node_name = layout_dict.get("display_node_name", False)
    if hemi:
        combined_metadatas = handle_hemisphere(combined_metadata, layout_dict)
        # sort each metadata DataFrame by the grouping column if it exists
        if isinstance(combined_metadatas, tuple):
            combined_metadatas = list(combined_metadatas)  # make mutable
            for i, metadata in enumerate(combined_metadatas):
                if grouping is not None and grouping in metadata.columns:
                    combined_metadatas[i] = metadata.sort_values(by=grouping)

                # if node_name value  is in columns, rename it to node_name
                if nodal_name in metadata.columns:
                    # metadata.rename(columns={nodal_name: "node_name"}, inplace=True)
                    combined_metadatas[i] = metadata.rename(
                        columns={nodal_name: "node_name"}
                    )

            # unpack the tuple into separate variables
            right_metadata, left_metadata = combined_metadatas[:2]

            if other:
                other_metadata = combined_metadatas[2]
                # return all three metadata DataFrames as a dict and the display boolians
                return (
                    {"L": right_metadata, "R": left_metadata, "Other": other_metadata},
                    display_node_name,
                    display_group_name,
                    hemi,
                )

            return (
                {"L": right_metadata, "R": left_metadata},
                display_node_name,
                display_group_name,
                hemi,
            )

    if grouping is not None:
        if not hemi:
            # sort the DataFrame by the grouping column
            combined_metadata = combined_metadata.sort_values(by=grouping)
            # change the grouping column name to 'group_name'
            combined_metadata.rename(columns={grouping: "group_name"}, inplace=True)

            return (
                {"All": combined_metadata},
                display_node_name,
                display_group_name,
                hemi,
            )
    else:
        # if grouping is None, just return the combined metadata DataFrame
        # keep columns node_index, node_name, and hemi
        combined_metadata = (
            combined_metadata[["node_index", "node_name", "hemi"]]
            if "hemi" in combined_metadata.columns
            else combined_metadata[["node_index", "node_name"]]
        )
        return {"All": combined_metadata}, display_node_name, display_group_name, hemi
    # just for Mypy to shut up
    return {"All": combined_metadata}, display_node_name, display_group_name, hemi


def reordering_all(
    combined_metadata: pd.DataFrame, layout_dict: Optional[Dict[str, Any]]
) -> tuple[Dict[str, pd.DataFrame], bool, bool, bool]:
    """
    Reorder the combined metadata DataFrame based on the layout preferences.

    Parameters
    ----------
    combined_metadata : pd.DataFrame
        The combined metadata DataFrame.
    layout_dict : Optional[Dict[str, Any]]
        The layout preferences dictionary.
    If None, default layout preferences will be used.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing the reordered combined metadata DataFrame.

    bool
        A boolean indicating whether to display the node names.
    bool
        A boolean indicating whether to display the group names.
    bool
        A boolean indicating whether to handle the hemisphere preferences.
    """
    if layout_dict is None:
        warnings.warn(
            "The layout_dict is None. Using default layout preferences.",
            UserWarning,
        )
        return (
            combined_metadata,
            False,
            False,
            False,
        )  # type: ignore[unreachable]

    # to make mypy happy
    # assert layout_dict is not None

    reorderd_dfs, bool_display_node, bool_display_group, bool_hemi = handle_layout(
        combined_metadata, layout_dict
    )
    # check if reorderd_dfs is a dict
    if not isinstance(reorderd_dfs, dict):
        raise ValueError(
            "The handle_layout function must return a dictionary of DataFrames."
        )

    reordered_combined_metadata = pd.concat(reorderd_dfs.values(), ignore_index=True)
    return (
        reordered_combined_metadata,
        bool_display_node,
        bool_display_group,
        bool_hemi,
    )


def handle_layers(
    comb_meta: dict[str, pd.DataFrame], layers_list: list[str,], label: str
) -> Dict[str, pd.DataFrame]:
    """
    Handle the layers preferences and return the combined metadata DataFrame with the layers column.
    Parameters
    ----------
    comb_meta : dict[str,pd.DataFrame]
    the combined metadata DataFrames packed in a dictionary.

    layers_list : list[str]
        The list of layers to handle.
    label : str
        The label column with numbers of ROIs in the metadata DataFrame.

    Returns
    -------
    pd.DataFrame
        The combined metadata DataFrame filtered based on layers column added.
    """
    # use check_layout_list_Multilayer to check the validity of the layers_list
    validity_bool = check_layout_list_Multilayer(layers_list, comb_meta)
    if not validity_bool:
        raise ValueError("The layers_list is not valid. Please check the layers names.")

    # check if group_name in the columns of comb_meta
    initial_lis = ["node_index", "node_name", "group_name", "hemi", label]
    if "L" in comb_meta.keys():
        if "group_name" not in comb_meta["L"].columns:
            # remove group_name from the initial_lis
            initial_lis.remove("group_name")
        if "hemi" not in comb_meta["L"].columns:
            # remove hemi from the initial_lis
            initial_lis.remove("hemi")
    else:
        if "group_name" not in comb_meta["All"].columns:
            # remove group_name from the initial_lis
            initial_lis.remove("group_name")
        if "hemi" not in comb_meta["All"].columns:
            # remove hemi from the initial_lis
            initial_lis.remove("hemi")
    filterd_dict = {}
    try:
        # run on keys and values of comb_meta
        for key, df in comb_meta.items():
            # check if all layers in the layers_list are in the DataFrame
            if all(layer in df.columns for layer in layers_list):
                # filter the DataFrame by the layers_list and initial_lis
                filterd_dict[key] = df[list(set(initial_lis + layers_list))]
            else:
                raise ValueError(
                    f"One or more layers not found in the combined metadata DataFrame for key: {key}"
                )

    except ValueError as e:
        raise ValueError(
            f"Error while filtering the combined metadata DataFrame: {e}"
        ) from e

    return filterd_dict


# creating a dictionary of ROIs grouped by a grouping variable


def create_dictionary(grouped_by_hemisphere, grouping_name, label, roi_names):
    """
    This function groups the ROIs according to a grouping variable within hemisphere.

    Parameters
    ----------
    grouped_by_hemisphere: DataFrame
        Part of the matrix that related to specific hemisphere.
    grouping_name: string
        The name of variable by which we will group ROIs in the graph.
        the name of grouping variaible must be
    label: string
        Name of column in the atlas that contains the numbers (labels) of the ROIs
    roi_names: string
        Name of column in the atlas that contains the ROIs names.
        These names will be presented on the circular graph

    Returns
    -------
    groups: Dictionary<string,List<(int, string)>
        Dictionary of groups of ROIs, divided by the grouping variable.
        The keys are the groups names. The values are lists of tuples, each tuple represents a ROI in the group.
        Each tuple contains the index of a ROI in the connectivity matrix (starting from zero) and the ROI name.
        for example:  {"Frontal lobe": [(0, precentral gyrus), (1, SFG), (2, MFG), (3, IFG)}

    """
    # check if grouping_name is in the grouped_by_hemisphere columns
    if isinstance(grouped_by_hemisphere, pd.Series):
        grouped_by_hemisphere = grouped_by_hemisphere.to_frame()

    if grouping_name not in grouped_by_hemisphere.columns:
        if "group_name" in grouped_by_hemisphere.columns:
            grouping_name = "group_name"
        else:
            raise KeyError(
                f"Neither '{grouping_name}' nor 'group_name' found in DataFrame columns: {grouped_by_hemisphere.columns.tolist()}"
            )

    grouped_atlas = grouped_by_hemisphere.groupby([grouping_name])
    groups_names = list(grouped_atlas.groups.keys())
    groups = {}
    for group in groups_names:
        group_df = grouped_atlas.get_group(group)
        # check if roi_names is in the group_df columns
        if roi_names not in group_df.columns:
            # use ROIname instead
            roi_names = "node_name"
        groups[group] = list(zip(group_df[label] - 1, group_df[roi_names]))
    return groups

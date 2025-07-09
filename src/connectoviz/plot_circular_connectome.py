# connectoviz/plotting/circular_plots.py

from typing import Optional, List, Union
import numpy as np
import pandas as pd

from connectoviz.core.connectome import Connectome
from connectoviz.visualization.circular_graph_legacy import visualize_connectome


def plot_circular_connectome(
    con_mat: np.ndarray,
    atlas: pd.DataFrame,
    metadata_df: pd.DataFrame,
    hemispheric_par: bool = False,
    group_by: Optional[str] = None,
    include_other: Optional[bool] = True,
    display_group_names: bool = False,
    display_node_names: bool = False,
    label: str = "Label",
    roi_names: str = "ROIname",
    tracks: Optional[List[str]] = None,
    index_mapping: Optional[Union[dict, pd.DataFrame]] = None,
    weights: Optional[np.ndarray] = None,
    edge_threshold: Optional[float] = None,
    # Styling kwargs
    group_cmap: str = "Pastel1",
    metadata_cmap: str = "pink",
    edge_cmap: str = "managua",
    gap: float = 2.0,
    figsize: tuple = (10, 10),
    start_angle: float = 0.0,
    node_size=10,
    edge_alpha=0.8,
    edge_scaling=3,
    save_path=None,
    show_graph=False,
    **kwargs,
):
    """
    Main API to generate a circular connectome plot.

    Parameters
    ----------
    con_mat : np.ndarray
        Square connectivity matrix (n x n).
    atlas : pd.DataFrame
        Atlas reference table.
    metadata_df : pd.DataFrame
        Node-level metadata (same length as con_mat).
    hemispheric_par : bool
        Whether to use symmetrical hemisphere-based layout.
    include_other : bool
        Whether to include nodes not grouped by the specified hemisphere.
    group_by : str, optional
        Metadata column to group nodes by.
    display_group_names : bool
        Whether to display group names in the plot.
    display_node_names : bool
        Whether to display node names in the plot.
    label : str
        Column name  for Labels(numbers) for the nodes ROIs.
    roi_names : str
        Column name in metadata_df for ROI names.

    tracks : list of str, optional
        Metadata columns to draw as concentric rings.
    index_mapping : dict or pd.DataFrame, optional
        Optional remapping of node indices to labels.
    weights : np.ndarray, optional
        Matrix of same shape as con_mat to apply as a mask.
    edge_threshold : float, optional
        Threshold for edge weights to visualize.
    cmap : str
        Colormap name for edges and tracks.
    gap : float
        Angular gap between groups.
    figsize : tuple
        Size of the figure.
    start_angle : float
        Starting angle in degrees.
    kwargs : dict
        Other styling overrides.
    """

    # Step 1: Construct the Connectome
    connectome = (
        Connectome.from_inputs(
            con_mat=con_mat,
            atlas=atlas,
            node_metadata=metadata_df,
            mapping=index_mapping,
        )
        if index_mapping is not None
        else Connectome.from_inputs(
            con_mat=con_mat,
            atlas=atlas,
            node_metadata=metadata_df,
            index_col=label,
            label_col=roi_names,
        )
    )  # i know its confusing- need to refactor this later and to change thew name of label_col to roi_names

    # Step 2: Apply weights/mask if provided
    if weights is not None:
        connectome.apply_mask(weights)

    # Step 3: Apply reordering and grouping if requested
    layout_dict = {
        "hemi": hemispheric_par,
        "other": include_other,
        "grouping": group_by,
        "node_name": roi_names,
        "display_node_name": display_node_names,
        "display_group_name": display_group_names,
    }
    connectome.reorder_nodes(layout_dict=layout_dict)

    # Step 4: Validate track and group_by fields

    if tracks:
        # choose the value from merged that the jey is L- if L isnt in merged_metadata, use All
        if "L" in connectome.merged_metadata.keys():
            missing = [
                t for t in tracks if t not in connectome.merged_metadata["L"].columns
            ]
        else:
            missing = [
                t for t in tracks if t not in connectome.merged_metadata["All"].columns
            ]
        # missing = [t for t in tracks if t not in connectome.merged_metadata.columns]
        if missing:
            raise ValueError(f"Tracks not found in metadata: {missing}")

    # filter and order the metadata DataFrame based on the tracks if provided
    if tracks:
        # Ensure tracks are in the metadata DataFrame
        print(f"Filtering metadata by tracks: {tracks}")
        connectome.apply_layers(tracks, label)

        #####only for now as we dont have time to implement the logic for multiple tracks

        track_by = tracks[0]
        # Step 5: Build and render the circular plot
        circ_graph = visualize_connectome(
            connectome=connectome,
            layout_dict=layout_dict,
            label=label,
            roi_names=roi_names,
            track_by=track_by,
            threshold=edge_threshold,
        )
    else:
        # If no tracks are specified, just visualize the connectome without tracks
        circ_graph = visualize_connectome(
            connectome=connectome,
            layout_dict=layout_dict,
            label=label,
            roi_names=roi_names,
            threshold=edge_threshold,
        )
    # step 6: show the graph(unless you want to customize it further)

    fig, ax = circ_graph.generate_graph(
        group_cmap="Pastel1",
        metadata_cmap="pink",
        edge_cmap="managua",
        node_size=10,
        edge_alpha=0.8,
        figsize=(8, 8),
        edge_scaling=3,
        save_path=None,
        show_graph=False,
    )
    return fig, ax

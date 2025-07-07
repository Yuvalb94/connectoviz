from pathlib import Path
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt

from connectoviz.visualization.circular_graph_legacy import (
    load_data,
    normalize_and_set_threshold,
    CircularGraph,
)


def plot_circular_connectome_alt(
    connectivity_matrix_path: Union[str, Path],
    atlas_path: Union[str, Path],
    grouping_name: str = "Lobe",
    label: str = "Label",
    roi_names: str = "ROIname",
    hemisphere: str = "Hemi",
    left_symbol: str = "L",
    right_symbol: str = "R",
    metadata: Optional[str] = None,
    display_node_names: bool = False,
    display_group_names: bool = False,
    threshold: float = 0.1,
    group_cmap: str = "tab20",
    metadata_cmap: str = "viridis",
    edge_cmap: str = "plasma",
    node_size: int = 10,
    edge_alpha: float = 0.8,
    figsize: Tuple[int, int] = (10, 10),
    edge_scaling: int = 3,
    save_path: Optional[Union[str, Path]] = None,
    show_graph: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Alternative plotting function using the CircularGraph class.

    Parameters
    ----------
    connectivity_matrix_path : str or Path
        Path to CSV connectivity matrix.
    atlas_path : str or Path
        Path to CSV atlas file with region info.
    grouping_name : str
        Column in atlas to group ROIs (e.g., lobe).
    label : str
        Column with node indices (starting at 1).
    roi_names : str
        Column with human-readable ROI names.
    hemisphere : str
        Column indicating hemisphere ('L', 'R').
    left_symbol : str
        Value for left hemisphere.
    right_symbol : str
        Value for right hemisphere.
    metadata : str, optional
        Column with scalar metadata (for ring overlay).
    display_node_names : bool
        Whether to display node names.
    display_group_names : bool
        Whether to display group names.
    threshold : float
        Threshold below which connections are discarded.
    group_cmap : str
        Colormap for group ring.
    metadata_cmap : str
        Colormap for metadata ring.
    edge_cmap : str
        Colormap for edges.
    node_size : int
        Size of node dots.
    edge_alpha : float
        Transparency of edges.
    figsize : tuple
        Size of the figure.
    edge_scaling : int
        Multiplier for edge width.
    save_path : str or Path, optional
        Path to save output image.
    show_graph : bool
        Whether to display the graph.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    (
        conn,
        groups,
        metadata_map,
        metadata_label,
        row_names_map,
        disp_nodes,
        disp_groups,
    ) = load_data(
        connectivity_matrix_path=connectivity_matrix_path,
        atlas_path=atlas_path,
        grouping_name=grouping_name,
        label=label,
        roi_names=roi_names,
        hemisphere=hemisphere,
        left_symbol=left_symbol,
        right_symbol=right_symbol,
        metadata=metadata,
        display_node_names=display_node_names,
        display_group_names=display_group_names,
    )

    filtered = normalize_and_set_threshold(conn, threshold=threshold)

    cg = CircularGraph(
        connectivity_matrix=filtered,
        groups=groups,
        metadata_map=metadata_map,
        metadata_label=metadata_label,
        row_names_map=row_names_map,
        display_node_names=disp_nodes,
        display_group_names=disp_groups,
    )

    fig, ax = cg.generate_graph(
        group_cmap=group_cmap,
        metadata_cmap=metadata_cmap,
        edge_cmap=edge_cmap,
        node_size=node_size,
        edge_alpha=edge_alpha,
        figsize=figsize,
        edge_scaling=edge_scaling,
        save_path=save_path,
        show_graph=False,
    )

    return fig, ax

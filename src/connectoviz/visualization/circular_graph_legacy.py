import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from matplotlib.path import Path as MplPath
import matplotlib.patches as patches
from connectoviz.core.connectome import Connectome
from typing import Dict, Any, Optional, Tuple
from connectoviz.utils.handle_layout_prefrences import create_dictionary


def normalize_and_set_threshold(
    connectivity_matrix: np.ndarray, threshold: Optional[float] = 0.1
):
    """
    This function gets a connectivity matrix and normalize its values between 0 to 1.
    After normalization, the function zero the matrix values that are lower than  the threshold

    Parameters
    ----------
    connectivity_matrix: np.ndarray
        n X n matrix where n is the number of ROIs in  the atlas.
        Each cell in the matrix describes the connection between each pair of ROIs.

    threshold: float
        A float between 0 to 1 by. Values lower than threshold are set to zero.

    Returns
    -------
    filtered_matrix: np.ndarray
        connecitivty matrix after thresholding and normalization
    """
    if threshold is None:
        threshold = 0.1
    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold value must be between 0-1!")

    filtered_matrix = (connectivity_matrix - np.min(connectivity_matrix)) / (
        np.max(connectivity_matrix) - np.min(connectivity_matrix)
    )

    filtered_matrix[filtered_matrix < threshold] = 0

    return filtered_matrix


class CircularGraph:
    def __init__(
        self,
        connectivity_matrix: np.ndarray,
        groups: list,
        metadata_map: dict,
        metadata_label: str | None,
        row_names_map: dict,
        display_node_names: bool,
        display_group_names: bool,
    ):
        """
        Main plotting function

        Args:
            connectivity_matrix (np.ndarray):
                Weighted connectivity matrix
            groups (list):
                List of dicts, one for each hemi, of a format Dictionary<string,List<(int, string)>.
                Dictionary of groups of ROIs, divided by the grouping variable.
                The keys are the groups names. The values are lists of tuples, each tuple represents a ROI in the group.
                Each tuple contains the index of a ROI in the connectivity matrix (starting from zero) and the ROI name.
                for example:  {"Frontal lobe": [(0, precentral gyrus), (1, SFG), (2, MFG), (3, IFG)}
                Used to compute the layout
            metadata_map (dict):
                Dict mapping node_label to metadata value. Used to color code metadata ring
            metadata_label (str or None):
                Name of metadata parameter. Used to handle metadata display
            row_names_map (dict):
                Dict mapping node_label to node name value. Used to display node names
            display_node_names (bool):
                Flag for node labels display mode
            display_group_names (bool):
                Flag for group labels display mode
        """
        self.matrix = connectivity_matrix
        self.groups = groups
        self.metadata_map = metadata_map
        self.metadata_label = metadata_label
        self.row_names_map = row_names_map
        self.disp_nodes = display_node_names
        self.disp_groups = display_group_names

    def _compute_positions(
        self,
        small_gap_arc: float = 0.05,  # radians between groups
        large_gap_arc: float = 0.3,  # radians to leave clear at top
    ):
        """
        Compute positions so that:
          - A large gap of `large_gap_arc` sits centered at 90° (π/2).
          - If there are any 'else' nodes, carve out `else_arc = small_gap_arc` at 270°,
            and shrink each hemi by half of that.
          - Within each hemi, groups get arcs proportional to their node counts,
            separated by fixed `small_gap_arc`.
        """
        left_dict, right_dict, else_dict = (
            self.groups
            if len(self.groups) == 3
            else (self.groups[0], self.groups[1], {})
        )
        # group_names = list(left_dict.keys()) joseph added
        group_names = sorted(set(left_dict.keys()).union(right_dict.keys()))
        # Ensure all groups are represented in the order of left, right if there isnt symmetry
        H = len(group_names)

        sg = small_gap_arc
        lg = large_gap_arc

        # 1) see if we have any bottom‐(else) nodes
        all_else = [idx for grp in else_dict for idx, _ in else_dict[grp]]
        n_else = len(all_else)
        if n_else:
            left_counts = [len(v) for v in left_dict.values()]
            right_counts = [len(v) for v in right_dict.values()]
            else_counts = [len(v) for v in else_dict.values()]

            total_left = sum(left_counts)
            total_right = sum(right_counts)
            total_else = sum(else_counts)

            # count interior small gaps
            gaps_left = max(len(left_counts) - 1, 0)
            gaps_right = max(len(right_counts) - 1, 0)
            gaps_else = max(len(else_counts) - 1, 0)

            # total nodes & total small‐gap length
            total_nodes = total_left + total_right + total_else
            total_small_gaps = small_gap_arc * (gaps_left + gaps_right + gaps_else)

            # 1) compute per-node spacing so that:
            #    2π = large_top_gap + total_small_gaps + per_node_arc * total_nodes
            per_node_arc = (
                2 * math.pi - large_gap_arc - total_small_gaps
            ) / total_nodes

            # 2) turn counts into group‐arcs
            left_arcs = [per_node_arc * c for c in left_counts]
            right_arcs = [per_node_arc * c for c in right_counts]
            else_arcs = [per_node_arc * c for c in else_counts]

            else_arc = sum(else_arcs) + small_gap_arc * gaps_else
        else:
            else_arc = 0.0

        # carve out top gap (2 * lg) and bottom gap (2 * lg + else_arc), split remaining half/half
        hemi_arc = math.pi - lg - (else_arc / 2)

        # 3) compute how much of hemi_arc each group gets
        left_counts = [len(left_dict.get(grp, [])) for grp in group_names]
        right_counts = [len(right_dict.get(grp, [])) for grp in group_names]
        total_left = sum(left_counts) or 1
        total_right = sum(right_counts) or 1

        avail_arc = hemi_arc - (H - 1) * sg
        left_arcs = [avail_arc * (c / total_left) for c in left_counts]
        right_arcs = [avail_arc * (c / total_right) for c in right_counts]

        # 4) starting angles for each hemi
        left_start = math.pi / 2 + lg / 2
        right_start = math.pi / 2 - lg / 2

        angles = {}

        # LEFT hemi: counterclockwise
        theta = left_start
        for arc, grp, cnt in zip(left_arcs, group_names, left_counts):
            items = left_dict.get(grp, [])
            if cnt:
                for j, (idx, _) in enumerate(items):
                    frac = (j + 0.5) / cnt
                    angles[idx] = theta + frac * arc
            theta += arc + sg

        # RIGHT hemi: clockwise
        theta = right_start
        for arc, grp, cnt in zip(right_arcs, group_names, right_counts):
            items = right_dict.get(grp, [])
            if cnt:
                for j, (idx, _) in enumerate(items):
                    frac = (j + 0.5) / cnt
                    angles[idx] = theta - frac * arc
            theta -= arc + sg

        # 5) ELSE group at bottom, spanning else_arc
        if n_else:
            for j, idx in enumerate(all_else):
                frac = (j + 0.5) / n_else
                angles[idx] = 3 * math.pi / 2 + (frac - 0.5) * else_arc

        # 6) build your position dicts
        base_pos = {n: (math.cos(a), math.sin(a)) for n, a in angles.items()}
        inner_pos = base_pos.copy()
        outer_pos = {n: (1.1 * x, 1.1 * y) for n, (x, y) in base_pos.items()}
        labels_pos = {n: (1.05 * x, 1.05 * y) for n, (x, y) in base_pos.items()}

        return base_pos, inner_pos, outer_pos, labels_pos, angles

    def _create_graph(self, edge_scaling=3) -> Tuple[plt.Figure, plt.Axes]:
        """
        Creates a directed NetworkX graph from the filtered matrix,
        and sets edge weights, metadata, and group attributes.
        """
        g = nx.from_numpy_array(self.matrix).to_directed()
        nx.set_edge_attributes(
            g,
            {
                e: w * edge_scaling
                for e, w in nx.get_edge_attributes(g, "weight").items()
            },
            "doubled_weight",
        )
        nx.set_node_attributes(g, self.metadata_map, "metadata")

        node_group_map = {}
        for hemi_dict in self.groups:
            for grp_label, items in hemi_dict.items():
                for idx, _ in items:
                    node_group_map[idx] = grp_label
        nx.set_node_attributes(g, node_group_map, "group")

        return g

    def _get_node_colors_and_labels(self, g):
        """
        Returns group colormap indices, metadata values (or None), and node label dict.
        """
        # Group mapping → integer colormap values
        grp_vals = [g.nodes[n]["group"] for n in g.nodes()]
        unique_grps = list(dict.fromkeys(grp_vals))
        grp_to_int = {grp: i for i, grp in enumerate(unique_grps)}
        grp_nums = [grp_to_int[grp] for grp in grp_vals]

        # Metadata values if available
        meta_vals = None
        if self.metadata_label is not None:
            meta_vals = [float(g.nodes[n]["metadata"]) for n in g.nodes()]

        # Labels
        labels = self.row_names_map if self.disp_nodes else {}
        return grp_nums, meta_vals, labels

    def _draw_nodes(
        self,
        ax,
        g,
        inner_pos,
        outer_pos,
        labels_pos,
        node_colors_group,
        node_colors_meta,
        node_labels,
        node_size=10,
    ):
        """
        Draws group-colored nodes (inner ring), metadata-colored nodes (outer ring),
        and node labels. Pure draw function — inputs must be precomputed.
        """

        # --- Draw metadata ring (outer layer) ---
        if node_colors_meta is not None:
            nc = nx.draw_networkx_nodes(
                g,
                pos=outer_pos,
                node_color=node_colors_meta,
                cmap=self._metadata_cmap,
                node_size=node_size,
                ax=ax,
            )
            self._meta_colorbar_node = nc  # store for later use in _draw_legends

        # --- Draw group-colored ring (inner layer) ---
        nx.draw_networkx_nodes(
            g,
            pos=inner_pos,
            node_color=node_colors_group,
            cmap=self._group_cmap,
            node_size=node_size,
            ax=ax,
        )

        # --- Optional node labels ---
        if self.disp_nodes and node_labels:
            nx.draw_networkx_labels(
                g, pos=labels_pos, labels=node_labels, font_size=2.5, ax=ax
            )

    def _draw_edges(self, ax, g, inner_pos, angles, edge_alpha=0.8):
        """
        Draws curved edges between nodes, using Bézier curves to the center.
        The edges are colored by weight, and their width is doubled for visibility.
        """
        cmap = self._edge_cmap
        edge_attrs = nx.get_edge_attributes(g, "weight")
        norm = plt.Normalize(
            vmin=min(edge_attrs.values()), vmax=max(edge_attrs.values())
        )

        for u, v, attr in g.edges(data=True):
            w = attr["weight"]
            ww = attr["doubled_weight"]
            color = cmap(norm(w))

            x1, y1 = inner_pos[u]
            x2, y2 = inner_pos[v]
            verts = [(x1, y1), (0, 0), (x2, y2)]
            codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
            path = MplPath(verts, codes)

            patch = patches.PathPatch(
                path, edgecolor=color, linewidth=ww, alpha=0.8, facecolor="none"
            )
            ax.add_patch(patch)

        # Store colormap and normalization for later use in legend
        self._edge_cmap = cmap
        self._edge_norm = norm

    def _draw_group_labels(self, ax, angles):
        """
        Places group labels (e.g., lobes) outside the node ring,
        aligned by hemisphere (left/right/else).
        """
        if not self.disp_groups:
            return

        for hemi_index, hemi_dict in enumerate(self.groups):  # left, right, else
            for group_name, items in hemi_dict.items():
                indices = [idx for idx, _ in items]
                group_angles = [angles[idx] for idx in indices]
                if not group_angles:
                    continue

                # Average position (circular mean)
                mean_sin = np.mean([math.sin(a) for a in group_angles])
                mean_cos = np.mean([math.cos(a) for a in group_angles])
                theta = math.atan2(mean_sin, mean_cos)
                x, y = 1.5 * math.cos(theta), 1.5 * math.sin(theta)

                # Alignment based on hemisphere
                ha = "center"
                if hemi_index == 0:
                    ha = "left"  # left hemisphere
                elif hemi_index == 1:
                    ha = "right"  # right hemisphere

                ax.text(x, y, group_name, ha=ha, va="center", fontsize=8)

    def _draw_legends(self, fig, ax, g):
        """
        Draws legends for metadata and edge weights.
        """
        # Metadata colorbar
        if hasattr(self, "_meta_colorbar_node"):
            fig.colorbar(
                self._meta_colorbar_node,
                ax=ax,
                location="right",
                fraction=0.046,
                pad=0.04,
                label=self.metadata_label,
            )

        # Edge weight colorbar
        if hasattr(self, "_edge_cmap") and hasattr(self, "_edge_norm"):
            sm = plt.cm.ScalarMappable(cmap=self._edge_cmap, norm=self._edge_norm)
            sm.set_array([])
            fig.colorbar(
                sm,
                ax=ax,
                location="bottom",
                fraction=0.046,
                pad=0.04,
                label="Edge weight",
            )

    @staticmethod
    def _resolve_cmap(cmap, default_name):
        if cmap is None:
            return plt.get_cmap(default_name)
        if isinstance(cmap, str):
            return plt.get_cmap(cmap)
        return cmap  # assume it's a valid Colormap object

    def generate_graph(
        self,
        group_cmap=None,
        metadata_cmap=None,
        edge_cmap=None,
        node_size=10,
        edge_alpha=0.8,
        figsize=(8, 8),
        edge_scaling=3,
        save_path=None,
        show_graph=False,
    ):

        # 1. Layout
        base_pos, inner_pos, outer_pos, labels_pos, angles = self._compute_positions()

        # 2. Graph & attributes
        g = self._create_graph(edge_scaling=edge_scaling)

        # Set colormaps
        self._group_cmap = self._resolve_cmap(group_cmap, "tab20")
        self._metadata_cmap = self._resolve_cmap(metadata_cmap, "viridis")
        self._edge_cmap = self._resolve_cmap(edge_cmap, "plasma")

        # 3. Color and label inputs
        grp_nums, meta_vals, labels = self._get_node_colors_and_labels(g)

        # 4. Plot setup
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.axis("off")

        # 5. Draw everything
        self._draw_nodes(
            ax,
            g,
            inner_pos,
            outer_pos,
            labels_pos,
            grp_nums,
            meta_vals,
            labels,
            node_size,
        )

        self._draw_edges(ax, g, inner_pos, angles, edge_alpha)
        self._draw_group_labels(ax, angles)
        self._draw_legends(fig, ax, g)

        # 6. Save or display
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        if show_graph:
            plt.show()

        return fig, ax


def calcu_missing_pos_nodes_amount(
    connectivity_matrix: np.ndarray, groups: list
) -> tuple[list[tuple[str, int]], list[int]]:
    """
    Calculate the missing nodes in the connectivity matrix based on the row names map.
    Parameters
    ----------
    connectivity_matrix: np.ndarray
        n X n matrix where n is the number of ROIs in  the atlas.
        Each cell in the matrix describes the connection between each pair of ROIs.
    row_names_map: dict
        A dictionary mapping node labels to node names.
    Returns
    -------
    tuple that contains:
    list[tuple[str,int]]
         A list of tuples where each tuple contains the group name and the number of
           nodes in that group.
        includes all nodes in the connectivity matrix, all those that are grouped and those that are not grouped.
    list
        A list of ungrouped nodes in the connectivity matrix.
    """
    # All nodes in the matrix
    all_nodes = set(range(connectivity_matrix.shape[0]))

    # Nodes that are in any group
    grouped_nodes = set()
    for hemi in groups:
        for grp_items in hemi.values():
            grouped_nodes.update(idx for idx, _ in grp_items)

    # Ungrouped nodes
    ungrouped_nodes = all_nodes - grouped_nodes

    print(f"Total nodes: {len(all_nodes)}")
    print(f"Grouped nodes: {len(grouped_nodes)}")
    print(f"Ungrouped nodes: {len(ungrouped_nodes)} → {sorted(ungrouped_nodes)}")
    # create a tuple for total, grouped and ungrouped nodes
    all_nodes_tuple = ("Total nodes", len(all_nodes))
    grouped_nodes_tuple = ("Grouped nodes", len(grouped_nodes))
    ungrouped_nodes_tuple = ("Ungrouped nodes", len(ungrouped_nodes))
    return [
        all_nodes_tuple,
        grouped_nodes_tuple,
        ungrouped_nodes_tuple,
    ], list(ungrouped_nodes)


# Generate functions to handle ungrouped nodes in the connectome visualization
# These functions provide two options for dealing with ungrouped nodes:
# 1. Assign them to a group named 'Ungrouped' in each hemisphere.
# 2. Filter them out of the connectivity matrix entirely.
# --- Option A: Assign ungrouped nodes to 'Ungrouped' ---
def assign_ungrouped_to_group(groups, ungrouped_nodes):
    """
    Adds all ungrouped nodes into a group named 'Ungrouped' in each hemisphere.
    """
    for hemi_dict in groups:
        if ungrouped_nodes:
            hemi_dict["Ungrouped"] = [
                (idx, f"Node_{idx}") for idx in sorted(ungrouped_nodes)
            ]
    return groups


# --- Option B: Filter ungrouped nodes out of graph ---
def filter_ungrouped_from_graph(connectome, ungrouped_nodes):
    """
    Filters ungrouped nodes out of the connectivity matrix.
    """
    filtered_matrix = connectome.con_mat.copy()
    for idx in sorted(ungrouped_nodes):
        filtered_matrix[idx, :] = 0
        filtered_matrix[:, idx] = 0
    return filtered_matrix


# Generate function that visualizes a connectome using the CircularGraph class
def visualize_connectome(
    connectome: Connectome,
    layout_dict: Dict[str, Any],
    label: str = "Label",
    roi_names: str = "ROIname",
    track_by: Optional[str] = None,
    threshold: Optional[float] = 0.0,
) -> CircularGraph:
    """
    Visualize a connectome using a circular graph layout.
    Parameters
    ----------
    connectome: Connectome
        A Connectome object containing the connectivity matrix ,atlas and more.
    layout_dict: Dict[str, Any]
        A dictionary containing layout preferences, including:
          - 'hemi': bool, whether to reorder by hemisphere.
          - 'other': bool, whether to include nodes not grouped by the specified hemisphere.
          - 'grouping': str, metadata column to group nodes by.
          - 'node_name': str, column name for node names in the metadata.
          - 'display_node_name': bool, whether to display node names.
          - 'display_group_name': bool, whether to display group names.
    label: str
        The column name in the atlas that contains the labels(numbers) for the ROIs.
    roi_names: str
        The column name in the atlas that contains the names of the ROIs.
    track_by: str
        The column name in the atlas or metadata that contains the metadata to track by (e.g., Yeo_7network).
    threshold: float
        A float between 0 and 1 to threshold the connectivity matrix after normalization.
        Values lower than this threshold are set to zero.
    Returns
    -------
    circular_graph
        An instance of the circular_graph class containing the visualized connectome.

    """
    if connectome.merged_metadata is None:
        raise ValueError("merged_metadata is None. Cannot continue.")

    # now taking the merged metadata and divide to different hemi DataFrames
    hemis_dfs = []
    for hemi in connectome.merged_metadata.keys():
        # get the merged metadata_dict and filter by  key 'hemi'
        hemi_df = connectome.merged_metadata[hemi]
        hemis_dfs.append(hemi_df)
    # use create-dictionary to create groups
    groups = []
    for hemi_df in hemis_dfs:
        groups.append(
            create_dictionary(hemi_df, layout_dict["grouping"], label, roi_names)
        )
    lis_len_nodes, ungrouped_nodes = calcu_missing_pos_nodes_amount(
        connectome.con_mat, groups
    )
    # Option A: Assign ungrouped to 'Ungrouped'
    groups = assign_ungrouped_to_group(groups, ungrouped_nodes)
    conn = connectome.con_mat
    # Option B: Filter ungrouped nodes completely
    # conn = filter_ungrouped_from_graph(connectome, grouped_nodes)
    # get the connectivity matrix and normalize it

    filtered = normalize_and_set_threshold(conn, threshold=threshold)
    # get the metadata_map and metadata_label
    if (
        connectome.merged_metadata is None
        or track_by is None
        or not any(track_by in df.columns for df in connectome.merged_metadata.values())
    ):
        metadata_map = dict(zip(connectome.atlas[label] - 1, connectome.atlas[label]))
        metadata_label = None
    else:
        # if track_by is in the atlas, use it, otherwise use the node_metadata
        if track_by in connectome.atlas.columns:
            metadata_track = track_by
            # if values in connectome.atlas[metadata_track] are int or float continue
            if connectome.atlas[metadata_track].dtype in [np.int64, np.float64]:
                metadata_map = dict(
                    zip(connectome.atlas[label] - 1, connectome.atlas[metadata_track])
                )
            else:
                # take all unique values and map them to integers
                unique_values = connectome.atlas[metadata_track].unique()
                value_to_int = {v: i for i, v in enumerate(unique_values)}
                metadata_map = dict(
                    zip(
                        connectome.atlas[label] - 1,
                        connectome.atlas[metadata_track].map(value_to_int),
                    )
                )
            # metadata_map = dict(
            #     zip(connectome.atlas[label] - 1, connectome.atlas[metadata_track])
            # )
            metadata_label = metadata_track
        elif (
            connectome.node_metadata is not None
            and track_by in connectome.node_metadata.columns
        ):
            metadata_track = track_by
            if connectome.node_metadata[metadata_track].dtype in [np.int64, np.float64]:
                metadata_map = dict(
                    zip(
                        connectome.atlas[label] - 1,
                        connectome.node_metadata[metadata_track],
                    )
                )
            else:
                # take all unique values and map them to integers
                unique_values = connectome.node_metadata[metadata_track].unique()
                value_to_int = {v: i for i, v in enumerate(unique_values)}
                # map the values to integers
                metadata_map = dict(
                    zip(
                        connectome.atlas[label] - 1,
                        connectome.node_metadata[metadata_track].map(value_to_int),
                    )
                )

            metadata_label = metadata_track
        else:
            raise ValueError(
                f"Metadata '{track_by}' not found in atlas or node metadata."
            )

    row_names_map = dict(zip(connectome.atlas[label] - 1, connectome.atlas[roi_names]))
    cg = CircularGraph(
        connectivity_matrix=filtered,
        groups=groups,
        metadata_map=metadata_map,
        metadata_label=metadata_label,
        row_names_map=row_names_map,
        display_node_names=layout_dict["display_node_name"],
        display_group_names=layout_dict["display_group_name"],
    )
    # cg.show_graph()
    return cg

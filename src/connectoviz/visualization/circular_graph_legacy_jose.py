import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from matplotlib.path import Path as MplPath
import matplotlib.patches as patches
from connectoviz.core.connectome import Connectome
from typing import Dict, Any, Optional
from connectoviz.utils.handle_layout_prefrences import create_dictionary


# def load_data(
#     connectivity_matrix_path,
#     atlas_path,
#     grouping_name="Lobe",
#     label="Label",
#     roi_names="ROIname",
#     hemisphere="Hemi",
#     left_symbol="L",
#     right_symbol="R",
#     metadata=None,
#     display_node_names: bool = False,
#     display_group_names: bool = False,
# ):
#     """
#     Now returns:
#       connectivity_matrix, groups, metadata_map,
#       row_names_map, display_node_names, display_group_names

#     Modified to allow atlases that contain ONLY left and right (no 'else'),
#     while still supporting a third 'else' hemisphere if present.
#     """
#     conn = pd.read_csv(connectivity_matrix_path, header=None).values
#     atlas = pd.read_csv(atlas_path)

#     # basic shape & column checks
#     n, m = conn.shape
#     num_rois = atlas[label].max()
#     if n != m or n != num_rois:
#         raise ValueError("Connectivity matrix size must match atlas labels.")

#     for col in (grouping_name, label, roi_names, hemisphere, metadata):
#         if col not in atlas.columns:
#             raise ValueError(f"Atlas missing required column '{col}'")


#     # optional metadata
#     if metadata is not None and metadata not in atlas.columns:
#         raise ValueError(f"Atlas missing required column '{metadata}'")

#     # build metadata and name maps (or empty if none)
#     if metadata is None:
#         metadata_map = {}
#         metadata_label = None
#     else:
#         metadata_map   = dict(zip(atlas[label] - 1, atlas[metadata]))
#         metadata_label = metadata

#     row_names_map = dict(zip(atlas[label] - 1, atlas[roi_names]))

#     # enforce L/R/else
#     atlas = atlas.copy()
#     atlas[hemisphere] = atlas[hemisphere].apply(
#         lambda x: x if x in (left_symbol, right_symbol) else "else"
#     )
#     groups_hemi = atlas.groupby(hemisphere)

#     # helper: get group‐DataFrame or empty
#     def _get(side):
#         return (
#             groups_hemi.get_group(side)
#             if side in groups_hemi.groups
#             else atlas.iloc[0:0]
#         )

#     # build the three hemisphere‐based dictionaries
#     left_df = _get(left_symbol)
#     right_df = _get(right_symbol)
#     else_df = _get("else")

#     groups = [
#         create_dictionary(left_df, grouping_name, label, roi_names),
#         create_dictionary(right_df, grouping_name, label, roi_names),
#         create_dictionary(else_df, grouping_name, label, roi_names),
#     ]

#     return (
#         conn,
#         groups,
#         metadata_map,

#         metadata_label,
#         row_names_map,
#         display_node_names,
#         display_group_names,
#         )


def normalize_and_set_threshold(connectivity_matrix, threshold=0.5):
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
    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold value must be between 0-1!")

    normalized_connectivity_matrix = (
        connectivity_matrix - np.min(connectivity_matrix)
    ) / (np.max(connectivity_matrix) - np.min(connectivity_matrix))

    filtered_matrix = normalized_connectivity_matrix
    filtered_matrix[normalized_connectivity_matrix < threshold] = 0

    return filtered_matrix


class circular_graph:
    def __init__(
        self,
        filtered_matrix: np.ndarray,
        groups,
        metadata_map,
        metadata_label,
        row_names_map,
        display_node_names: bool,
        display_group_names: bool,
    ):
        self.filtered = filtered_matrix
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
        left_dict, right_dict, else_dict = self.groups
        group_names = list(left_dict.keys())
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

            # how many interior small gaps?
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

        # 2) carve out top gap (lg) and bottom gap (else_arc), split remaining half/half
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

        # LEFT hemi: CCW
        theta = left_start
        for arc, grp, cnt in zip(left_arcs, group_names, left_counts):
            items = left_dict.get(grp, [])
            if cnt:
                for j, (idx, _) in enumerate(items):
                    frac = (j + 0.5) / cnt
                    angles[idx] = theta + frac * arc
            theta += arc + sg

        # RIGHT hemi: CW
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

    def show_graph(self):
        # --- build graph & attrs (unchanged) ---
        g = nx.from_numpy_array(self.filtered).to_directed()
        nx.set_edge_attributes(
            g,
            {e: w * 3 for e, w in nx.get_edge_attributes(g, "weight").items()},
            "doubled_weight",
        )
        nx.set_node_attributes(g, self.metadata_map, "metadata")

        node_group_map = {}
        for hemi_dict in self.groups:
            for grp_label, items in hemi_dict.items():
                for idx, _ in items:
                    node_group_map[idx] = grp_label
        nx.set_node_attributes(g, node_group_map, "group")

        # --- build symmetric L/R sequence with gaps ---
        base_pos, inner_pos, outer_pos, labels_pos, angles = self._compute_positions()

        # --- prepare color data (unchanged) ---
        meta_vals = (
            [float(g.nodes[n]["metadata"]) for n in g.nodes()]
            if self.metadata_label
            else None
        )  # jposeph added line
        grp_vals = [g.nodes[n]["group"] for n in g.nodes()]
        unique_grp = list(dict.fromkeys(grp_vals))
        grp_to_int = {g: i for i, g in enumerate(unique_grp)}
        grp_nums = [grp_to_int[g] for g in grp_vals]

        # --- draw ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.axis("off")

        # --- optional metadata ring (outer) ---
        if self.metadata_label is not None:
            meta_vals = [float(g.nodes[n]["metadata"]) for n in g.nodes()]
            nc = nx.draw_networkx_nodes(
                g,
                pos=outer_pos,
                node_color=meta_vals,
                cmap=plt.get_cmap("viridis"),
                node_size=10,
                ax=ax,
            )

            # add the colorbar for metadata ring
            fig.colorbar(
                nc,
                ax=ax,
                location="right",
                fraction=0.046,
                pad=0.04,
                label=self.metadata_label,
            )

        # group ring (inner)
        nx.draw_networkx_nodes(
            g,
            pos=inner_pos,
            node_color=grp_nums,
            cmap=plt.get_cmap("tab20"),
            node_size=10,
            ax=ax,
        )

        # curved edges via Bézier into the center
        cmap = plt.get_cmap("plasma")
        edge_attrs = nx.get_edge_attributes(g, "weight")
        min_w, max_w = min(edge_attrs.values()), max(edge_attrs.values())
        norm = plt.Normalize(vmin=min_w, vmax=max_w)

        for u, v, attr in g.edges(data=True):
            w = attr["weight"]
            ww = attr["doubled_weight"]
            color = cmap(norm(w))

            x1, y1 = inner_pos[u]
            x2, y2 = inner_pos[v]
            # control point at the center (0,0):
            verts = [(x1, y1), (0, 0), (x2, y2)]
            # codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            # path = Path(verts, codes)
            codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
            path = MplPath(verts, codes)

            patch = patches.PathPatch(
                path, edgecolor=color, linewidth=ww, alpha=0.8, facecolor="none"
            )
            ax.add_patch(patch)

        # add the colorbar for egdes
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(
            sm, ax=ax, location="bottom", fraction=0.046, pad=0.04, label="Edge weight"
        )

        # add node labels
        if self.disp_nodes:
            nx.draw_networkx_labels(
                g, pos=labels_pos, labels=self.row_names_map, font_size=2.5, ax=ax
            )

        # --- group labels with hemisphere‐specific alignment ---
        if self.disp_groups:
            # self.groups = [left_dict, right_dict, else_dict]
            for side_idx, hemi_dict in enumerate(self.groups):
                for grp_label, items in hemi_dict.items():
                    # centroid angle
                    indices = [idx for idx, _ in items]
                    thetas = [angles[idx] for idx in indices]
                    mean_sin = sum(math.sin(t) for t in thetas) / len(thetas)
                    mean_cos = sum(math.cos(t) for t in thetas) / len(thetas)
                    mean_theta = math.atan2(mean_sin, mean_cos)
                    # position just outside the node‐ring

                    tx, ty = 1.5 * math.cos(mean_theta), 1.5 * math.sin(mean_theta)

                    # choose horizontal alignment per hemisphere
                    if side_idx == 0:
                        ha = "left"
                    elif side_idx == 1:
                        ha = "right"
                    else:
                        ha = "center"
                    ax.text(tx, ty, grp_label, ha=ha, va="center", fontsize=8)

        plt.show()


# ---------------------------- usage ----------------------------

# conn, groups, metadata_map, metadata_label, row_names_map, disp_nodes, disp_groups = load_data(
#  "/Users/elijah/Desktop/courses/py_for_ns/connectogram_draft/conn_274.csv",
# "/Users/elijah/Desktop/courses/py_for_ns/connectogram_draft/mapping.csv",


# conn, groups, metadata_map, metadata_label, row_names_map, disp_nodes, disp_groups = load_data(
#     matrix_path,
#     atlas_path,
#     grouping_name="Lobe",
#     label="Label",
#     roi_names="ROIname",
#     hemisphere="Hemi",
#     metadata="Yeo_7network",
#     display_node_names=False,
#     display_group_names=True,
# )
# print('Groups')
# print(groups)
# print('Matadata dict')
# print(metadata_map)


# filtered = normalize_and_set_threshold(conn, threshold=0.1)
# bna = circular_graph(
#     filtered, groups, metadata_map, metadata_label, row_names_map, display_node_names=disp_nodes, display_group_names=disp_groups
# )
# bna.show_graph()


# do it but inside a funcf:
def visualize_connectome(
    connectome: Connectome,
    layout_dict: Dict[str, Any],
    label: str = "Label",
    roi_names: str = "ROIname",
    track_by: Optional[str] = None,
) -> circular_graph:
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
    # get the connectivity matrix and normalize it
    conn = connectome.con_mat
    filtered = normalize_and_set_threshold(conn, threshold=0.1)
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
    bna = circular_graph(
        filtered,
        groups,
        metadata_map,
        metadata_label,
        row_names_map,
        display_node_names=layout_dict["display_node_name"],
        display_group_names=layout_dict["display_group_name"],
    )
    # bna.show_graph()
    return bna


# # example uasge witg Connectome class:
# atlas_pd = pd.read_csv(atlas_path)
# con_mat = pd.read_csv(matrix_path, header=None).values
# connectome = Connectome.from_inputs(
#     con_mat=con_mat, atlas=atlas_pd, node_metadata=None, mapping=None
# )
# # print(connectome.atlas)
# layout_dict = {
#     "hemi": True,
#     "other": True,
#     "grouping": "Lobe",
#     "node_name": "ROIname",
#     "display_node_name": False,
#     "display_group_name": True,
# }
# # set vars for frouping_name, label, roi_names, hemisphere, metadata
# # grouping_name = "Lobe"
# # label = "Label"
# # roi_names = "ROIname"
# # hemisphere = "Hemi"
# # metadata = "Yeo_7network"  # or None if no metadata

# connectome.reorder_nodes(layout_dict)

# bnas = visualize_connectome(
#     connectome, layout_dict, label="Label", roi_names="ROIname", track_by="Yeo_7network"
# )
# bnas.show_graph()


# #now take the merged metadata and divide to different hemi DataFrames
# hemis_dfs =[]
# for hemi in connectome.merged_metadata.keys():
#     #get the merged metadata_dict and filter by  key 'hemi'
#     hemi_df = connectome.merged_metadata[hemi]
#     hemis_dfs.append(hemi_df)
# #use create-dictionary to create groups
# groups = []
# for hemi_df in hemis_dfs:
#     groups.append(
#         create_dictionary(hemi_df, grouping_name, label, roi_names)
#     )
# filtered = normalize_and_set_threshold(connectome.con_mat, threshold=0.1)
# metadata_map   = dict(zip(connectome.atlas[label] - 1, connectome.atlas[metadata]))
# metadata_label = metadata

# row_names_map = dict(zip(connectome.atlas[label] - 1, connectome.atlas[roi_names]))
# bna = circular_graph(
#      filtered, groups, metadata_map, metadata_label, row_names_map,
#      display_node_names=layout_dict["display_node_name"], display_group_names=layout_dict["display_group_name"]
#  )
# bna.show_graph()

# now use the groups to visualize the connectome

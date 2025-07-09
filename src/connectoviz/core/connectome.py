# all imports of parser and handle_layout_preferences should be edited back
# basd on original package structure  (ALSo change parser to parsers)

#
# connectoviz/core/connectome.py

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Any
from connectoviz.io.parsers import (
    parse_matrix,
    masking,
    check_metadata,
    check_mapping,
    compare_mapping,
    atlas_check,
    merge_metadata,
)
from connectoviz.utils.handle_layout_prefrences import (
    handle_layout,
    handle_layers,
)
import warnings

# next line commented out so pre commit wont kill me
# from connectoviz.utils.validation import validate_connectome_inputs


# 1.07.25- ideas for improvements:
# - add a method to convert the connectome to a networkx graph
# - use @property @setter and so on to encapsulate diffrent funcions and attributes as properties


class Connectome:
    """
    Internal data structure to store parsed connectome info.

    Parameters
    ----------
    con_mat : array-like
        Connectivity matrix (square 2D structure).
    node_metadata : dict or DataFrame
        Metadata table (e.g. lobe, hemisphere, etc.).
    atlas : str
        Atlas name (e.g. 'Schaefer100').
    """

    def __init__(
        self,
        con_mat: Union[np.ndarray, pd.DataFrame, List[List[float]], str],
        atlas: Optional[str] = None,
        node_metadata: Union[pd.DataFrame, Dict[str, List], str] = None,
        mapping: Optional[
            Union[Dict[str, str], pd.DataFrame]
        ] = None,  # verse 1 for mapping
        node_vec: Optional[Union[List[str], np.ndarray]] = None,
        label_vec: Optional[Union[List[str], np.ndarray]] = None,  # verse 2 for mapping
        # when no mapping is provided, we can use index_col and label_col to specify the columns in the metadata
        index_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ):
        self.con_mat: np.ndarray = parse_matrix(con_mat)
        self.atlas: pd.DataFrame = atlas
        self.mapping, self.index_col, self.label_col = (
            self._validate_maps(mapping, node_vec, label_vec, index_col, label_col)
            if mapping is not None
            else (None, index_col, label_col)
        )
        self.node_metadata: pd.DataFrame = self._process_metadata(node_metadata)

        self.merged_metadata: Optional[Dict[str, pd.DataFrame]] = (
            self._apply_merge()
        )  # merge the atlas with metadata-
        # in a dict format in case of grouping by hemi.

    def _process_metadata(self, metadata):
        # Your logic for checking and standardizing metadata
        if metadata is None:
            return None
        # Assuming `atlas` is already loaded and validated

        return check_metadata(metadata, atlas=self.atlas, mapping=self.mapping)

    def _apply_merge(self):
        """
        use the merge_metadata function to merge the atlas with metadata

        Returns
        -------
        a dict of DataFrames with the merged metadata.
        """
        if (
            self.node_metadata is None
            or (
                isinstance(self.merged_metadata, pd.DataFrame)
                and self.merged_metadata.empty
            )
            or (isinstance(self.merged_metadata, dict) and not self.merged_metadata)
        ):
            # raise a warning  and return the atlas
            warnings.warn(
                "Node metadata is not specified or empty. setting the atlas as the merged metadata."
            )

            return {"All": self.atlas}

        if self.atlas is None or self.atlas.empty:
            raise ValueError("Atlas is not specified or empty.")
        # merge the metadata with the atlas and get the number of nodes from con_mat
        if self.con_mat is None or self.con_mat.size == 0:
            raise ValueError("Connectivity matrix is not specified or empty.")
        # Check if the number of nodes in the metadata matches the connectivity matrix
        if self.node_metadata.shape[0] != self.con_mat.shape[0]:
            raise ValueError(
                "Node metadata does not match the number of nodes in the connectivity matrix."
            )
        combined_metadata = merge_metadata(self.atlas, self.node_metadata)
        merged_meta = {"All": combined_metadata}

        return merged_meta

    def _validate_maps(
        self,
        mapping: Optional[Union[Dict[str, str], pd.DataFrame]],
        node_vec: Optional[Union[List[str], np.ndarray]] = None,
        label_vec: Optional[Union[List[str], np.ndarray]] = None,
        index_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ):
        if mapping is not None:
            # Validate mapping structure
            validated_mapping = check_mapping(mapping, node_vec, label_vec)

            # Check mapping values against atlas
            _, label_col_r = compare_mapping(validated_mapping, self.atlas)
            # validate them and the atlas on the way:
            _, validated_index_col, validated_label_col = atlas_check(
                self.atlas, index_col, label_col_r
            )
            return mapping, validated_index_col, validated_label_col
        # if mapping is None:
        return None, None, None

    # create a method to create  connnectom named
    # add a function from_input to create a Connectome from inputs
    # functions assumes con_mat is a square 2D structure, atlas is a DataFrame, and node_metadata is a DataFrame
    # and mapping is an optional dict or DataFrame
    @classmethod
    def from_inputs(
        cls,
        con_mat: np.ndarray,
        atlas: pd.DataFrame,
        node_metadata: Optional[pd.DataFrame],
        mapping: Optional[Union[Dict[str, str], pd.DataFrame]] = None,
        index_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ):
        """
        Create a Connectome instance from input parameters.

        Parameters
        ----------
        con_mat : array-like
            Connectivity matrix (square 2D structure).
        atlas : pd.DataFrame
            Atlas reference table.
        node_metadata : pd.DataFrame
            Node-level metadata (same length as con_mat).
        mapping : dict or DataFrame, optional
            Optional remapping of node indices to labels.
        index_col : str, optional
            Column name in the atlas for node indices.
        label_col : str, optional
            Column name in the atlas for node labels.

        Returns
        -------
        Connectome
            A Connectome instance.
        """

        # check mapping

        # Validate and process the atlas
        if mapping is not None:
            atlas_bool, indx_col_r, label_col_r = atlas_check(
                atlas, index_col=None, label_col=None, mapping=mapping
            )

        else:
            atlas_bool, _, _ = atlas_check(atlas, index_col, label_col, mapping=None)

        if not atlas_bool:
            raise ValueError("Atlas is not valid or empty.")
        if mapping is not None:
            return Connectome(
                con_mat=con_mat,
                atlas=atlas,
                node_metadata=node_metadata,
                mapping=mapping,
                index_col=indx_col_r,
                label_col=label_col_r,
            )
        else:
            return Connectome(
                con_mat=con_mat,
                atlas=atlas,
                node_metadata=node_metadata,
                index_col=index_col,
                label_col=label_col,
            )

    def reorder_nodes(self, layout_dict: Dict[str, Any]):
        """
        Reorder the connectivity matrix by indicated input.
        Parameters
        ----------
        layout_dict : dict
            Dictionary containing layout preferences. It can include:
            - 'hemi': bool, whether to reorder by hemisphere.
            - 'other': bool, whether to include nodes not grouped by the specified hemisphere.
            - 'grouping': str, metadata column to group nodes by.
            - 'node_name': str, column name for node names in the metadata.
            - 'display_node_name': bool, whether to display node names.
            - 'display_group_name': bool, whether to display group names.
        Returns
        h



        """
        # sorting nodes based on the layout_dict by the nodal metadata or atlas
        if self.node_metadata is None or self.node_metadata.empty:
            atlas_bool, idx_col, label_col = atlas_check(
                self.atlas, self.index_col, self.label_col, self.mapping
            )
            if not atlas_bool:
                raise ValueError("Atlas is not valid or empty.")
            else:
                # add a coloumn named 'node_index' to the atlas based on the len of the con_mat
                if (
                    "node_index" not in self.atlas.columns
                    and self.con_mat.shape[0] == self.atlas.shape[0]
                ):
                    self.atlas["node_index"] = np.arange(1, self.con_mat.shape[0] + 1)
                # also if grouping inr nonw and is name of a column in the atlas, change the name of that coloumn to node_name
                if (
                    layout_dict["grouping"] is not None
                    and layout_dict["grouping"] in self.atlas.columns
                ):
                    self.atlas["group_name"] = self.atlas[layout_dict["grouping"]]

                    # self.atlas.rename(columns={layout_dict["grouping"]: "group_name"}, inplace=True)
                self.index_col = idx_col
                self.label_col = label_col
                handled_nodes, _, _, _ = handle_layout(self.atlas, layout_dict)
        else:
            handled_nodes, _, _, _ = handle_layout(self.node_metadata, layout_dict)
        # set the handeled nodes as the merged metadata
        self.merged_metadata = handled_nodes

    def apply_layers(self, layers_list: List[str], label: str = "Label"):
        """
        Apply layers to the connectivity matrix based on the provided layers list.
        for know- just filtering the metadata based on it
        Parameters
        ----------
        layers_list : list
            List of metadata columns to apply as layers.
        label : str
            The label column with numbers of ROIs in the metadata DataFrame. by default is "Label".
        Returns
        -------
        """
        if self.merged_metadata is None:
            raise ValueError("Merged metadata is not specified or empty.")
        # check if the layers_list is valid
        gilterd_metadata = handle_layers(self.merged_metadata, layers_list, label)
        if gilterd_metadata is None:
            raise ValueError("Filtered metadata is not specified or empty.")

        self.merged_metadata = gilterd_metadata

    def to_numpy(self):
        return self.con_mat.copy()

    def to_dataframe(self):
        # Convert the connectivity matrix to a DataFrame with node metadata as index and columns.
        if self.node_metadata is None or self.node_metadata.empty:
            raise ValueError("Node metadata is not specified or empty.")
        return pd.DataFrame(
            self.con_mat,
            index=self.node_metadata.index,
            columns=self.node_metadata.index,
        )

    def apply_mask(self, mask: Union[np.ndarray, List[bool]]):
        """
        Apply a binary mask to the connectivity matrix.

        Parameters
        ----------
        mask : array-like
            Binary mask to apply to the connectivity matrix.
        """
        self.con_mat = masking(mask, self.con_mat)

    def summary(self):
        return (
            f"Atlas: {self.atlas or 'unspecified'}\n"
            f"Nodes: {self.con_mat.shape[0]}\n"
            f"Metadata: {list(self.node_metadata.columns or "none specified")}\n"
            f"\nMapping: {self.mapping or 'none specified'}\n"
            f"Index Column: {self.index_col or 'none specified'}\n"
            f"Label Column: {self.label_col or 'none specified'}"
        )

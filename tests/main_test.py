import pandas as pd
from pathlib import Path
from connectoviz.plot_circular_connectome import plot_circular_connectome
import matplotlib.pyplot as plt

# Path to the current script
SCRIPT_DIR = Path(__file__).resolve().parent


# Path to the data directory (going up one level from visualization/)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # src/connectoviz

DATA_DIR = PACKAGE_ROOT / "src" / "connectoviz" / "data"
ATLAS_DIR = DATA_DIR / "atlases" / "available_atlases"
MAT_DIR = DATA_DIR / "connectomes"

# test fan2016 atlas and matrix
atlas_fname = r"fan2016/MNI152/space-MNI152_atlas-fan2016_res-1mm_dseg.csv"
matrix_fname = r"fan2016.csv"
# Now construct full paths
atlas_path = ATLAS_DIR / atlas_fname

matrix_path = MAT_DIR / matrix_fname


# atlas_fname = r"fan2016/MNI152/space-MNI152_atlas-fan2016_res-1mm_dseg.csv"
# matrix_fname = r"fan2016.csv"
# # Now construct full paths
# atlas_path = ATLAS_DIR / atlas_fname

# matrix_path = MAT_DIR / matrix_fname

# example uasge witg Connectome class:
atlas_pd = pd.read_csv(atlas_path)
con_mat = pd.read_csv(matrix_path, header=None).values

# print(connectome.atlas)
layout_dict = {
    "hemi": True,
    "other": True,
    "grouping": "Lobe",
    "node_name": "ROIname",
    "display_node_name": False,
    "display_group_name": True,
}
fig, ax = plot_circular_connectome(
    con_mat=con_mat,
    atlas=atlas_pd,
    metadata_df=None,
    hemispheric_par=True,
    include_other=True,
    group_by="Lobe",
    display_group_names=True,
    display_node_names=False,
    label="Label",
    roi_names="ROIname",
    tracks=["Yeo_7network"],
    show_graph=True,
)
plt.show()

# test huang2022
atlas_fname = r"huang2022/MNI152/space-MNI152_atlas-huang2022_res-1mm_dseg.csv"
matrix_fname = r"huang2022.csv"
# Now construct full paths
atlas_path = ATLAS_DIR / atlas_fname
matrix_path = MAT_DIR / matrix_fname
atlas_pd = pd.read_csv(atlas_path)
con_mat = pd.read_csv(matrix_path, header=None).values
plot_circular_connectome(
    con_mat=con_mat,
    atlas=atlas_pd,
    metadata_df=None,
    hemispheric_par=True,
    include_other=True,
    group_by="CortexDivision_name",
    display_group_names=True,
    display_node_names=False,
    label="HCPex_label",
    roi_names="RegionName1",
)

# # # test schaefer2018
atlas_fname = r"schaefer2018tian2020/MNI152/space-MNI152_atlas-schaefer2018tian2020_res-1mm_den-100_div-7networks_dseg.csv"
matrix_fname = r"schaefer2018tian2020_100_7.csv"
# Now construct full paths
atlas_path = ATLAS_DIR / atlas_fname
matrix_path = MAT_DIR / matrix_fname
atlas_pd = pd.read_csv(atlas_path)
con_mat = pd.read_csv(matrix_path, header=None).values
plot_circular_connectome(
    con_mat=con_mat,
    atlas=atlas_pd,
    metadata_df=None,
    hemispheric_par=True,
    include_other=False,  # set to false as there is no 'other' group in this atlas
    group_by="component",
    display_group_names=True,
    display_node_names=False,
    label="index",  # Assuming 'index' is the label column in the atlas
    roi_names="name",
    tracks=["network"],
)
plt.show()

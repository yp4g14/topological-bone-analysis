import numpy as np
import pandas as pd
from importlib import reload
from os import rmdir, remove, listdir
from os.path import exists, isfile

import persistent_homology_SEDT as pers_hom
reload(pers_hom)

# TEST - peristent_homology_sublevel_cubic
######################################################
test_image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.5, 0.4, 0, 0, 0],
    [0, 0, 0.4, 0.8, 1, 0.7, 0.6, 0.8, 0, 0],
    [0, 0, 0.3, 1, 0, 0, 0, 1, 0.6, 0],
    [0, 0.2, 1, 0, 0, 0, 0, 1, 1, 0],
    [0,0.1, 1, 0, 0, 0, 0.2, 0.9, 0, 0],
    [0, 0, 0.3, 0, 0, 0.4, 0.9, 0.7, 0, 0],
    [0, 0, 0.3, 0.5, 1, 0.6, 0.4, 0, 0, 0],
    [0, 0, 0, 0.15, 0.4, 0.8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

filename = "test.tif"
save_path = "C:/test_"
pers_hom.peristent_homology_sublevel_cubic(
    test_image,
    filename,
    save_path,
    plot_persistence_diagrams=True)

# check it creates necessary paths
assert exists(save_path+'idiagrams/'),\
    "idiagram path does not exist"
assert exists(save_path+'persistence_diagrams/'),\
    "persistence diagram path does not exist"
assert exists(save_path+'persistence_diagrams/plots/'),\
    "persistence diagram plot path does not exist"

# check it creates necessary files
assert isfile(save_path+'idiagrams/test.idiagram'),\
    "idiagram file does not exist"
assert isfile(save_path+'persistence_diagrams/PD_dim_0_test.csv'),\
    "persistence file dim 0 path does not exist"
assert isfile(save_path+'persistence_diagrams/PD_dim_1_test.csv'),\
    "persistence file path dim 1 does not exist"
assert isfile(save_path+'persistence_diagrams/plots/PD_dim_0_test.svg'),\
    "persistence diagram plot dim 0 does not exist"
assert isfile(save_path+'persistence_diagrams/plots/PD_dim_1_test.svg'),\
    "persistence diagram plot dim 1 does not exist"

# rotation invariant test, (sublevel not SEDT)
test_im_90 = np.rot90(test_image)
test_im_180 = np.rot90(test_im_90)
test_im_270 = np.rot90(test_im_180)
rotated_images = [
    ('test_090.tif',test_im_90),
    ('test_180.tif',test_im_180),
    ('test_270.tif', test_im_270)
    ]
for filename, rot_test_image in rotated_images:
    pers_hom.peristent_homology_sublevel_cubic(
    rot_test_image,
    filename,
    save_path)
    PD_0 = pd.read_csv(
        f"{save_path}persistence_diagrams/PD_dim_0_test.csv",
        header=None
        )
    PD_1 = pd.read_csv(
        f"{save_path}persistence_diagrams/PD_dim_1_test.csv",
        header=None
        )
    PD_0_rotated = pd.read_csv(
        f"{save_path}persistence_diagrams/PD_dim_0_{filename[:-4]}.csv",
        header=None
        )
    PD_1_rotated = pd.read_csv(
        f"{save_path}persistence_diagrams/PD_dim_1_{filename[:-4]}.csv",
        header=None
        )
    assert (PD_0_rotated == PD_0).all().all(),\
        f"different persistence values when rotated {filename[5:7]} degrees"
    assert (PD_1_rotated == PD_1).all().all(),\
        f"different persistence values when rotated {filename[5:7]} degrees"

# remove test files from system
for file in listdir(save_path+'idiagrams'):
    remove(save_path+'idiagrams/'+file)
rmdir(save_path+'idiagrams/')
for file in listdir(save_path+'persistence_diagrams/plots'):
    remove(save_path+'persistence_diagrams/plots/'+file)
rmdir(save_path+'persistence_diagrams/plots/')
for file in listdir(save_path+'persistence_diagrams'):
    remove(save_path+'persistence_diagrams/'+file)
rmdir(save_path+'persistence_diagrams/')

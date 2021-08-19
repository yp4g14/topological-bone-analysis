import numpy as np
from importlib import reload
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
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

filename = "test.tif"
save_path = "C:/"
pers_hom.peristent_homology_sublevel_cubic(
    test_image,
    filename,
    save_path,
    plot_persistence_diagrams=True)

# check it creates necessary paths
assert exists(save_path+'idiagrams/'), "idiagram path does not exist"
assert exists(save_path+'persistence_diagrams/'), "persistence diagram path does not exist"
assert exists(save_path+'persistence_diagrams/plots/'), "persistence diagram plot path does not exist"

# check it creates necessary files
assert isfile(save_path+'idiagrams/test.idiagram'), "idiagram file does not exist"
assert isfile(save_path+'persistence_diagrams/PD_dim_0_test.csv'), "persistence file dim 0 path does not exist"
assert isfile(save_path+'persistence_diagrams/PD_dim_1_test.csv'), "persistence file path dim 1 does not exist"
assert isfile(save_path+'persistence_diagrams/plots/PD_dim_0_test.svg'), "persistence diagram plot dim 0 does not exist"
assert isfile(save_path+'persistence_diagrams/plots/PD_dim_1_test.svg'), "persistence diagram plot dim 1 does not exist"

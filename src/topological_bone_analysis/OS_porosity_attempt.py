# %% [markdown]
# OSTEOSARCOMA

# %%
from topological_bone_analysis import preprocessing_images as preprocess
from topological_bone_analysis import run_process 
from topological_bone_analysis import persistent_homology_SEDT as pers_hom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import logging
from os import listdir, mkdir
from os.path import isfile, join, exists
logger = logging.getLogger("example")


# %% [markdown]
# Configure the parameters in order to run the SEDT porosity process on the SHG images

# %%
path = "/media/findlay/HDD/Ysanne_Backup/data/OS_SHG/"
feature_cols = [
    '0_num_points',
    '0_num_points_less_eq_-2',
    '0_avg_birth',
    '0_stddev_birth',
    '0_skew_birth',
    '0_percentile_25_birth',
    '0_percentile_75_birth',
    '0_iqr_birth',
    '0_avg_death',
    '0_stddev_death',
    '0_skew_death',
    '0_percentile_25_death',
    '0_percentile_75_death',
    '0_iqr_death',
    '0_pers_entropy',
    '1_num_points',
    '1_avg_birth',
    '1_stddev_birth',
    '1_skew_birth',
    '1_percentile_25_birth',
    '1_percentile_75_birth',
    '1_iqr_birth',
    '1_avg_death',
    '1_stddev_death',
    '1_skew_death',
    '1_percentile_25_death',
    '1_percentile_75_death',
    '1_iqr_death',
    '1_pers_entropy']
filenames_map = {
    "Female_A2_L4_SHG.tif":"CO, female",
    "Female_A2_L13_SHG.tif":"CO, female",
    "Female_A2_L14_SHG.tif":"CO, female",
    "Female_A3_B1_SHG.tif":"OS, female",
    "Female_A3_D3_SHG.tif":"OS, female",
    "Female_A3_E3_SHG.tif":"OS, female",
    "Male_A1_B2_SHG.tif":"CO, male",
    "Male_A2_K10_SHG.tif":"CO, male",
    "Male_A2_L8_SHG.tif":"CO, male",
    "Male_A3_A3_SHG.tif":"OS, male",
    "Male_A3_E2_SHG.tif":"OS, male",
    "Male_A3_E10_SHG.tif":"OS, male",
    }

# %% [markdown]
# Run the SEDT porosity process with above configs

# %%
stats_df, results = run_process.topological_porosity_analysis(
    path,
    logger,
    preprocess.otsu_threshold,
    patch_shape=500,
    stride=500,
    save_persistence_diagrams=False,
    analysis_plots=True,
    classification=False,
    feature_cols=feature_cols,
    filenames_map=filenames_map,
    runs=10,
    strat_col=None,
    cross_val='stratkfold',
    param_grid_SVC = {'C': [1,2,3], 'kernel': ('rbf','linear')}
)

# %% [markdown]
# Let's look at a couple example images, and their binary conversion

# %%
# filenames = list(filenames_map.keys())
# print(filenames)
# path = "/media/findlay/HDD/Ysanne_Backup/data/OS_SHG/"
# save_path = "/media/findlay/HDD/Ysanne_Backup/data/OS_SHG/OS_porosity_pynb/"
# binary_path = save_path+'binary/'
# if not exists(binary_path):
#     mkdir(binary_path)
# patch_path = save_path+'patch/'
# if not exists(patch_path):
#     mkdir(patch_path)

# num = 4
# OS_f = np.array(Image.open(path+filenames[0]))
# OS_m = np.array(Image.open(path+filenames[3]))
# CO_f = np.array(Image.open(path+filenames[6]))
# CO_m = np.array(Image.open(path+filenames[9]))
# images = [OS_f, OS_m, CO_f, CO_m]
# files = [filenames[0], filenames[3], filenames[6], filenames[9]]

# binary_images = []
# for i in range(num):
#     binary_im = preprocess.otsu_threshold(images[i])
#     Image.fromarray(binary_im.astype(np.uint8)).save(save_path+'binary/'+files[i])
#     binary_images.append(binary_im)

# fig,ax = plt.subplots(2,num)
# for i in range(num):
#     ax[0,i].imshow(images[i])
#     ax[1,i].imshow(binary_images[i], cmap='binary_r')
#     ax[0,i].get_xaxis().set_visible(False)
#     ax[1,i].get_xaxis().set_visible(False)
#     ax[0,i].get_yaxis().set_visible(False)
#     ax[1,i].get_yaxis().set_visible(False)
# plt.show()

# %% [markdown]
# Now let's take a patch of each

# %%
# patches = []
# for i in range(num):
#     patch = binary_images[i][800:1000,800:1000]
#     Image.fromarray(patch.astype(np.uint8)).save(save_path+'patch/'+files[i])
#     patches.append(patch)

# fig, ax = plt.subplots(num,1)
# for i in range(num):
#     ax[i].imshow(patches[i], cmap='binary_r')
# plt.show()

# %% [markdown]
# Convert to SEDT and display, then run persistent homology

# %%
# SEDT_patches = []
# for i in range(num):
#     SEDT_patch = preprocess.SEDT(patches[i])
#     SEDT_patches.append(SEDT_patch)

# from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# # make custom colourmap, set zero as midpoint
# specific_colors = LinearSegmentedColormap.from_list('name',
#     [(0,'k'),(0.3,'b'),(0.5,'w'),(0.8,'g'),(1,'k')],N=1000)

# fig, ax = plt.subplots(1,num,gridspec_kw = {'wspace':0.5, 'hspace':0.5}, figsize=(10,10))
# for i in range(num):
#     SEDT_patch = SEDT_patches[i]
#     vmin, vcenter, vmax = np.min(SEDT_patch), 0, np.max(SEDT_patch)
#     divnorm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax) 
#     im = ax[i].imshow(SEDT_patch, cmap=specific_colors, norm=divnorm)
#     ax[i].get_xaxis().set_visible(False)
#     ax[i].get_yaxis().set_visible(False)
#     divider = make_axes_locatable(ax[i])
#     cax = divider.append_axes("right", size="5%", pad=0.1)
#     cbar = plt.colorbar(im, cax=cax,orientation='vertical', ticks=[vmin,0,vmax])

# idiagram_path= save_path+'idiagrams/'
# interval_path = path+'persistence_intervals/'
# if not exists(idiagram_path):
#     mkdir(idiagram_path)
# if not exists(interval_path):
#     mkdir(interval_path)

# for i in range(len(SEDT_patches)):
#     SEDT_patch = SEDT_patches[i]
#     name = f"{files[i][:-4]}_SEDT"
#     pers_hom.peristent_homology_sublevel_cubic(
#         SEDT_patch,
#         name,
#         path,
#         plot_persistence_diagrams=False)
        

# %% [markdown]
# Now we can look at the histograms (on a log scale) for H0 quadrant 2

# %%
# intervals = []
# for i in range(num):
#     vals = pd.read_csv(
#         interval_path+files[i],
#         header=None,
#         names=['birth','death']
#     )
#     vals['filename'] = files[i]
#     intervals.append(vals)

# intervals = pd.concat(intervals)
# Q2 = intervals.iloc[intervals['birth']<= 0]
# Q2 = Q2.iloc[Q2['death']>0]

# fig,ax = plt.subplots(1,num)
# for i in range(num):
#     df = Q2.iloc[Q2['filename']==files[i]]
#     ax[i].hist(df, log=True)
# plt.show()


# %% [markdown]
# We see a spike where?
# This should relate to Haversian canals in healthy bone?

# %%
# plot the inverses of the large ones to check sizes.



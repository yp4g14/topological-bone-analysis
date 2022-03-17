import matplotlib.pyplot as plt
import topological_bone_analysis as tba
import numpy as np
import pandas as pd
from os import listdir, mkdir
from os.path import isfile
from PIL import Image
from skimage.segmentation import flood, flood_fill

dwn_binary_image_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/haversian2022_01_13_Time_13_51/downsized_binary/"
full_binary_image_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/haversian2022_01_13_Time_13_51/binary/"
image_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/"
interval_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/haversian2022_01_13_Time_13_51/persistence_intervals/" 
save_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/haversian2022_01_13_Time_13_51/fibres/"
stats_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/haversian2022_01_13_Time_13_51/all_statistics.csv"
plot_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/haversian2022_01_13_Time_13_51/plots/"
filenames_map = {
    'Female_A2_L14.tif':'F, CO',
    'Female_A2_L4.tif':'F, CO',
    'Female_A2_L13.tif':'F, CO',
    'Male_A1_B2.tif':'M, CO',
    'Male_A2_K10.tif':'M, CO',
    'Male_A2_L8.tif':'M, CO',
    'Female_A3_D3.tif':'F, OS',
    'Female_A3_E3.tif':'F, OS',
    'Female_A3_B1.tif':'F, OS',
    'Male_A3_E10.tif':'M, OS',
    'Male_A3_E2.tif':'M, OS',
    'Male_A3_A3.tif':'M, OS'
    }

# mkdir(save_path)
# stats_df = pd.read_csv(stats_path, index_col=0)

interval_filenames = [file for file in listdir(interval_path) if isfile(interval_path+file)]
intervals_list = []
for file in interval_filenames:
    ints = pd.read_csv(
        interval_path+file,
        names=['birth','death','b_y','b_x','d_y','d_x'])
    ints['dim'] = int(file[7])
    # ints['patch_number'] = int(file[-7:-4])
    ints['filename'] = file[9:-4]
    intervals_list.append(ints)
    del ints
intervals_df = pd.concat(intervals_list)
H0_intervals = intervals_df.loc[intervals_df['dim'] == 0]
H0_Q2_intervals = H0_intervals.loc[(H0_intervals['birth'] <= 0) & (H0_intervals['death'] > 0)]
# group by OS control male/female
H0_Q2_intervals['sex'] = H0_Q2_intervals['filename'].apply(lambda x: x[0])
H0_Q2_intervals['tag'] = H0_Q2_intervals['filename'].apply(lambda x: f"{x.split('_')[1]}_{x.split('_')[2]}")
tags = ['A1_B2','A2_K10','A2_L8','A2_L4','A2_L13','A2_L14',
'A3_A3','A3_B1','A3_D3','A3_E10','A3_E2','A3_E3']
condition = ['CO','CO','CO','CO','CO','CO','OS','OS','OS','OS','OS','OS']
tag_map = dict(zip(tags, condition))
H0_Q2_intervals['condition'] = H0_Q2_intervals['tag'].map(tag_map)
limit = -10
canals = H0_Q2_intervals.loc[H0_Q2_intervals['birth'] <= limit]
canals['b_x'] = canals['b_x'].astype(int)
canals['b_y'] = canals['b_y'].astype(int)
canals.dtypes

# pick a single file
tag = 'A2_k10'
filename='Male_A2_K10_SHG.tif'

# canals df
canals = canals.loc[canals['tag']=='A2_K10']
# binary image
binary_im_dwn = np.array(Image.open(dwn_binary_image_path+filename))
binary_im_full = np.array(Image.open(full_binary_image_path+filename))
im_full = np.array(Image.open(image_path+filename))

# need to scale up birth and death to got back to full image
canals['b_x_3'] = canals['b_x'] * 3
canals['b_y_3'] = canals['b_y'] * 3
canals['birth_3'] = canals['birth'] * 3
canals['death_3'] = canals['death'] * 3

plt.imshow(im_full)
im = im_full
sh = im.shape
num_canals = canals.shape[0]
# lets start the fibre circle idea
import matplotlib

for i in range(num_canals):
    i=8
    canal = canals.iloc[i]
    tol = 200
    left, right = max(0,canal.b_x_3-200), min(sh[0],canal.b_x_3+200)
    top, bottom = max(0,canal.b_y_3-200), min(sh[1],canal.b_y_3+200)
    region = im[top:bottom,left:right]
    binary_region = binary_im_full[top:bottom,left:right]
    fig,ax = plt.subplots()
    ax.imshow(region, cmap='binary_r')
    centre = (canal.b_x_3-left, canal.b_y_3-top)
    ax.scatter(centre[0],centre[1])
    cc = plt.Circle(centre,abs(canal.birth_3), alpha=0.5)
    ax.add_artist(cc)
    area = flood(binary_region, centre).astype(int)
    masked_array = np.ma.masked_where(area == 1, region)
    cmap = matplotlib.cm.gray
    cmap.set_bad(color='red', alpha=0.2)
    ax.imshow(masked_array, cmap=cmap)
    plt.savefig(save_path+f"ex_finding_canal_pixels_{i}.png")

import skimage.filters as ft
thresh = (region > ft.threshold_multiotsu(region))
upper, lower = 30, 12
thresh = 2*(region > upper) + ((region < upper) & (region >lower))
plt.imshow(thresh)


# # superlevel on image looks like?
# mina, maxa = min(region.flatten()), max(region.flatten())
# step = (maxa - mina)/16
# fig,ax = plt.subplots(4,4)
# for j in range(0,4):
#     for i in range(1,5):
#         ax[j,i-1].axis('off')
#         ax[j,i-1].imshow(region * (region > (maxa-j*4*step-i*step)),cmap='gray')
# plt.savefig(save_path+'superlevelset_filtration.png')

plt.hist(im[im>20].flatten(), bins=100)

import skimage.feature as feat
# calculate gradient of array
width = 20

upward_slice = region[0:centre[1],centre[0]-width:centre[0]+width]
side_slice = region[centre[1]-width:centre[1]+width,centre[0]:]

plt.imshow(upward_slice, cmap='gray')
plt.savefig(save_path+'upward_slice.png')
plt.imshow(side_slice, cmap='gray')
plt.savefig(save_path+'side_slice.png')


plt.imshow(binary_region, cmap='binary_r')
plt.savefig(save_path+'binary_region.png')
plt.imshow(region, cmap='gray')
plt.savefig(save_path+'region.png')

fig,ax = plt.subplots()
ax.imshow(np.gradient(upward_slice)[0], cmap='seismic')
plt.savefig(save_path+'gradient_upward_slice.png')
fig,ax = plt.subplots()
ax.imshow(np.gradient(upward_slice)[1], cmap='seismic')
plt.savefig(save_path+'gradient_upward_slice_1.png')

fig,ax = plt.subplots()
ax.imshow(np.gradient(side_slice)[0], cmap='seismic')
plt.savefig(save_path+'gradient_side_slice.png')
fig,ax = plt.subplots()
ax.imshow(np.gradient(side_slice)[1], cmap='seismic')
plt.savefig(save_path+'gradient_side_slice_1.png')

 
grad_side_slice = np.gradient(side_slice)[1]
pos= np.sum(grad_side_slice>0, axis=0)
neg = np.sum(grad_side_slice<0, axis=0)
plt.plot(pos-neg)
plt.savefig(save_path+'gradient_change_pos_neg_difference_1.png')

diff = pos-neg

where_diff = np.where(diff>20)[0]

from operator import itemgetter
from itertools import groupby
ranges =[]

for k,g in groupby(enumerate(where_diff),lambda x:x[0]-x[1]):
    group = (map(itemgetter(1),g))
    group = list(map(int,group))
    ranges.append((group[0],group[-1]))

ridge_width = [j-i for (i,j) in ranges]


## voronoi 
from scipy.spatial import Voronoi, voronoi_plot_2d

points = np.array(canals[['b_x_3','b_y_3']])
fig,ax = plt.subplots()
ax.imshow(im_full, cmap='gray')
voronoi_plot_2d(Voronoi(points), show_vertices=True, line_colors='orange',line_width=2, line_alpha=0.6, point_size=2, ax=ax)
plt.savefig(save_path+'voronoi_canals.png')

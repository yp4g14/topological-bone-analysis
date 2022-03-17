import matplotlib.pyplot as plt
import topological_bone_analysis as tba
import numpy as np
import pandas as pd
from os import listdir, mkdir
from os.path import isfile
from PIL import Image
from skimage.segmentation import flood, flood_fill

binary_image_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/haversian2022_01_13_Time_13_51/downsized_binary/"
image_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/haversian2022_01_13_Time_13_51/downsized_binary/"
interval_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/haversian2022_01_13_Time_13_51/persistence_intervals/" 
save_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/haversian2022_01_13_Time_13_51/haversian_analysis/"
# mkdir(save_path)
interval_filenames = [file for file in listdir(interval_path) if isfile(interval_path+file)]

intervals_list = []
for file in interval_filenames:
    ints = pd.read_csv(
        interval_path+file,
        names=['birth','death','b_x','b_y','d_x','d_y'])
    ints['dim'] = int(file[7])
    # ints['patch_number'] = int(file[-7:-4])
    ints['filename'] = file[9:-4]
    intervals_list.append(ints)
    del ints

intervals_df = pd.concat(intervals_list)
H0_intervals = intervals_df.loc[intervals_df['dim'] == 0]
H0_Q2_intervals = H0_intervals.loc[(H0_intervals['birth'] <= 0) & (H0_intervals['death'] > 0)]

# # per image, plot histogram of births
# filenames = list(set(H0_Q2_intervals['filename']))
# for name in filenames:
#     full_binary_image = np.array(Image.open(binary_image_path+name+'.tif'))
#     single_im_df = H0_Q2_intervals.loc[H0_Q2_intervals['filename'] == name]
#     fig,ax = plt.subplots(2,1, gridspec_kw={'height_ratios':[3,1]})
#     ax[0].imshow(full_binary_image, cmap='binary_r')
#     ax[1].hist(single_im_df['birth'],log=True,bins=100)
#     plt.savefig(save_path+'SEDT_H0_Q2_birth_histogram'+name+'.png')

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
canals.groupby(['condition','sex', 'tag']).count()

f_co = H0_Q2_intervals.loc[(H0_Q2_intervals['sex']=='F') & (H0_Q2_intervals['condition']=='CO')]
m_co = H0_Q2_intervals.loc[(H0_Q2_intervals['sex']=='M') & (H0_Q2_intervals['condition']=='CO')]
f_os = H0_Q2_intervals.loc[(H0_Q2_intervals['sex']=='F') & (H0_Q2_intervals['condition']=='OS')]
m_os = H0_Q2_intervals.loc[(H0_Q2_intervals['sex']=='M') & (H0_Q2_intervals['condition']=='OS')]
dfs = [f_co, m_co, f_os, m_os]
labels = ['F Co', 'M Co', 'F OS', 'M OS']

fig,ax = plt.subplots(4, sharex=True)
for i in range(4):
    ax[i].hist(dfs[i]['birth'], log=True)
    ax[i].set_ylabel(labels[i])
plt.savefig(save_path+'grouped canals comparison.png')

# plot SEDT 3 images with bettter colour scale
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

SEDT3_path = "/media/ysanne/TOSHIBA EXT/data/OS_SHG/haversian2022_01_13_Time_13_51/SEDT/"
SEDT_3_ims = []
for name in filenames:
    image = np.asarray(Image.open(SEDT3_path+name+'.tif'))

    specific_colors = LinearSegmentedColormap.from_list(
        'name',
        [(0,'k'),(0.2,'b'),(0.5,'w'),(0.8,'g'),(1,'k')],N=1000)
    vmin, vcenter = np.min(image), 0
    vmax = sorted(list(set(image.flatten())))[-2]
    divnorm = TwoSlopeNorm(vmin=vmin, vmax=vmax,vcenter=vcenter) 
    cbar = plt.cm.ScalarMappable(norm=divnorm, cmap=specific_colors)

    fig,ax = plt.subplots(1)
    plt.axis('off')
    ax.grid(False)
    masked_array = np.ma.masked_where(image == np.inf, image)
    cmap = matplotlib.cm.Blues  # Can be any colormap that you want after the cm
    cmap = specific_colors
    cmap.set_bad(color='grey')
    im = ax.imshow(masked_array, cmap=cmap, norm=divnorm)
    ax.set_title('SEDT')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax,orientation='vertical', ticks=[vmin,0,vmax])
    cbar.outline.set_visible(False)
    plt.savefig(save_path+'SEDT_3'+name+'.png')

# regular plots for full flooded images
from topological_bone_analysis import plots
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

stats_df = pd.read_csv(stats_path, index_col=0)
feature_cols = [col for col in stats_df.columns if col[0]=='0']
plots.analysis_plots(
    stats_df,
    filenames_map,
    plot_path,
    feature_cols)

# plot the biths of the canals on the images
limit = -10
for i in range(len(filenames)):
    df = H0_Q2_intervals.loc[H0_Q2_intervals['filename'] == filenames[i]]
    df = df.loc[df['birth'] < limit]
    num_to_plot = df.shape[0]
    fig,ax = plt.subplots(1)
    binary_im = np.array(Image.open(binary_image_path+filenames[i]+'.tif'))
    ax.imshow(binary_im, cmap='binary_r')
    ax.scatter(df['b_y'], df['b_x'], color='red', s=10)
    ax.set_xlabel(f"{df.shape[0]} canals larger than {abs(limit)}")
    plt.savefig(save_path+'canal_centres_bigger_10_'+filenames[i]+'.png')
    plt.close()

# lets start the fibre circle idea
image = np.array(Image.open(binary_image_path+filenames[-2]+'.tif'))
df = H0_Q2_intervals.loc[H0_Q2_intervals['filename'] == filenames[-2]]
df = df.loc[df['birth'] < limit]
num_canals = df.shape[0]
# for i in range(num_canals)
canal = df.iloc[0]
b_x, b_y = int(canal['b_x']), int(canal['b_y'])
tol = min(b_x,b_y, 200)
region = image[-tol+b_x:b_x+tol,-tol+b_y:b_y+tol]
plt.imshow(region, cmap='binary_r')
# take slices out in directions and look for gradient changes
## HERE
def SEDT_3(
    image
    ):
    """
    Calculates a Signed Euclidean Distance Transform of a binary image array.
    Leaves any values that are connected to black corners (using flood) set to np.inf
    
    Technically before flooding adds one pixel deep border in black, to avoid erroneous white coreners, then floods, then removes this for final image

    Args:
        image : (numpy array) trinary image to transform.

    Returns:
        sedt_3 : (numpy array) SEDT 3 phase of image.
    """
    from scipy import ndimage
    from skimage.segmentation import flood

    # pad image with border width 1 of value 0
    image_to_flood = np.pad(image, ((1,1),(1,1)), 'constant', constant_values=0)
    # flood from all black (0) corners
    x,y = [0,0,-1,-1],[0,-1,0,-1]
    all_floods = flood(image_to_flood, (x[0],y[0])).astype(int)
    for i in range(1,4):
        all_floods += flood(image_to_flood, (x[i],y[i])).astype(int)

    all_floods = (all_floods>0).astype(int)
    all_floods = np.where(all_floods == 1, -np.inf,1)
    #trim off border we added above
    all_floods = all_floods[1:-1,1:-1]
    
    #SEDT
    inverse_image = np.logical_not(image).astype(int)
    edt_1_to_0 = ndimage.distance_transform_edt(image)
    edt_0_to_1 = ndimage.distance_transform_edt(inverse_image)

    # negative distances for 0 (pore/air) to 1 (material)
    edt_0_to_1 = - edt_0_to_1

    # where image is True (material) replace the distance 
    # with distance from material to pore/air
    neg = np.multiply(inverse_image, edt_0_to_1)
    pos = np.multiply(image, edt_1_to_0)
    sedt = neg+pos
    sedt_3 = sedt * all_floods
    return sedt_3

sedt_3 = SEDT_3(full_binary_image)

import homcloud.interface as hc

# down scale images
def down_scale_image(im, factor=2):
    og_shape = im.shape
    div_len = im.shape[0] % factor
    div_wid = im.shape[1] % factor
    if div_len != 0:
        print(f"droppping {div_len} rows")
        im  = im[:-div_len,:]
    if div_wid != 0:
        print(f"droppping {div_wid} cols")
        im = im[:,:-div_wid]
    print(f"og shape {og_shape} \n new shape before downscale {im.shape}")

    b=im.shape[0]//factor
    smol_im = im.reshape(-1, factor, b, factor).sum((-1, -3)) / (factor**2)
    print(f"final shape {smol_im.shape}")
    return smol_im

smol_binary_im = down_scale_image(full_binary_image, factor=3)
smol_binary_im.shape

image  = SEDT_3(smol_binary_im)
filename = 'sedt_3_downsampled_factor_3.tif'

idiagram_path=save_path+'idiagrams/'
idiagram_filename = filename[:-4]+".idiagram"
interval_path = save_path+'persistence_intervals/'

import pandas as pd
int_0 = pd.read_csv(f"{save_path}persistence_intervals/PD_dim_0_{filename[:-4]}.csv", names=['birth','death','b_x','b_y','d_x','d_y'])
q2 = int_0.loc[(int_0['birth']<0)&(int_0['death']>0)]


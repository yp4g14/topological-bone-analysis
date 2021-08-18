
import numpy as np
from math import ceil, floor
from skimage.util import view_as_windows
from itertools import product
from scipy import ndimage

# def trim_original(image, edge_val=0):
#     """Trims an image of full rows and columns containing edge_val

#     Args:
#         image (numpy array): image as numpy array
#         edge_val (int, optional): The edge value to trim. Defaults to 0.

#     Returns:
#         image as numpy array: trimmed image
#     """
#     # trim left side
#     background=True
#     left=0
#     while background == True:
#         left_col = image[:,left]
#         if (left_col == edge_val).all():
#             left+=1
#         else:
#             background = False
#     background=True
#     # right side
#     right=1
#     while background == True:
#         right_col = image[:,-right]
#         if (right_col == edge_val).all():
#             right+=1
#         else:
#             background = False
#     # top side
#     background=True
#     top = 0
#     while background == True:
#         top_row = image[top,:]
#         if (top_row == edge_val).all():
#             top+=1
#         else:
#             background = False
#     # bottom side
#     background=True
#     bottom = 1
#     while background == True:
#         bottom_row = image[-bottom,:]
#         if (bottom_row == edge_val).all():
#             bottom+=1
#         else:
#             background = False
#     trim_vals = np.array([left, right, top, bottom])
#     if (trim_vals >=1).any():
#         if bottom == 0 and right == 0:
#             trimmed_image = image[top:, left:]
#         elif bottom == 0:
#             trimmed_image = image[top:, left:-right]
#         elif right == 0:
#             trimmed_image = image[top:-bottom, left:]
#         else:
#             trimmed_image = image[top:-bottom, left:-right]

#         print(f"image size {image.shape} \n trimmed image size {trimmed_image.shape}")    
#         return trimmed_image
#     else:
#         print(f"image not trimmed as no full borders of edge value {edge_val} found")
#         return image

def trim(image, edge_val=0):
    """Trims an image removing full rows and columns containing edge_val from the borders

    Args:
        image (numpy array): image as numpy array
        edge_val (int, optional): The edge value to trim. Defaults to 0.

    Returns:
        image as numpy array: trimmed image
    """
    n_rows, n_cols = image.shape
    col_sums = image.sum(axis=0)
    row_sums = image.sum(axis=1)
    non_zero_cols = np.where(col_sums!=n_rows*edge_val)[0]
    non_zero_rows = np.where(row_sums!=n_cols*edge_val)[0]
    left_trim = min(non_zero_cols)
    right_trim = max(non_zero_cols)
    top_trim = min(non_zero_rows)
    bottom_trim = max(non_zero_rows)
    trimmed_image = image[top_trim:bottom_trim+1,left_trim:right_trim+1]
    return trimmed_image


# basic test image 
# image =  np.array([[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]])
# trimmed_image = np.array([[1,2],[3,4]])

def extract_patches(
    image,
    patch_shape,
    pad_val,
    percent_background=1.0,
    background=0):
    """
    Takes a 2D image array and cuts it into non-overlapping square patches of patch_shape.
    To do this it first pads image to be exact multiples of the patch shape in each direction.

    Args:
        image (numpy array): numpy array of image
        patch_shape (int) : length (width) of desired square patches.
        pad_val (int) : value in [0,1] that pads the binary image
        percent_background (float) : percentage in [0,1] of patch that is background for it to be discarded. Default is 1.
        set to 0 to keep all patches
        background (int, optional) : value that is the image background

    Returns:
        image : (array) padded image
        patches : (array) array of patches
        patches[i] will return the i^th patch.
        patch_coords : list of all patch coordinates (top left per patch)
        image.shape : tuple dimensions of padded image

    """
    num_rows, num_cols = image.shape

    # pad image
    if type(pad_val) != int:
        print(f"padded value given {pad_val} is not an integer, has type {type(pad_val)}")
    length_to_add = ceil(image.shape[0]/patch_shape)*patch_shape - image.shape[0]
    width_to_add = ceil(image.shape[1]/patch_shape)*patch_shape - image.shape[1]

    image = np.pad(image,
                    ((ceil(length_to_add/2),floor(length_to_add/2)), (ceil(width_to_add/2),floor(width_to_add/2))),
                    'constant',
                    constant_values=(pad_val,pad_val))
    num_rows, num_cols = image.shape
    
    # take patches of padded image
    stride=patch_shape
    patches = view_as_windows(image,
                              patch_shape,
                              stride)

    p_num_rows, p_num_cols, patch_height, patch_width = patches.shape
    num_patches = p_num_rows * p_num_cols
    patches =  np.reshape(patches, (num_patches, patch_height, patch_width))
    
    # get the coordinates of all the patches
    col_coords = np.array([])
    base_coords = np.array([i*(patch_width) for i in range(int(num_cols/patch_width)+1)])
    offsets = np.arange(0,patch_width, step=stride)
    for offset in offsets:
        starts = base_coords + offset
        starts = starts[starts <= num_cols - patch_width]
        col_coords = np.concatenate([col_coords, starts])
    col_coords = np.array(sorted(col_coords))
    
    row_coords = np.array([])
    base_coords = np.array([i*(patch_height) for i in range(int(num_rows/patch_height)+1)])
    offsets = np.arange(0,patch_height, step=stride)
    for offset in offsets:
        starts = base_coords + offset
        starts = starts[starts <= num_rows - patch_height]
        row_coords = np.concatenate([row_coords, starts])
    row_coords = np.array(sorted(row_coords))
    
    patch_coords = list(product(row_coords, col_coords))
    
    # check number of coordinates found is the same as number of patches taken
    if patches.shape[0] != len(patch_coords):
        print(f"{patches.shape[0]} patches found, {patch_coords} coordinates found")
    return image, patches, patch_coords, image.shape


def SEDT(images):
    """
    Calculates a Signed Euclidean Distance Transform of a list of images.

    Args:
        images : (list of 2d numpy arrays) List of images.

    Returns:
        sedt_images : (list of 2d numpy arrays) List of SEDT of images.

    """
    sedt_images = []
    for image in images:
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
        sedt_images.append(sedt)

    return sedt_images

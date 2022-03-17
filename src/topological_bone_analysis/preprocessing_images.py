import numpy as np
from scipy import ndimage
from math import ceil, floor
from skimage.util import view_as_windows
from skimage.filters import threshold_otsu
from itertools import product
from os import listdir
from os.path import isfile, join
from PIL import Image
from . import utils as ut
from skimage.segmentation import flood

def trim(
    image,
    edge_val=0
    ):
    """Trims an image array removing full rows and columns containing edge_val
     from the borders

    Args:
        image (numpy array): image as numpy array
        edge_val (int, optional): The edge value to trim. Defaults to 0.

    Returns:
        image as numpy array: trimmed image
    """
    edge_cols = np.any((image!=edge_val),axis=0)
    edge_rows = np.any((image!=edge_val),axis=1)

    non_zero_cols = np.where(edge_cols)[0]
    non_zero_rows = np.where(edge_rows)[0]

    left_trim = min(non_zero_cols)
    right_trim = max(non_zero_cols)

    top_trim = min(non_zero_rows)
    bottom_trim = max(non_zero_rows)

    trimmed_image = image[top_trim:bottom_trim+1,left_trim:right_trim+1]
    return trimmed_image

def otsu_threshold(
    image
    ):
    """Takes a grayscale image and binarizes using Otsu's threshold.

    Args:
        image (numpy array): grayscale image (numpy array)
    
    Returns:
        binary_image (numpy array): binary ([0,1]) image (numpy array)
    """
    threshold_val = threshold_otsu(image)
    binary_image = (image > threshold_val).astype(np.uint8)
    return binary_image

def minimum_threshold(image):
    """Takes a grayscale image and binarizes using minimum method from the
    skimage.filters

    Args:
        image (numpy array): grayscale image (numpy array)

    Returns:
        binary_image (numpy array): binary ([0,1]) image (numpy array)
    """
    from skimage.filters import threshold_minimum
    val = threshold_minimum(image)
    binary_im = (image > val).astype(int)
    return binary_im

def extract_patches(
    image,
    patch_shape,
    pad_val=0,
    stride=None
    ):
    """
    Takes a 2D image array and cuts it into non-overlapping square patches 
    of patch_shape. To do this it first pads image to be exact multiples of the
    patch shape in each direction. The image is padded with the constant 
    pad_val, which defaults to 0

    Args:
        image (numpy array): numpy array of image
        patch_shape (int) : length (width) of desired square patches.
        pad_val (int, optional): value in [0,1] that pads the binary image. 
            efaults to 0.
        stride (int, optional): stride across image before taking next patch. 
            Defaults to patch_shape so patches don't overlap

    Returns:
        image : (array) padded image
        patches : (array) array of patches, patches[i] will return i^th patch.
        patch_coords : list of all patch coordinates (top left per patch)
    """
    num_rows, num_cols = image.shape

    # pad image
    length_to_add = ceil(num_rows/patch_shape)*patch_shape - num_rows
    width_to_add = ceil(num_cols/patch_shape)*patch_shape - num_cols

    image = np.pad(image,
                    ((ceil(length_to_add/2),floor(length_to_add/2)),
                        (ceil(width_to_add/2),floor(width_to_add/2))),
                    'constant',
                    constant_values=(pad_val,pad_val))
    num_rows, num_cols = image.shape
    
    # take patches of padded image
    if stride is None:
        stride=patch_shape

    patches = view_as_windows(image,
                              patch_shape,
                              stride)

    p_num_rows, p_num_cols, patch_height, patch_width = patches.shape
    num_patches = p_num_rows * p_num_cols
    patches =  np.reshape(patches, (num_patches, patch_height, patch_width))
    
    # get the coordinates of all the patches
    col_coords = get_coords(stride, num_cols, patch_shape)
    row_coords = get_coords(stride, num_rows, patch_shape)
    patch_coords = list(product(row_coords, col_coords))
    
    return image, patches, patch_coords

def get_coords(
    stride,
    axis_size,
    patch_shape
    ):
    """Calculates top left patch coordinate for image with axis size length,
    with patch_shape sized patches.

    Args:
        stride (int): stride between patches
        axis_size (int): length of axis to cut into patches
        patch_shape (int): length of patch

    Returns:
        numpy array: coordinates along axis of patches length patch_shape,
            stride distance apart 
    """
    coords = np.array([])
    base_coords = np.array(
        [i*(patch_shape) for i in range(int(axis_size/patch_shape)+1)]
        )
    offsets = np.arange(0,patch_shape, step=stride)
    for offset in offsets:
        starts = base_coords + offset
        starts = starts[starts <= axis_size - patch_shape]
        coords = np.concatenate([coords, starts])

    coords = np.array(sorted(coords))
    return coords

def SEDT(
    image
    ):
    """
    Calculates a Signed Euclidean Distance Transform of an image array.

    Args:
        image : (numpy array) binary image to transform.

    Returns:
        sedt_image : (numpy array) SEDT of image.
    """
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
    return sedt

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
def image_to_patches(
    path,
    filename,
    logger,
    patch_shape,
    stride,
    pad_val=0,
    percentage_background=1,
    background_val=None,
    trim_first=True,
    edge_val=0):
    binary_path = path+'binary/'
    padded_path = path+'padded/'
    patch_path = path+'patches/'

    #make parameter list to save in run output location
    params = [f"path: {path}",
              f"patch shape: {patch_shape}",
              f"stride: {stride}",
              f"percentage background: {percentage_background}",
              f"background threshold: {background_val}"]
    # get files
    image = ut.import_images(binary_path, [filename])[0]
    
    # OPTIONAL TRIM FUNCTION 
    if trim_first ==True:
        image = trim(image, edge_val)

    #total pixels per patch
    total_pixels = patch_shape**2
    
    # extracts square patches size patch_size x patch_size stride distance apart
    padded_image, patches, coords = extract_patches(
        image,
        patch_shape,
        pad_val=pad_val,
        stride=stride)
    
    # check number of coordinates found is the same as number of patches taken
    if patches.shape[0] != len(coords):
        logger.warning(f"{patches.shape[0]} patches found\
        but {len(coords)} coordinates found")

    Image.fromarray(padded_image).save(padded_path+filename)
    im_shape = padded_image.shape
    # test for a single patch if the percentage of pixels
    # under the background threshold is more than percentage_background permitted
    patch_index = 1
    patches_discarded = 0
    for j in range(patches.shape[0]):
        patch = patches[j]
        # how much of the patch is background?
        patch_background_percentage = np.sum(patch <= background_val)/total_pixels
        # is the patch low enough background to save?
        if  (patch_background_percentage < percentage_background):
            #save patch as image
            image_patch = Image.fromarray(patch)
            image_patch.save(f"{patch_path}{filename[:-4]}_{'{:03d}'.format(patch_index)}.tif")
            with open(f"{path}patch_coords.csv", "a") as outfile:
                outfile.write(f"{filename},{im_shape[0]},{im_shape[1]},{patch_index},{patch_shape},{patch_shape},{coords[j][0]},{coords[j][1]}\n")
            #increment index for save filename only if index has been used
            patch_index +=1
        else:
            patches_discarded += 1
    logger.info(f"image: {filename} completed, \
                {patch_index-1} patches saved, \
                {patches_discarded} patches discarded")
    params.append(f"image: {filename} completed, \
                {patch_index-1} patches saved, \
                {patches_discarded} patches discarded")

    with open(path+"patch_params.txt", "a") as filehandle:
        for param in params:
            filehandle.write('%s\n' % param)

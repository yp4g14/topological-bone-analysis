import numpy as np
from scipy import ndimage
from math import ceil, floor
from skimage.util import view_as_windows
from skimage.filters import threshold_otsu
from itertools import product
from os import listdir, rmdir, remove

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
    binary_image = (image > threshold_val).astype(int)
    return binary_image


# def custom_threshold(
#     image,
#     threshold=10):
#     """Calculates the custom threshold and applies to a grayscale image.

#     Args:
#         image (array): [description]
#         threshold (int, optional): [description]. Defaults to 10.
#     """
#     binary_im = (image > threshold).astype(int)
#     return binary_im

# from skimage.filters import threshold_minimum
# def minimum(
#     image,
#     binary_path,
#     filename,
#     logger):
#     val = threshold_minimum(image)
#     binary_im = (image > val).astype(int)
#     return binary_im

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
    
    # check number of coordinates found is the same as number of patches taken
    if patches.shape[0] != len(patch_coords):
        print(f"{patches.shape[0]} patches found,\
         {patch_coords} coordinates found")
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

def image_to_patches(
    path,
    filename,
    save_path,
    logger,
    patch_shape,
    stride,
    pad_val=0,
    file=None,
    percentage_background=1,
    background_val=None,
    trim_first=True,
    edge_val=0):

    #make parameter list to save in run output location
    params = [f"path: {path}",
              f"patch shape: {patch_shape}",
              f"stride: {stride}",
              f"percentage background: {percentage_background}",
              f"background threshold: {background_val}"]
    # get files
    if file == None:
        filenames = [file for file in listdir(path) 
                     if isfile(join(path, file))]
    else:
        filenames = [file]
    params.append(f"filenames: {filenames}")
    image = ut.import_images(path, filenames)
    
    # OPTIONAL TRIM FUNCTION 
    if trim_first ==True:
        image = trim(image, edge_val)

    #total pixels per patch
    total_pixels = patch_shape**2
    # take patches, test how much background is in the patch, save
    with open(f"{patch_path}patch_coords.csv", "w") as outfile:
        outfile.write("filename,image_shape_x,image_shape_y,patch_number,patch_width,patch_height,coord_array_row,coord_array_col\n")
    
    # extracts square patches size patch_size x patch_size stride distance apart
    padded_image, patches, coords, im_shape = extract_patches(
        image,
        patch_shape,
        stride,
        pad_val)
    
    Image.fromarray(padded_image).save(padded_ims_path+filename)

    # test for a single patch if the percentage of pixels
    # under the background threshold is more than percentage_background permitted
    patch_index = 1
    patches_discarded = 0
    for j in range(patches.shape[0]):
        patch = patches[j]
        coord = coords[j]
        # how much of the patch is background?
        patch_background_percentage = np.sum(patch <= background_val)/total_pixels
        # is the patch low enough background to save?
        if  (patch_background_percentage < percentage_background):
            #save patch as image
            image_patch = Image.fromarray(patch)
            image_patch.save(f"{patch_path}{filename[:-4]}_{'{:03d}'.format(patch_index)}.tif")
            with open(f"{patch_path}patch_coords.csv", "a") as outfile:
                outfile.write(f"{filename},{im_shape[0]},{im_shape[1]},{patch_index},{patch_width},{patch_height},{coords[j][0]},{coords[j][1]}\n")
            #increment index for save filename only if index has been used
            patch_index +=1
        else:
            patches_discarded += 1
    logger.info(f'''image: {filename} completed, 
                {patch_index-1} patches saved,
                {patches_discarded} patches discarded''')
    params.append(f'''image: {filename} completed, 
                {patch_index-1} patches saved,
                {patches_discarded} patches discarded''')

    with open(patch_path+'params.txt', 'w') as filehandle:
        for param in params:
            filehandle.write('%s\n' % param)

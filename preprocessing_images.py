import numpy as np
from scipy import ndimage

def trim(image, edge_val=0):
    """Trims an image array removing full rows and columns containing edge_val from the borders

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


def SEDT(image):
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

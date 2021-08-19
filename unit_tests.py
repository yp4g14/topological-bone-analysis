import numpy as np
from importlib import reload
from itertools import product

import preprocessing_images as pre_process
reload(pre_process)

# TEST - trim
######################################################
# basic test image 
test_im_0_edges =  np.array([[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]])
trim_test_im_0_edges = np.array([[1,2],[3,4]])
test_im_1_edges =  np.array([[1,1,1,1],[1,3,2,1],[1,3,4,1],[1,1,1,1]])
trim_test_im_1_edges = np.array([[3,2],[3,4]])

#check shape matches
assert pre_process.trim(test_im_0_edges,edge_val=0).shape == trim_test_im_0_edges.shape, "Shape doesn't match"
assert pre_process.trim(test_im_1_edges,edge_val=1).shape == trim_test_im_1_edges.shape, "Shape doesn't match"
#check values match
assert (pre_process.trim(test_im_0_edges,edge_val=0) == trim_test_im_0_edges).all(), "trim failed unit test, zero background value"
assert (pre_process.trim(test_im_1_edges, edge_val=1) == trim_test_im_1_edges).all(), "trim failed unit test, non-zero background value"

# TEST - threshold otsu
######################################################
test_image =  np.array([
    [rt2,rt2,rt5,rt5],
    [rt2,rt2,rt5,rt5],
    [rt5,rt5,rt2,rt2],
    [rt5,rt5,rt2,rt2]])
test_binary = np.array([
    [0,0,1,1],
    [0,0,1,1],
    [1,1,0,0],
    [1,1,0,0]])

binary_image, threshold_val = pre_process.otsu_threshold(test_image)

assert (test_binary.shape == binary_image.shape), "otsu threshold, test failed binary image incorrect shape"
assert (set(list(binary_image.flatten())) == {0,1}), "otsu threshold, test failed binary has values other than {0,1}"
assert (test_binary == binary_image).all(), "otsu threshold, test failed binary doesn't match"

# TEST  - extract_patches
######################################################
test_input_im = np.array([
    [1,0,0,0,1,1,1],
    [1,0,0,0,1,0,1],
    [1,0,0,0,1,1,1],
    [1,1,0,0,0,0,0],
    [1,1,1,0,0,1,0],
    [1,1,0,0,0,0,0]])
test_padded_image = np.array([
    [0,1,0,0,0,1,1,1,0],
    [0,1,0,0,0,1,0,1,0],
    [0,1,0,0,0,1,1,1,0],
    [0,1,1,0,0,0,0,0,0],
    [0,1,1,1,0,0,1,0,0],
    [0,1,1,0,0,0,0,0,0]])
test_coords = list(product([0,3],[0,3,6]))
test_patches = np.array([
    test_padded_image[0:3,0:3],
    test_padded_image[0:3:,3:6],
    test_padded_image[0:3,6:9],
    test_padded_image[3:6,0:3:],
    test_padded_image[3:6,3:6],
    test_padded_image[3:6,6:9]])
padded_image, patches, coords, shape = pre_process.extract_patches(test_input_im, patch_shape=3, pad_val=0)

assert padded_image.shape == test_padded_image.shape, "extract patches padded to incorrect shape"
assert (padded_image.shape[0]%3 == 0) and (padded_image.shape[1]%3 == 0), "failed to pad to integer multiple of patch shape"
assert (test_padded_image == padded_image).all(), "extract patches padded with incorrect value"
for i in range(len(test_patches)):
    assert (test_coords[i] == coords[i]), "extract patches coordinates don't match"
    assert (test_patches[i] == patches[i]).all(), "extract patches failed to extract correct patches"

# TEST  - SEDT (Signed Euclidean distance transform)
######################################################
test_input_im = np.array([
    [1,0,0,0,1,1,1],
    [1,0,0,0,1,0,1],
    [1,0,0,0,1,1,1],
    [1,1,0,0,0,0,0],
    [1,1,1,0,0,1,0],
    [1,1,0,0,0,0,0]])
from math import sqrt
rt2 = sqrt(2)
rt5 = sqrt(5)
test_output_im = np.array([
    [1,-1,-2,-1,1,1,rt2],
    [1,-1,-2,-1,1,-1,1],
    [1,-1,-rt2,-1,1,1,1],
    [rt2,1,-1,-rt2,-1,-1,-1],
    [rt5,rt2,1,-1,-1,1,-1],
    [2,1,-1,-rt2,-rt2,-1,-rt2]])

#check the shape isn't changed
assert pre_process.SEDT(test_input_im).shape == test_input_im.shape, "SEDT changed shape of test image"
# check the distance values are correct for test image
assert(pre_process.SEDT(test_input_im) == test_output_im).all(), "SEDT function failed test"

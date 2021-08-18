import numpy as np
from importlib import reload

import preprocessing_images as pre_process
reload(pre_process)

# TEST THE TRIM FUNCTION
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


# Signed Euclidean distance transform
######################################################
test_input_im = np.array([
    [1,0,0,0,1,1,1],
    [1,0,0,0,1,0,1],
    [1,0,0,0,1,1,1],
    [1,1,0,0,0,0,0],
    [1,1,1,0,0,1,0],
    [1,1,0,0,0,0,0]])
test_output_im = np.array([
    [1,0,0,0,1,1,1],
    [1,0,0,0,1,0,1],
    [1,0,0,0,1,1,1],
    [1,1,0,0,0,0,0],
    [1,1,1,0,0,1,0],
    [1,1,0,0,0,0,0]])

#check the shape isn't changed
assert pre_process.SEDT(test_input_im).shape == test_input_im.shape, "SEDT changed shape of test image"
# check the distance values are correct for test image
assert(pre_process.SEDT(test_input_im) == test_output_im).all(), "SEDT function failed test"

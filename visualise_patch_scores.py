# -*- coding: utf-8 -*-
import logging
from os.path import isfile, join
from os import listdir
from importlib import reload
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import ceil, floor
from matplotlib import gridspec
from matplotlib import rcParams
import datetime as dt
import utils as ut
reload(ut)

logger = logging.getLogger("visualisation")
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(stream_formatter)
logger.addHandler(console)

def visualise_patch_scores(
    logger,
    image_path,
    patch_path,
    score_path,
    save_path,
    colour='bwr',
    normalise_scores='per_image',
    alpha=0.3,
    score_column='score',
    quadrant=None
    ):
    """Visualises patch scores on a full image.
    Takes paths to the images/scores csvs and reads the data.
    Combines the scores and patch coordinates on filename and 
    patch_number columns. Calls create_visualisations_patch_scores per image,
    which will save a visualisation of the patch scores on the full image
    in directory save_path.

    Args:
        logger (logger.logging object): logging
        image_path (string): location of full images as string
        patch_path (string): location of patches as string
        score_path (string): location of 'scores' per patch as string
        save_path (string): location to save visualisations as string
        colour (str, optional): matplotlib colourscheme. 
        Defaults to 'bwr' for blue white red.
        normalise_scores (str, optional): normalise scores options
            'per_image' normalises the scores per image
            'per_set' normalises the scores per image set contained in scores
            'no' doesn't normalise scores, colour max is max score per image
            'no_set' doesn't normalise scores, colour max is max score per
            image set. Defaults to 'per_image'.
        alpha (float, optional): transparency of colour layer. Defaults to 0.3.
        score_column (str): name of score column to visualise.
        quadrant (int, optional): If not none, reduces scores to 
            scores for quadrant column is quadrant. If None does not limit.
            Defaults to None.
    """
    # create save directory if neccesary
    ut.directory(save_path)
    
    # read in patch scores
    scores = pd.read_csv(score_path, index_col=0)

    if score_column not in scores.columns:
        logger.error(f"Couldn't find {score_column} column in scores")

    if quadrant is not None:
        scores = scores.loc[scores['quadrant'] == quadrant]

    # read in patch coords usually output by images_to_patches
    coords = pd.read_csv(patch_path+'/patch_coords.csv')

    # join scores and coords datasets on the filename and patch_number
    # drops filename/patch_name combinations that aren't in BOTH dataframes
    scores = coords.merge(scores, on=['filename','patch_number'], how='inner')
    
    # get the set of image names in the data
    image_names = list(set(scores['filename']))

    # check can find all image files in image_path,
    # warn and remove from list if not found
    im_filenames = [file for file in listdir(image_path) 
                    if isfile(join(image_path, file))]
    
    im_filenames = ut.check_ext(im_filenames, ['tif', 'tiff'], logger)
    
    for name in image_names:
        if name not in im_filenames:
            logger.warning(f"Can't find {name} in {image_path}")
            image_names.remove(name)
    logger.info(f"Total {len(image_names)} images")
        
    # visualise the patch scores for each image
    for name in image_names:
        image = ut.import_images(image_path, [name])[0]
        create_visualisations_patch_scores(
            image,
            name,
            scores,
            save_path,
            logger,
            colour,
            normalise_scores,
            alpha,
            score_column
            )
    logger.info(f"Completed all visualisations, see {save_path}")


def create_visualisations_patch_scores(
    image,
    image_name,
    scores,
    save_path,
    logger,
    colour='bwr',
    normalise_scores='per_image',
    alpha=0.3,
    score_column='score'
    ):
    """Creates a single visualisation of patch scores on an image.
    The full image scores are created by averaging the scores
    for overlapping patches and alpha blending ontop of original image.

    Args:
        image (numpy array): full image as array
        image_name (string): image filename
        scores (pandas DataFrame): scores to overlay as colour layer
        save_path (string): location to save the visualisation as string
        logger (logging.logger object): logging
        colour (str, optional): colourscheme to plot scores. Defaults to 'bwr'.
        normalise_scores (str, optional): normalise scores options
            'per_image' normalises the scores per image
            'per_set' normalises the scores per image set contained in scores
            'no' doesn't normalise scores, colour max is max score per image
            'no_set' doesn't normalise scores, colour max is max score per
            image set. Defaults to 'per_image'.
        alpha (float, optional): transparency of colour layer. Defaults to 0.3.
        score_column (str): name of score column to visualise.
    """
    # check settings
    norm_score_settings = ['per_image', 'per_set', 'no', 'no_set']
    if normalise_scores not in norm_score_settings:
        logger.error(f"normalise_scores set incorrectly,\
         please choose one of {norm_score_settings}")
    if (alpha <= 0) or (alpha>=1):
        logger.error(f"alpha {alpha} not between (0,1)")

    if any(scores[score_column] < 0):
        positive_scores_only=False
    else:
        positive_scores_only=True

    #  if normalise_score is 'per_set'
    # to normalise the scores accross the whole image set
    # we need absolute min and max score
    if normalise_scores in ['per_set', 'no_set']:
        max_score = max(abs(scores[score_column]))
        min_score = min(abs(scores[score_column]))
        neg_min_score = min(scores[score_column])
        pos_max_score = max(0,max(scores[score_column]))

    # limit the scores to just the current image
    patch_scores = scores[scores['filename'] == image_name]
    
    # get full image shape and individual patch shape
    # checking consistent accross image
    im_shape_x = list(set(patch_scores['image_shape_x']))
    im_shape_y = list(set(patch_scores['image_shape_y']))
    patch_width = list(set(patch_scores['patch_width']))
    patch_height = list(set(patch_scores['patch_height']))
    if len(im_shape_x) + len(im_shape_y) > 2:
        logger.error("Error: inconsistent image shape")
    if len(patch_width)+len(patch_height) > 2:
        logger.error("Error: inconsistent patch shape")
    else:
        im_shape_x = im_shape_x[0]
        im_shape_y = im_shape_y[0]
        patch_width = patch_width[0]
        patch_height = patch_height[0]

    # initialise zero scores and zero counter array in full image shape
    image_scores = np.zeros((im_shape_x,im_shape_y),dtype=float)
    counter = np.zeros((im_shape_x,im_shape_y),dtype=int)
    # initialise a patch full of zeros in patch shape
    empty_patch = np.zeros((patch_width,patch_height),dtype=float)
    
    # loop over all patches, 
    # create an array of the score per pixel in the patch 
    # pad so that scores are in correct location in full image size
    # add the scores to the overall image scores
    # add a counter to every pixel where the score was non-zero
    
    for i in range(patch_scores.shape[0]):
        # get the score for the patch
        score = patch_scores[score_column].iloc[i]
        # get the row and column location of patch
        loc_row = int(patch_scores['coord_array_row'].iloc[i])          
        loc_col = int(patch_scores['coord_array_col'].iloc[i])
        # create a patch sized array full of the score
        score_array = empty_patch + score
        # create a patch sized array full of ones 
        counter_array = empty_patch + 1
        # work out how much image comes before and after the patch
        pre_patch_length = loc_row
        pre_patch_width = loc_col
        post_patch_length = im_shape_x - (loc_row+patch_height)
        post_patch_width = im_shape_y - (loc_col+patch_width)
        # create 2 full image sized arrays 
        # one that contains the score inside the patch and 0s surrounding the patch
        # and one that contains 1s inside the patch and 0s surrounding the patch
        score_array = np.pad(score_array,
                                ((pre_patch_length, post_patch_length), (pre_patch_width,post_patch_width)),
                                'constant',
                                constant_values=(0,0))
        counter_array = np.pad(counter_array,
                                ((pre_patch_length, post_patch_length), (pre_patch_width,post_patch_width)),
                                'constant',
                                constant_values=(0,0))
        # add the patches scores into the full image scores
        image_scores = image_scores + score_array
        # add the patch counters into the full image of patch counters
        counter = counter + counter_array
    
    # average scores from patches in shape of full original image
    image_scores = image_scores/counter
    # where dividing by 0s in counter creates NaN types, replace with zeros
    image_scores = np.nan_to_num(image_scores)
    
    # normalise scores and set histogram limits
    if normalise_scores == 'no':
        norm_image_scores = image_scores
        max_score = np.max(image_scores)
        upper_map_limit = ceil(max_score)
        if positive_scores_only:
            lower_map_limit = 0
        else:
            lower_map_limit = floor(np.min(image_scores))
    
    if normalise_scores == 'no_set':
        norm_image_scores = image_scores
        upper_map_limit = ceil(pos_max_score)
        if positive_scores_only:
            lower_map_limit = 0
        else:
            lower_map_limit = floor(neg_min_score)
    
    if normalise_scores == 'per_image':
        # need the max and min score for this image
        # to normalise scores per image
        max_score = np.max(abs(image_scores))
        min_score = np.min(abs(image_scores))
        # normalise the image scores
        norm_image_scores = (image_scores-min_score)/(max_score-min_score)
        upper_map_limit = 1
        if positive_scores_only:
            lower_map_limit = 0
        else:
            lower_map_limit = -1

    if normalise_scores == 'per_set':
        # max and min set earlier when all scores in dataframe
        # normalise the image scores
        norm_image_scores = (image_scores-min_score)/(max_score-min_score)
        upper_map_limit = 1
        if positive_scores_only:
            lower_map_limit = 0
        else:
            lower_map_limit = -1
    
    # test image read in matches shape of scores array
    if image.shape != norm_image_scores.shape:
        logger.error(f"shape error, image given has shape {image.shape}\
            the patch scores have final shape {norm_image_scores.shape}")

    # plot the normalised scores on top of the image
    # with the histogram of scores acting as a colourbar
    
    #histogram of scores mapped to colourmap
    step_size=(upper_map_limit-lower_map_limit)/100
    if upper_map_limit == 0:
        bins = np.arange(lower_map_limit, upper_map_limit+2*step_size, step_size) 
    else:
        bins = np.arange(lower_map_limit, upper_map_limit+step_size, step_size) 
    #bins = [j*0.1 for j in range(10*lower_map_limit,(upper_map_limit*10)+1)]

    counts,bin_edges = np.histogram(
        norm_image_scores[norm_image_scores!=0],
        bins=bins)
    patch_area = patch_width*patch_height
    counts = counts/patch_area
    bin_span = bin_edges.max()-bin_edges.min()
    bin_width = bin_edges[1]-bin_edges[0]
    colmap = plt.cm.get_cmap(colour)
    c = [colmap(((x-bin_edges.min())/bin_span)) for x in bin_edges]
    
    # rotate figure if not landscape
    x, y = image.shape
    if x > y:
        image = np.rot90(image)
        norm_image_scores = np.rot90(norm_image_scores)

    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=1, nrows=2, bottom=0.2,
                            height_ratios=[6, 1])
    fig.set_figwidth(15)
    fig.set_figheight(13)
    fig.tight_layout()
    rcParams.update({'font.size': 24})
    # add image
    ax0 = fig.add_subplot(spec[0])
    ax0.imshow(image, cmap='gray')
    ax0.imshow(
        norm_image_scores,
        cmap=colour,
        alpha=alpha,
        vmin=lower_map_limit,
        vmax=upper_map_limit)
    ax0.set_axis_off()

    # add histogram
    ax1 = fig.add_subplot(spec[1])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)        
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    tol=0.05 * bin_width
    ax1.set_xlim([bin_edges[0]-tol,bin_edges[-1]+5*tol])
    ax1 = plt.bar(
        bin_edges[:-1]+(bin_width/2),
        counts,
        color=c,
        width=bin_width,
        linewidth=1,
        edgecolor='k')
    # plt.title(f"Histogram of average patch scores (per patch area)")
    ytic = np.arange(0,int(max(counts)+1), step=2)
    plt.yticks(list(ytic))
    
    if normalise_scores == 'no':
        save_name = f"{image_name[:-4]}_{score_column}_{colour}.svg"
    else:
        date = dt.datetime.now().strftime("%Y_%m_%d_Time_%H_%M")
        save_name = f"{image_name[:-4]}_{score_column}_{colour}_{date}_{normalise_scores}.svg"
    plt.show()

    #  save visualisation of scores to save_path as svg
    plt.savefig(f"{save_path}{save_name}")
    plt.close()
    logger.info(f"Completed visualisation for {image_name}")        
    
if __name__ == "__main__":

    # create some test visualisations
    run_path = "D:/topological-bone-analysis/example/2021_08_25_Time_13_48/"
    image_path = run_path+"padded/"
    patch_path = run_path+"patches/"
    score_path = run_path+"all_statistics.csv"
    save_path =  run_path+"visualisations/"
    visualise_patch_scores(
        logger,
        image_path,
        patch_path,
        score_path,
        save_path,
        colour='hot',
        normalise_scores='per_image',
        positive_scores_only=False,
        alpha=0.4,
        score_column='score'
        )

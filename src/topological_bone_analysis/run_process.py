# -*- coding: utf-8 -*-

from os import listdir, rmdir, remove
from os.path import isfile, join, exists
import logging
import time
import datetime as dt
import pandas as pd
import numpy as np
from PIL import Image

# local imports
from . import utils as ut
from . import preprocessing_images as preprocess
from . import persistent_homology_SEDT as ph
from . import persistence_statistics_per_quadrant as stats
from . import plots
from . import SVM as svm

# setup logging
logger = logging.getLogger("run_porosity_analysis")
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(stream_formatter)
logger.addHandler(console)


def topological_porosity_analysis(
    path,
    logger,
    threshold_func,
    patch_shape=300,
    background_val = 0,
    stride=None,
    trim=True,
    single_file=None,
    split_radius=-2,
    save_persistence_diagrams=False,
    analysis_plots=False,
    classification=False,
    feature_cols=None,
    filenames_map=None,
    runs=100,
    strat_col=None,
    cross_val='stratkfold',
    param_grid_SVC = {'C': [1,2,3], 'kernel': ('rbf','linear')}
    ):
    """Takes a set of grayscale images, or a single file,
    thresholds to binary with threshold_func
    optionally trims the binary images of background_val
    pads them into exact multiples of patch_shape
    and then cuts them into square patches stride apart.
    Each binary patch is transformed with a Signed Euclidean Distance transform
    before the persistent homology is calculated using cubical complexes
    with the filtration built over sublevel sets of the SEDT transformed patch.
    The persistence birth, death intervals are saved out for 0 and 1 homology,
    and the persistence diagrams are made and optionally plotted.
    Summary statistics of these intervals are calculated per quadrant.
    split radius calculates the number of points in quadrant 2 of H_0
    in [-inf, split_radius] and (split_radius, 0].

    Args:
        path (string): location as a string of grayscale images as .tiff files
        logger (logging.Logger object): logs the process
        threshold_func (function): takes in a numpy array of a grayscale image
            and returns a binary image
        patch_shape (int, optional): pixel length and width of patches to take.
             Defaults to 300.
        background_val (int, optional): value [0,1] of image that is background
            after binary. Defaults to 0.
        stride (int, optional): number of pixels to move accross/down image 
            before taking next patch. Defaults to None.
        trim (bool, optional): whether to trim empty rows and columns before
             padding to take patches. Defaults to True.
        single_file (string, optional): filename if single file not all images
             in path location. Defaults to None.
        split_radius (int, optional): split radius calculates the number of 
            points in quadrant 2 of H_0 in [-inf, split_radius] and 
            (split_radius, 0]. Defaults to -2.
        save_persistence_diagrams (bool, optional): saves plots of persistence
             diagrams. Defaults to False.
        analysis_plots (bool, optional): If True saves box plots comparing groups 
            defined by filenames map. Defaults to False
        classification (bool, optional). If True, a SVC will be used to classify
            into groups (filenames_map) using the specified feature_cols, 
            with runs, strat_col, cross_val and param_grid_SVC specifying 
            parameters. Defaults to False.
        feature_cols (list, optional). If None, will use all statistics. 
            If list of string statistic names, will use specified. 
        filenames_map (dict, optional) dictionary with each image filename 
            (string) as keys, and a group name as values.
        runs (optional, int). Number of training runs for SVC. Defaults to 100. 
        strat_col (optional, string) column to stratify the kfold CV across.
        cross_val (optional, string) Defaults to 'stratkfold'
        param_grid_SVC (parameter grid to train SVC. Defaults to
            {'C': [1,2,3], 'kernel': ('rbf','linear')}

    Returns:
        pandas DataFrame: topological statistics per quadrant for .tiff image 
            files in path
    """
    start_time = time.time()
    date = dt.datetime.now().strftime("%Y_%m_%d_Time_%H_%M")
    logger.info(f"process started {date}")

    #define all paths    
    run_path = f"{path}{date}/"
    binary_path = f"{run_path}binary/"
    padded_path = f"{run_path}padded/"
    patch_path = f"{run_path}patches/"
    SEDT_path = f"{run_path}SEDT/"
    idig_path = f"{run_path}idiagrams/"
    interval_path = f"{run_path}persistence_intervals/"
    if save_persistence_diagrams:
        pd_path = f"{run_path}persistence_diagrams/"
    stats_path = f"{run_path}statistics/"
    if analysis_plots:
        plot_path = f"{run_path}plots/"
    # check or create all paths
    ut.directory(run_path)
    if threshold_func:
        ut.directory(binary_path)
    ut.directory(padded_path)
    ut.directory(patch_path)
    ut.directory(SEDT_path)
    ut.directory(idig_path)
    ut.directory(interval_path)
    if save_persistence_diagrams:
        ut.directory(pd_path)
    ut.directory(stats_path)
    if analysis_plots:
        ut.directory(plot_path)
    logger.info("directories created")

    logger.info(f"path to images: {path}")
    # if is modification for a single file from a folder
    # else gets all filenames from folder
    if single_file is not None:
        filenames = [single_file]
        logger.info(f"single image file: { filenames}")
    else:
        filenames = [file for file in listdir(path)
                        if isfile(join(path, file))]

    #check tif or tiff files only    
    filenames = ut.check_ext(filenames,['tif','tiff'],logger)
    logger.info(f"number of image files: {len(filenames)}")
    
    # read images from filenames
    images = ut.import_images(path,filenames)
    logger.info(f"Imported {len(images)} images")

    # threshold to binary
    if threshold_func:
        for k in range(len(images)):
            binary_im = threshold_func(images[k])
            # check images binary
            if not np.isin(binary_im,[0,1]).all():
                logger.error("image is not binary")
            else:
                Image.fromarray(binary_im).save(f"{binary_path}{filenames[k]}")
        
    logger.info(f"Completed threshold for {len(filenames)} images")

    binary_filenames = [file for file in listdir(binary_path)
                        if isfile(join(binary_path, file))]
    binary_filenames = ut.check_ext(binary_filenames,['tif','tiff'],logger)

    # initialise coordinate save file
    with open(f"{run_path}patch_coords.csv", "w") as outfile:
        outfile.write("filename,image_shape_x,image_shape_y,patch_number,"\
            +"patch_width,patch_height,coord_array_row,coord_array_col\n")
    for filename in binary_filenames:
        preprocess.image_to_patches(
            run_path,
            filename,
            logger,
            patch_shape,
            stride=stride,
            pad_val=0,
            percentage_background=1,
            background_val=background_val,
            trim_first=trim,
            edge_val=0)
    logger.info("Completed patching images")

     #read in patches from patch path, so filenames in correct order
    patch_filenames = [file for file in listdir(patch_path)
                        if isfile(join(patch_path, file))]

    patch_filenames = ut.check_ext(patch_filenames,['tif','tiff'],logger)

    # Signed Euclidean Distance Transform of each patch                
    for name in patch_filenames:
        patch = ut.import_images(patch_path, [name])[0]
        SEDT_patch = preprocess.SEDT(patch)
        Image.fromarray(SEDT_patch).save(f"{SEDT_path}{name}")

    SEDT_patch_filenames = [file for file in listdir(SEDT_path)
                            if isfile(join(SEDT_path, file))]
    SEDT_patch_filenames = ut.check_ext(
        SEDT_patch_filenames,
        ['tif','tiff'],
        logger
        )
    
    #creating idiagrams, calculates persistent homology on SEDT patches
    logger.info("starting persistence")

    for SEDT_patchname in SEDT_patch_filenames:
        SEDT_patch = ut.import_images(SEDT_path, [SEDT_patchname])[0]
        ph.peristent_homology_sublevel_cubic(
            SEDT_patch,
            SEDT_patchname,
            run_path,
            plot_persistence_diagrams=save_persistence_diagrams)

    logger.info("patches persistent homology calculated")
    
    #finding existing interval files
    intervals_to_find = [f"PD_dim_{dim}_{name[:-4]}.csv"\
                            for name in SEDT_patch_filenames for dim in [0,1]]
    interval_files = ut.find_files(
        intervals_to_find,
        interval_path,
        logger)

    for interval_name in interval_files:
        dim = int(interval_name[7])
        intervals = pd.read_csv(interval_path+interval_name,
                                names=["birth", "death"])
        stats.quadrant_statistics(
            intervals,
            dim,
            interval_name,
            stats_path,
            split_radius=split_radius)
    
    # combine all statistics files
    stats_df = ut.combine_stats_files(
        stats_path,
        run_path,
        "all_statistics.csv",
        logger
        )
    logger.info(f"Combined statistics files: {run_path}all_statistics.csv")
    # check single stats file exists
    if exists(f"{run_path}all_statistics.csv"):
        #  remove single stats files 
        stats_files = [file for file in listdir(stats_path)
                            if isfile(join(stats_path, file))]
        for name in stats_files:
            remove(stats_path+name)
        rmdir(stats_path)
        for name in SEDT_patch_filenames:
            remove(SEDT_path+name)
        rmdir(SEDT_path)
        for name in binary_filenames:
            remove(binary_path+name)
        rmdir(binary_path)

    else:
        logger.warn("Failed to combine statistics files")
    
    if analysis_plots:
        plots.analysis_plots(
            stats_df,
            filenames_map,
            plot_path,
            feature_cols)
        logger.info(f"Plots saved to {plot_path}")

    if classification:
        logger.info("Classification beginning")
        svm_results = svm.classification_one_v_one(
            stats_df,
            run_path,
            logger,
            feature_cols,
            filenames_map,
            runs=runs,
            strat_col=strat_col,
            cross_val=cross_val,
            param_grid_SVC = param_grid_SVC
        )
        
        return stats_df, svm_results

    logger.info("pipeline executed time(m): "
                +str(round((time.time()-start_time)/60, 2)))
    return stats_df
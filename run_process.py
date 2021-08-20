# -*- coding: utf-8 -*-

from os import listdir, rmdir, remove
from os.path import isfile, join, exists
import logging
import time
import datetime as dt
from importlib import reload
import pandas as pd
import numpy as np
from PIL import Image

# local imports
import utils as ut
import preprocessing_images as preprocess
import persistent_homology_SEDT as ph
import persistence_statistics_per_quadrant as stats
reload(ut)
reload(preprocess)
reload(ph)
reload(stats)

# setup logging
logger = logging.getLogger("run_porosity_analysis")
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(stream_formatter)
logger.addHandler(console)

def ph_pipeline_SEDT(path,
                     logger,
                     threshold_func,
                     patch_shape=300,
                     background_val = 0,
                     stride=150,
                     trim=True,
                     single_file=None,
                     conf=None,
                     save_persistence_diagrams=False,
                     birth_limits=None):
    """
    Takes a set of grayscale images,
    thresholds to binary with threshold_func
    pads them into exact multiples of patch_shape
    and then cuts them into patches stride apart.
    Each binary patch is transformed with a Signed Euclidean Distance transform
    before the persistent homology is calculated using cubical complexes
    with the filtration built over sublevel sets of the SEDT transformed patch.
    The barcode birth, death intervals are saved out for zero and 1 homology,
    and the persistence diagrams are made and saved.
    Summary statistics of these intervals are calculated per quadrant, 
    with an optional confidence band conf.
    The mode setting allows you to pick up part way through the process.

    Parameters:
    -----------
    path (STRING): location of image files
    logger (logging.Logger object)
    threshold_func ([type]): [description]
    patch_shape (list, optional): Shape of patches in pixels. Defaults to [300,300].
    background_val (int, optional): [description]. Defaults to 0.
    stride (int, optional): [description]. Defaults to 150.
    single_file ([type], optional): [description]. Defaults to None.
    conf ([type], optional): [description]. Defaults to None.
    mode (str, optional): [description]. Defaults to 'new'.
    date (STRING, optional) : specify a previous date as a run id for a mode other than new. 
    Do not include in path. Defaults to None.
    """
        
    start_time = time.time()
    date = dt.datetime.now().strftime("%Y_%m_%d_Time_%H_%M")
    logger.info(f"process started {date}")

    #define all paths    
    run_path = f"{path}{date}/"
    binary_path = f"{run_path}binary/"
    patch_path = f"{run_path}patches/"
    SEDT_path = f"{run_path}SEDT/"
    idig_path = f"{run_path}idiagrams/"
    interval_path = f"{run_path}persistence_intervals/"
    if save_persistence_diagrams:
        pd_path = f"{run_path}persistence_diagrams/"
    else:
        pd_path=None
    stats_path = f"{run_path}stats_birthlimits_{birth_limits}/"

    # check or create all paths
    ut.directory(run_path)
    if threshold_func:
        ut.directory(binary_path)
    ut.directory(patch_path)
    ut.directory(SEDT_path)
    ut.directory(idig_path)
    ut.directory(interval_path)
    if save_persistence_diagrams:
        ut.directory(pd_path)
    ut.directory(stats_path)
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
    filenames, dropped = ut.check_ext(filenames,['tif','tiff'])
    logger.info(f"number of image files: {len(filenames)}")
    if len(dropped) > 0:
        logger.warning(f"Files dropped incorrect extensions: {dropped}")
    
    # read images from filenames
    images = ut.import_images(
        path,
        filenames
        )
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
    binary_filenames, dropped = ut.check_ext(binary_filenames,['tif','tiff'])

    for filename in binary_filenames:
        preprocess.image_to_patches(
            binary_path,
            filename,
            patch_path,
            logger,
            patch_shape,
            stride,
            pad_val=0,
            percentage_background=1,
            background_val=background_val,
            trim_first=trim,
            edge_val=0)

     #read in patches from patch path, so filenames in correct order
    patch_filenames = [file for file in listdir(patch_path)
                        if isfile(join(patch_path, file))]
    patch_filenames,dropped = ut.check_ext(patch_filenames,['tif','tiff'])

    # Signed Euclidean Distance Transform of each patch                
    for name in patch_filenames:
#        patch = ut.import_images(name, )
        SEDT_patch = preprocess.SEDT(patch)
        Image.fromarray(SEDT_patch).save(f"{SEDT_path}{name}")

    
    SEDT_patch_filenames = [file for file in listdir(SEDT_path)
                            if isfile(join(SEDT_path, file))]
    SEDT_patch_filenames = ut.check_ext(SEDT_patch_filenames,['tif','tiff'])
    
    #creating idiagrams, calculates persistent homology on SEDT patches
    logger.info("starting persistence")

    for SEDT_patchname in SEDT_patch_filenames:
        SEDT_patch = ut.import_images([SEDT_patchname])
        ph.peristent_homology_sublevel_cubic(
            SEDT_patch,
            SEDT_patchname,
            run_path,
            plot_persistence_diagrams=save_persistence_diagrams)

    logger.info("patches persistent homology calculated")
    
    #finding existing interval files
    intervals_to_find = [f"dim_{dim}_intervals_{name[:-4]}.csv"\
                            for name in SEDT_patch_filenames for dim in [0,1] ]
    interval_files = ut.find_files(
        intervals_to_find,
        interval_path,
        logger)

    # check 2 files were found per filename, HERE

    for interval_name in interval_files:
        dim = int(interval_name[4])
        intervals = pd.read_csv(interval_path+interval_name,
                                names=["birth", "death"])
        stats.quadrant_statistics(
            intervals,
            dim,
            interval_name,
            stats_path,
            radius_split=-2)
    

    # combine all statistics files
    stats = ut.combine_stats_files(
        stats_path,
        run_path,
        "all_statistics.csv"
        )
    logger.info(f"Combined statistics files: {stats_path}all_statistics.csv")
    # check single stats file exists
    if exists(f"{stats_path}all_statistics.csv"):
        #  remove single stats files 
        stats_files = [file for file in listdir(stats_path)
                            if isfile(join(stats_path, file))]
        for name in stats_files:
            remove(stats_path+name)
        rmdir(stats_path)
    else:
        logger.warn("Failed to combine statistics files")
    
    logger.info("pipeline executed time(m): "
                +str(round((time.time()-start_time)/60, 2)))
    return stats


if __name__ == "__main__":
    path = "C:/Users/yp4g14/Documents/topological-bone-analysis/"
    stats = ph_pipeline_SEDT(path, logger, preprocess.otsu_threshold, patch_shape=300, stride=300, birth_limits=None, save_persistence_diagrams=False)

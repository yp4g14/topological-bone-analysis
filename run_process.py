# -*- coding: utf-8 -*-

from os import listdir, rmdir, remove
from os.path import isfile, join
import logging
import time
import datetime as dt
from importlib import reload
import pandas as pd
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

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
        tot_im = len(filenames)
        logger.info(f"number of image files: {tot_im}")

    #check tif or tiff files only    
    filenames, dropped = ut.check_ext(filenames,['tif','tiff'])
    if len(dropped) > 0:
        logger.warn(f"Files dropped incorrect extensions: {dropped}")
    
    # read images from filenames
    images = ut.import_images(
        path,
        filenames
        )
    logger.info(f"Imported {len(images)} images")

    binary_images = []
    if threshold_func:
        for k in range(len(images)):
            binary_im = threshold_func(images[k])
            binary_images.append(binary_im)
            Image.fromarray(binary_im).save(f"{binary_path}{filenames[k]}")
    logger.info(f"Completed threshold for {len(filenames)} images")

    images = binary_images
    # check images binary
    for image in images:
        assert np.isin(image,[0,1]).all(), 'image is not binary'

    for image in images:
        # trim image
        trimmed = preprocess.trim(image,edge_val=background_val)
        # take patches
        image, patches, patch_coords = preprocess.extract_patches(
            trimmed,
            patch_shape,
            pad_val=background_val,
            stride=stride
            )
    # HERE SAVE image, PATCHES, coords IF LESS percentage_background=1.0
        for patch in patches:


    #read in patches from patch path, so filenames in correct order
#        patch_path = binary_path # remove!
    patch_filenames = [file for file in listdir(patch_path)
                        if isfile(join(patch_path, file))]
    patch_filenames = utils.check_ext(patch_filenames, logger, ['tif','tiff'])
    # Signed Euclidean Distance Transform of each patch                
    utils.SEDT_images(patch_path,
                        SEDT_path,
                        logger,
                        x=4,
                        remove_filename=['params.txt',
                                        'patch_coords.csv'])
    
    SEDT_patch_filenames = [file for file in listdir(SEDT_path)
                            if isfile(join(SEDT_path, file))]
    
    SEDT_patches = utils.get_grey_images(SEDT_path,
                                            SEDT_patch_filenames,
                                            logger)

    #creating idiagrams, calculates persistent homology on SEDT patches
    logger.info("starting persistence")
    ph.grey_ph_sublevel_cubic(SEDT_path,
                                idig_path,
                                SEDT_patch_filenames,
                                SEDT_patches,
                                logger)
    logger.info("patches persistent homology calculated")
    
    SEDT_patch_filenames = [file for file in listdir(SEDT_path)
                            if isfile(join(SEDT_path, file))]
    #finding existing idiagram files
    idigs_to_find = [patch_name[:-4]+".idiagram" for patch_name in SEDT_patch_filenames]
    idig_list = utils.find_files(idigs_to_find, idig_path, logger)

    # get intervals
    ph.extract_pd_and_intervals(idig_path,
                                idig_list,
                                interval_path,
                                logger,
                                pd_path,
                                save_persistence_diagrams)
    logger.info("intervals complete")

    SEDT_patch_filenames = [file for file in listdir(SEDT_path)
                            if isfile(join(SEDT_path, file))]
    SEDT_patch_filenames = utils.check_ext(SEDT_patch_filenames,
                                            logger,
                                            ['tif','tiff'])
    #finding existing interval files
    intervals_to_find = [f"dim_{dim}_intervals_{name[:-4]}.csv"\
                            for name in SEDT_patch_filenames for dim in [0,1] ]
    interval_files = utils.find_files(intervals_to_find,
                                        interval_path,
                                        logger)
    if cleanup:
        # remove the SEDT of patches as images
        SEDT_files = [file for file in listdir(SEDT_path)
                            if isfile(join(SEDT_path, file))]
        for name in SEDT_files:
            remove(SEDT_path+name)
        rmdir(SEDT_path)
    
    # check 2 files were found per filename,
    # if not use single dim version and lookup dictionary single_dim
    single_dim = {}
    two_dim = []
    for file in SEDT_patch_filenames:
        interval_files_match = [s for s in interval_files if file[:-4] in s]
        if len(interval_files_match) == 2:
            two_dim.append(file)
        if len(interval_files_match) < 2:
            if len(interval_files_match) == 1:
                single_dim[file] = int(interval_files_match[0][4])
        
        if len(interval_files_match) > 2:
            SEDT_patch_filenames.remove(file)
            logger.warning(f"Will not compute statistics for {file[:-4]}\
                            as too many files found")

    for patch_name in two_dim:
        interval_files = [f"dim_0_intervals_{patch_name[:-4]}.csv",
                            f"dim_1_intervals_{patch_name[:-4]}.csv"]    
        if conf is not None:
            confidence_band=True
        else:
            confidence_band=False
        st_by_Q.quad_stats_from_intervals_csv(interval_path,
                                                interval_files,
                                                stats_path,
                                                patch_name,
                                                logger,
                                                x=4,
                                                birth_limits=birth_limits,
                                                confidence_band=confidence_band,
                                                conf_c=conf)

    for patch_name in list(single_dim.keys()):
        interval_files = [f"dim_{single_dim[patch_name]}_intervals_{patch_name[:-4]}.csv"]    

        st_by_Q.quad_stats_from_intervals_csv(interval_path,
                                                interval_files,
                                                stats_path,
                                                patch_name,
                                                logger,
                                                x=4,
                                                birth_limits=birth_limits,
                                                confidence_band=False,
                                                conf_c=conf,
                                                single_dim=single_dim[patch_name])
    stats = utils.combine_stats_files(stats_path,
                                        run_path,
                                        f"all_statistics_conf_{conf}.csv",
                                        logger)
        #  remove single stats files 
        stats_files = [file for file in listdir(stats_path)
                            if isfile(join(stats_path, file))]
        for name in stats_files:
            remove(stats_path+name)
        rmdir(stats_path)
    
    logger.info("pipeline executed time(m): "
                +str(round((time.time()-start_time)/60, 2)))
    return stats



if __name__ == "__main__":
    path = "C:/Users/yp4g14/Documents/PhDSToMI/PhDSToMI/data/TPF/"
    stats = ph_pipeline_SEDT(path, logger, preprocess.otsu_threshold, patch_shape=300, stride=300, birth_limits=None, save_persistence_diagrams=False)

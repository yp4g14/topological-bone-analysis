import numpy as np
import pandas as pd
from os import listdir, mkdir
from os.path import isfile, join, exists
from PIL import Image

def directory(path):
    """Creates a directory if doesn't exist

    Args:
        path (string): location to create
    """
    if not exists(path):
        mkdir(path)

def check_ext(filenames, extensions, logger):
    """Checks each filename in filenames list has an extension in extensions

    Args:
        filenames (list): list of filenames as strings
        extensions (list): list of suitable extensions

    Returns:
        list: filenames with correct extensions
    """
    ext_filenames = []
    for name in filenames:
        ext = name.split('.')[-1]
        if ext in extensions:
            ext_filenames.append(name)
    dropped = list(set(filenames) - set(ext_filenames))
    if len(dropped) > 0:
        logger.warning(f"Files dropped incorrect extensions: {dropped}")
    return ext_filenames

def import_images(path, filenames):
    """ Opens files from filenames list located in path directory.
    Converts to grayscale and adds them to list images.

    Args:
        path (stirng): location of image files
        filenames (list): list of image filenames as strings 
    """
    images=[]
    for i in range(len(filenames)):
        images.append(np.array(Image.open(path+filenames[i])))
    return images

def find_files(files_to_find, path, logger):
    """
    Function looks for file names in files_to_find list in the path directory.
    If a file is not found, updates the logger with a warning.
    Updates the logger with info on which files were found.

    Parameters
    ----------
    files_to_find : list of strings containing file names we are looking for.
    path : string of path diretory where files should be.
    logger : logger object.

    Returns
    -------
    found_files :  list of file names that exist in the directory.
    """
    
    existing_files = [file for file in listdir(path)
                      if isfile(join(path, file))]
    found_files = []
    files_not_found = []
    for file in files_to_find:
        if file in existing_files:
            found_files.append(file)
        else:
            files_not_found.append(file)
    if len(files_not_found) >1:
        logger.warning(f"Files not found: {files_not_found}")
    return found_files

def combine_stats_files(
    path,
    save_path,
    save_name,
    logger):
    """Combines statistics per image patch into a DataFrame of all statistics
    
    Args:
        path (string): location of stats files to combine
        save_path (string): 
        save_name (string): filename to save as csv

    Returns:
        pandas DatFrame: all statistics in a dataframe
    """
    files = [file for file in listdir(path) if isfile(join(path, file))]
    files = check_ext(files, ['csv'], logger)
    # split filenames
    filenames_h0 = [name for name in files if int(name[7])==0]
    filenames_h1 = [name for name in files if int(name[7])==1]
    dropped = [name for name in files if name not in (filenames_h0+filenames_h1)]
    if len(dropped)>0:
        logger.warning(f"Files not used: {dropped}")
    # combine all 0 dim intervals
    stats_h0 = []
    for name in filenames_h0:
        df = pd.read_csv(path+name, index_col=0)
        stats_h0.append(df)
    stats_h0 = pd.concat(stats_h0, axis=0).reset_index(drop=True)
    
    # combine all 1 dim intervals
    stats_h1 = []
    for name in filenames_h1:
        df = pd.read_csv(path+name, index_col=0)
        stats_h1.append(df)
    stats_h1 = pd.concat(stats_h1, axis=0).reset_index(drop=True)

    # join together
    common_cols = [col for col in stats_h0.columns if col in stats_h1.columns]\
        +[col for col in stats_h1.columns if col in stats_h0.columns]
    common_cols = list(set(common_cols))

    stats_df = stats_h0.merge(stats_h1, on=common_cols, how='outer')

    directory(save_path)
    stats_df.to_csv(f"{save_path}{save_name}")
    return stats_df

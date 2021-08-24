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
    stats = []
    columns = set()
    for name in files:
        df = pd.read_csv(path+name)
        stats.append(df)
        columns = columns|set(df.columns)
    # order the columns the same in all files before join
    for i in range(len(stats)):
        cols_present = set(stats[i].columns)
        cols_missing = columns - cols_present
        for col in list(cols_missing):
            stats[i][col] = None
        stats[i] = stats[i][list(columns)]
    stats = pd.concat(stats, ignore_index=True)
    try:
        stats = stats.drop("Unnamed: 0", axis=1)
        stats = stats.drop('name.1', axis=1)
    except:
        pass
    directory(save_path)
    stats.to_csv(f"{save_path}{save_name}")
    return stats


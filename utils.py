import numpy as np
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

def check_ext(filenames, extensions):
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
    return ext_filenames, dropped

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

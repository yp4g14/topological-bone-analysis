# -*- coding: utf-8 -*-
import numpy as np
import homcloud.interface as hc
import matplotlib.pyplot as plt

def peristent_homology_sublevel_cubic(image, save_path, filename):
    """Calculate sublevel set cubical homology persistence for an image (or patch)
    Input image should have undergone SEDT for SEDT transform filtration

    Args:
        image (numpy array): image values along which to take the sublevel filtration
        save_path (string): string of location to save the idiagram file
        filename (string): string of original image name
    """
    # calculate sublevel set cubical homology persistence for each image
    hc.PDList.from_bitmap_levelset(
        image,
        mode="sublevel",
        type="cubical",
        save_to=save_path+filename[:-4]+".idiagram")


def extract_pd_and_intervals(
    idig_path,
    idig_filename,
    interval_path,
    pd_path=None,
    save_persistence_diagrams=False):
    """
    Takes an idiagram file and pulls out the birth,death persistence intervals for two dimensions (0,1).
    Saves the persistence intervals as csv in interval_path
    Plots the persistence diagrams and saves as png in pd_path
    
    Parameters:
    -----------
    idig_path (string): location of .idiagram files
    idig_filename (string): filename for .idiagram 
    pd_path (string): location to save persistence diagrams, only if save_persistence_diagrams is True
    interval_path (string): location as string to save persistence diagram intervals
    logger (logging.Logger object)
    save_persistence_diagrams (bool, optional): if True will save the persistence diagram as png in pd_path. Default is False.
    """
    for dim in [0,1]:
        pd = hc.PDList(idig_path+idig_filename)
        pd = pd.dth_diagram(dim)
        # extract and save persistence intervals
        intervals = np.vstack([pd.births, pd.deaths]).transpose()
        ess_birth = list(pd.essential_births)
        for i in range(len(ess_birth)):
            intervals = np.vstack((intervals, [ess_birth[i],np.inf]))
        if intervals.shape[0] > 0:
            np.savetxt(f"{interval_path}dim_{dim}_intervals_{idig_filename[:-9]}.csv",
                    intervals,
                    delimiter=",")
            if save_persistence_diagrams:
                pd.histogram().plot(colorbar={"type": "log"})
                plt.savefig(f"{pd_path}dim_{dim}_{idig_filename[:-9]}.png")
                plt.close()

# -*- coding: utf-8 -*-
import numpy as np
import homcloud.interface as hc
import matplotlib.pyplot as plt
from os import mkdir
from os.path import exists

def peristent_homology_sublevel_cubic(
    image,
    filename,
    save_path,
    plot_persistence_diagrams=False):
    """Calculate sublevel set cubical homology persistence for an image
    Image should already have undergone SEDT for SEDT transform filtration
    Takes the idiagram file and pulls out the birth,death persistence intervals
    for two dimensions (0,1). Saves the persistence intervals as csv.
    Optionally plots the persistence diagrams and saves as svg.

    Args:
        image (numpy array): image values along which to take the filtration
        filename (string): filename for image as string
        save_path (string): location to save necessary files as string
        plot_persistence_diagrams (bool, optional): If True saves plots of the 
            persistence diagram. Defaults to False.
    """
    #initialise paths
    idiagram_path=save_path+'idiagrams/'
    idiagram_filename = filename[:-4]+".idiagram"
    interval_path = save_path+'persistence_intervals/'
    plot_path = save_path+'persistence_diagrams/'

    # calculate sublevel set cubical homology persistence for each image
    hc.PDList.from_bitmap_levelset(
        image,
        mode="sublevel",
        type="cubical",
        save_to=idiagram_path+idiagram_filename)

    # for dimensions 0 and 1, extract the births and deaths
    for dim in [0,1]:
        pd = hc.PDList(idiagram_path+idiagram_filename)
        pd = pd.dth_diagram(dim)

        # extract and save persistence intervals
        intervals = np.vstack([pd.births, pd.deaths]).transpose()
        ess_birth = list(pd.essential_births)
        for i in range(len(ess_birth)):
            intervals = np.vstack((intervals, [ess_birth[i],np.inf]))
        if intervals.shape[0] > 0:
            np.savetxt(
                f"{interval_path}PD_dim_{dim}_{idiagram_filename[:-9]}.csv",
                intervals,
                delimiter=",")

            # optional plot and save persistence diagrams
            if plot_persistence_diagrams:
                pd.histogram().plot(colorbar={"type": "log"})
                plt.savefig(
                    f"{plot_path}PD_dim_{dim}_{idiagram_filename[:-9]}.svg"
                    )
                plt.close()

        del pd

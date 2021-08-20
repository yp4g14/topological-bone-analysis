# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, exists

def distribution_stats_column(df, dim, colname):
    """Creates a dictionary summary of distribution statistics for a numeric
     column in a pandas dataframe.

    Args:
        df (pandas DataFrame): dataframe with column specified by "colname" 
            e.g. "birth"
        dim : [0,1] dimension of persistence diagram df belongs to.
        colname : numeric column name as string in df to be summarized with
             statistics.

    Returns:
        dictionary: statistics keyed by {dim}_{statistic}_{colname}
    """
    summary = {}
    summary[f"{dim}_avg_{colname}"] = df[colname].mean()
    summary[f"{dim}_max_{colname}"] = df[colname].max()
    summary[f"{dim}_min_{colname}"] = df[colname].min()
    summary[f"{dim}_stddev_{colname}"] = df[colname].std()
    summary[f"{dim}_skew_{colname}"] = df[colname].skew()
    summary[f"{dim}_kurtosis_{colname}"] = df[colname].kurtosis()
    summary[f"{dim}_percentile_25_{colname}"] = df[colname].quantile(q=0.25)
    summary[f"{dim}_percentile_75_{colname}"] = df[colname].quantile(q=0.75)
    summary[f"{dim}_iqr_{colname}"] = \
         summary[f"{dim}_percentile_75_{colname}"]\
         - summary[f"{dim}_percentile_25_{colname}"]
    
    for perc in [round(0.1*i,2) for i in range(1,10)]:
        summary[f"{dim}_percentile_{round(100*perc)}_{colname}"] =\
            df[colname].quantile(perc)
    return summary

def quadrant_statistics(
    intervals,
    dim,
    filename,
    save_path,
    radius_split=-2):
    """Takes in an array of birth, death intervals and calculates statistics
    per quadrant (1,2,3 as 4 naturally empty) for each dimension in dim.

    Args:
        intervals (numpy array): persistence intervals as array 
            with two columns 'birth', 'death'
        dim (int): persistence diagram for dimension in [0,1]
        filename (string): [description]
        radius_split (int, optional): (for quadrant 2 dim 0 only)
             will calculate the number of births less than value and
             number of births greater than or equal to value.
            Defaults to -2.
    Returns:
        pandas DataFrame: topological statistics calculated per quadrant.
    """

    # split into quadrants:
    Q1 = intervals.loc[intervals["birth"] >= 0]
    Q2 = intervals.loc[(intervals["birth"] < 0) & (intervals["death"] >= 0)]
    Q3 = intervals.loc[(intervals["birth"] < 0) & (intervals["death"] < 0)]

    intervals_quadrants = [Q1,Q2,Q3]
    stats_list = []
    # for each quadrant calculate the statistics in a dictionary
    for i in range(3):
        quadrant = intervals_quadrants[i]
        stats = {}
        stats["name"] = filename
        stats["quadrant"] = i+1
        if quadrant.shape[0] > 0:
            # number of points
            stats[f"{dim}_num_points"] = quadrant.shape[0]

            # birth statistics
            stats[f"{dim}_avg_birth"] =\
                 quadrant["birth"].sum()/quadrant.shape[0]
            birth_stats = distribution_stats_column(
                quadrant,
                dim,
                "birth"
                )
            stats.update(birth_stats)

            # remove inf death
            finite_quadrant = quadrant.copy(deep=True)
            finite_quadrant = finite_quadrant.loc[
                finite_quadrant["death"]!=np.inf
                ]
            # death statistics
            stats[f"{dim}_avg_death"] =\
                 finite_quadrant["death"].sum()/finite_quadrant.shape[0]

            death_stats = distribution_stats_column(
                finite_quadrant,
                dim,
                "death"
                )
            stats.update(death_stats)
            
            #lifetime = death - birth
            finite_quadrant["lifetime"] =\
                 finite_quadrant["death"] - finite_quadrant["birth"]

            #total persistance is the sum of all lifetimes
            stats[f"{dim}_total_persistence"] =\
                 finite_quadrant.sum()["lifetime"]

            # normalized_lifespan = (death - birth)/ sum_all(death-birth)
            finite_quadrant["normalized_lifespan"] =\
                 finite_quadrant["lifetime"]/stats[f"{dim}_total_persistence"]

            #let p be the normalised lifespan - persistent entropy 
            # can be viewed as the diversity of lifespans
            finite_quadrant["plogp"] = finite_quadrant["normalized_lifespan"]*\
                np.log(finite_quadrant["normalized_lifespan"])
            stats[f"{dim}_pers_entropy"] = - finite_quadrant["plogp"].sum()

            #number of births in quadrant 2 dim 0, 
            # less than equal to radius split and greater than radius split
            if (i+1 == 2) and (dim==0):
                stats[f"{dim}_num_points_less_eq_{radius_split}"] =\
                     quadrant.loc[quadrant["birth"]<=radius_split].shape[0]

                stats[f"{dim}_num_points_greater_{radius_split}"] =\
                     quadrant.loc[quadrant["birth"]>radius_split].shape[0]

            #convert to DataFrame
            stats = pd.DataFrame([stats])
            stats_list.append(stats)

    # combine all stats into a single DataFrame
    if len(stats_list) != 0:
        stats_df = pd.concat(stats_list).reset_index(drop=True)
        stats_df.to_csv(
            f"{save_path}{filename[:-4]}_statistics.csv")

def combine_stats_files(
    path,
    save_path,
    save_name):
    """Combines statistics per image patch into a DataFrame of all statistics
    
    Args:
        path (string): location of stats files to combine
        save_path (string): 
        save_name (string): filename to save as csv

    Returns:
        pandas DatFrame: all statistics in a dataframe
    """
    files = [file for file in listdir(path) if isfile(join(path, file))]
    # files = check_ext(files, ["csv"])
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
        stats = stats.drop("name.1", axis=1)
    except:
        pass
    if not exists(path):
        mkdir(path)
    stats.to_csv(f"{save_path}{save_name}")
    return stats

# def quad_stats_from_intervals_csv(interval_path,
#                                   interval_files,
#                                   stats_path,
#                                   image_filename,
#                                   logger,
#                                   x,
#                                   birth_limits=None,
#                                   confidence_band=False,
#                                   conf_c=None,
#                                   single_dim=None):
#     """
#     Takes paths to interval files of persistence diagrams 
#     for an image with name image_name and outputs the summary statistics
#     saving them in stats_path.
#     Can optionally apply a confidence band death > birth + 2*conf_c 
#     Usually takes 2 files for dim 0, dim 1 intervals, can be used for a single dim. 
    
#     Parameters
#     ----------
#     interval_path (STRING) :
#         path to interval files 
#     interval_files (LIST): 
#         list containing 2 interval filenames for dim 0 then dim 1 (in that order)
#     stats_path (STRING): 
#         location to save the statistics calculated
#     image_filename (STRING):
#         string of filename which the intervals correspond to 
#     logger (logging.Logger): 
#     x (int): 
#         length of file extension to remove from name
#     birth_limits (default None):
#         restrict to intervals with a birth value in this range
#     confidence_band (bool, optional): 
#         optionally adds a confidence band. Defaults to False.
#     conf_c (INT, optional): 
#         width of confidence_band when confidence_band is True, otherwise ignored. Defaults to None.
#     single_dim (INT, optional): 
#         If calculating the statistics for just dim 0, set to 0. Defaults to None.

#     Returns
#     -------
#     stats : pandas DataFrame containing statistics per quadrant
#     """

#     # if calculating quadrant stats for only one dimension of persistence diagram 
#     if single_dim is not None:
        
#         #read in interval csv files
#         intervals = pd.read_csv(interval_path+interval_files[0],
#                                 names=["birth", "death"])
        
#         if confidence_band == True:
#             intervals = intervals.loc[intervals["death"] > intervals["birth"]+2*conf_c]
#         if birth_limits is not None:
#             birth_limits.sort()
#             intervals = intervals.loc[intervals["birth"] >= birth_limits[0]]
#             intervals = intervals.loc[intervals["birth"] <= birth_limits[1]]
#             if intervals.shape[0] == 0:
#                 logger.warning(f"No stats in birth limits {birth_limits}")
#                 pass
#         stats = pd.DataFrame()
#         if intervals.shape[0] != 0:
#             stats_intervals = quadrant_statistics(intervals,
#                                                 single_dim,
#                                                 image_filename[:-x],
#                                                 logger)
            
#             stats_intervals.to_csv(f"{stats_path}{image_filename[:-x]}_conf_{conf_c}_statistics.csv")
#         else:
#             stats_intervals = None
#         return stats_intervals
        
#     else:
#         #read in interval csv files
#         intervals_0 = pd.read_csv(interval_path+interval_files[0],
#                                   names=["birth", "death"])
#         intervals_1 = pd.read_csv(interval_path+interval_files[1],
#                                   names=["birth", "death"])
    
#         if confidence_band == True:
#             intervals_0 = intervals_0.loc[intervals_0["death"] > intervals_0["birth"]+2*conf_c]
#             intervals_1 = intervals_1.loc[intervals_1["death"] > intervals_1["birth"]+2*conf_c]
#         if birth_limits is not None:
#             birth_limits.sort()
#             intervals_0 = intervals_0.loc[intervals_0["birth"] >= birth_limits[0]]
#             intervals_0 = intervals_0.loc[intervals_0["birth"] <= birth_limits[1]]
#             intervals_1 = intervals_1.loc[intervals_1["birth"] >= birth_limits[0]]
#             intervals_1 = intervals_1.loc[intervals_1["birth"] <= birth_limits[1]]
#         if intervals_0.shape[0] == 0:
#             if intervals_1.shape[0] == 0:
#                 logger.warning(f"No stats in birth limits {birth_limits}")
#                 pass

#         stats = pd.DataFrame()
#         if intervals_0.shape[0] != 0:
#             stats_intervals_0 = quadrant_statistics(intervals_0,
#                                                     0,
#                                                     image_filename[:-x],
#                                                     logger)
#         else:
#             stats_intervals_0 = None
#         if intervals_1.shape[1] != 0:
#             stats_intervals_1 = quadrant_statistics(intervals_1,
#                                                     1,
#                                                     image_filename[:-x],
#                                                     logger)
#         else:
#             stats_intervals_1 = None

#         if stats_intervals_0 is not None:
#             if stats_intervals_1 is not None:
#                 stats = stats_intervals_0.merge(stats_intervals_1,
#                                                 how="outer",
#                                                 on=["name","quadrant"])
#             else:
#                 stats = stats_intervals_0
#         elif stats_intervals_1 is not None:
#             stats = stats_intervals_1
#         if stats is not None:
#             stats.to_csv(f"{stats_path}{image_filename[:-x]}_birthlimits_{birth_limits}_conf_{conf_c}_statistics.csv")
#         return stats

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
    split_radius=-2):
    """Takes in an array of birth, death intervals and calculates statistics
    per quadrant (1,2,3 as 4 naturally empty) for each dimension in dim.

    Args:
        intervals (numpy array): persistence intervals as array 
            with two columns 'birth', 'death'
        dim (int): persistence diagram for dimension in [0,1]
        filename (string): [description]
        split_radius (int, optional): (for quadrant 2 dim 0 only)
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
        stats_dict = {}
        stats_dict["filename"] = filename
        stats_dict["patch_number"] = filename[:-4]
        stats_dict["quadrant"] = i+1
        if quadrant.shape[0] > 0:
            # number of points
            stats_dict[f"{dim}_num_points"] = quadrant.shape[0]

            # birth statistics
            stats_dict[f"{dim}_avg_birth"] =\
                 quadrant["birth"].sum()/quadrant.shape[0]
            birth_stats = distribution_stats_column(
                quadrant,
                dim,
                "birth"
                )
            stats_dict.update(birth_stats)

            # remove inf death
            finite_quadrant = quadrant.copy(deep=True)
            finite_quadrant = finite_quadrant.loc[
                finite_quadrant["death"]!=np.inf
                ]
            if finite_quadrant.shape[0] > 0:
                # death statistics
                stats_dict[f"{dim}_avg_death"] =\
                    finite_quadrant["death"].sum()/(finite_quadrant.shape[0])

                death_stats = distribution_stats_column(
                    finite_quadrant,
                    dim,
                    "death"
                    )
                stats_dict.update(death_stats)
                
                #lifetime = death - birth
                finite_quadrant["lifetime"] =\
                    finite_quadrant["death"] - finite_quadrant["birth"]

                #total persistance is the sum of all lifetimes
                stats_dict[f"{dim}_total_persistence"] =\
                    finite_quadrant.sum()["lifetime"]

                # normalized_lifespan = (death - birth)/ sum_all(death-birth)
                finite_quadrant["normalized_lifespan"] =\
                    finite_quadrant["lifetime"]/stats_dict[f"{dim}_total_persistence"]

                #let p be the normalised lifespan - persistent entropy 
                # can be viewed as the diversity of lifespans
                finite_quadrant["plogp"] = finite_quadrant["normalized_lifespan"]*\
                    np.log(finite_quadrant["normalized_lifespan"])
                stats_dict[f"{dim}_pers_entropy"] = - finite_quadrant["plogp"].sum()

            #number of births in quadrant 2 dim 0, 
            # less than equal to radius split and greater than radius split
            if (i+1 == 2) and (dim==0):
                stats_dict[f"{dim}_num_points_less_eq_{split_radius}"] =\
                     quadrant.loc[quadrant["birth"]<=split_radius].shape[0]

                stats_dict[f"{dim}_num_points_greater_{split_radius}"] =\
                     quadrant.loc[quadrant["birth"]>split_radius].shape[0]

            #convert to DataFrame
            stats_dict = pd.DataFrame([stats_dict])
            stats_list.append(stats_dict)

    # combine all stats into a single DataFrame
    if len(stats_list) != 0:
        stats_df = pd.concat(stats_list).reset_index(drop=True)
        stats_df.to_csv(
            f"{save_path}{filename[:-4]}_statistics.csv")

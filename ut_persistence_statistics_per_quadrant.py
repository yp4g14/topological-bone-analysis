import numpy as np
import pandas as pd
from importlib import reload
from os import rmdir, remove, listdir
from os.path import exists, isfile
from math import sqrt
import persistence_statistics_per_quadrant as stats
reload(stats)
from scipy.stats import skew, kurtosis
# TEST - quadrant_statistics
######################################################
test_intervals = np.array([
    [0, 0],
    [0, 4],
    [2, 4],
    [4, 6],
    [-3, 3],
    [-2, 4],
    [-1, 0],
    [-5, -1],
    [-4, -2]])

test_intervals = pd.DataFrame(
    test_intervals,
    columns=["birth","death"]
    )
test_intervals['birth'].skew()
filename = "test.tif"
save_path = "C:/test_"
quadrants = [1,2,3]
num_points = [4,3,2]
avg_birth = [6/4,-2,-9/2]
avg_death = [14/4,7/3,-3/2]
max_birth = [4,-1,-4]
min_birth = [0,-3,-5]
stdev_birth = [sqrt(11/3),1.,sqrt(0.5)]
skew_birth = [
    skew(np.array([0,0,2,4]),bias=False),
    skew(np.array([-3,-2,-1]),bias=False),
    skew(np.array([-5,-4]),bias=False)
    ]
kurt_birth = [
    kurtosis(np.array([0,0,2,4]),bias=False),
    kurtosis(np.array([-3,-2,-1]),bias=False),
    kurtosis(np.array([-5,-4]),bias=False)
    ]
percentile_list = [10*q for q in list(range(1,10))+[2.5,7.5]]
birth_percentiles_q1 = np.percentile(
    np.array([0,0,2,4]),
    percentile_list
    )
birth_percentiles_q2 = np.percentile(
    np.array([-3,-2,-1]),
    percentile_list
    )
birth_percentiles_q3 = np.percentile(
    np.array([-5,-4]),
    percentile_list
    )
birth_percentiles = np.array([
    birth_percentiles_q1,
    birth_percentiles_q2,
    birth_percentiles_q3]
    )
birth_percentiles = pd.DataFrame(
    birth_percentiles,
    columns=[
        '0_percentile_10_birth',
        '0_percentile_20_birth',
        '0_percentile_30_birth',
        '0_percentile_40_birth',
        '0_percentile_50_birth',
        '0_percentile_60_birth',
        '0_percentile_70_birth',
        '0_percentile_80_birth',
        '0_percentile_90_birth',
        '0_percentile_25_birth',
        '0_percentile_75_birth']
        )
birth_percentiles['0_iqr_birth'] =\
     birth_percentiles['0_percentile_75_birth']\
          - birth_percentiles['0_percentile_25_birth']

max_death = [6,4,-1]
min_death = [0,0,-2]

test_stats = pd.DataFrame(
    np.array([
        quadrants,
        num_points,
        avg_birth,
        avg_death,
        max_birth,
        min_birth,
        stdev_birth]).T,
    columns=[
        'quadrant',
        '0_num_points',
        '0_avg_birth',
        '0_avg_death',
        '0_max_birth',
        '0_min_birth',
        '0_stddev_birth']
        )

test_stats['filename'] = 'test.tif'
stat_df = stats.quadrant_statistics(test_intervals, 0,filename,radius_split=-2)


# check shape of stats

#check for NaNs

# TEST - distribution_stats_column
######################################################


# TEST - combine_stats_files
######################################################


# # TEST - quad_stats_from_intervals_csv
# ######################################################
# # check it creates necessary paths
# assert exists(save_path), "save_path does not exist for statistics"

# # check it creates necessary files
# assert isfile(save_path+"statistics.csv"), "stats file does not exist"

# # remove test files from system
# remove(save_path+"statistics.csv")
# rmdir(save_path+"idiagrams/")

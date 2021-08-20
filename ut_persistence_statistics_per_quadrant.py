import numpy as np
import pandas as pd
from importlib import reload
from os import rmdir, remove, listdir
from os.path import exists, isfile
from math import sqrt
import persistence_statistics_per_quadrant as stats
reload(stats)
from scipy.stats import skew, kurtosis, entropy
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

quad_births = [np.array([0,0,2,4]),np.array([-3,-2,-1]),np.array([-5,-4])]
quad_deaths = [np.array([0,4,4,6]),np.array([3,4,0]),np.array([-1,-2])]

#test answers
quadrants = [1,2,3]

num_points = [4,3,2]

total_persistence = [8,13,6]
points_less_eq_2 = [np.nan,2,np.nan]
points_greater_2 = [np.nan,1,np.nan]

persistence = [quad_deaths[i]-quad_births[i] for i in range(3)]
pers_entropy = [entropy(lifetimes) for lifetimes in persistence]

avg_birth = [6/4,-2,-9/2]
avg_death = [14/4,7/3,-3/2]

max_birth = [4,-1,-4]
max_death = [6,4,-1]

min_birth = [0,-3,-5]
min_death = [0,0,-2]

stdev_birth = [pd.DataFrame(births, columns=['birth'])['birth'].std()\
     for births in quad_births]
stdev_death = [pd.DataFrame(deaths, columns=['death'])['death'].std()\
     for deaths in quad_deaths]

skew_birth = [skew(births,bias=False) for births in quad_births]
skew_death = [skew(deaths,bias=False) for deaths in quad_deaths]

kurtosis_birth = [kurtosis(births,bias=False) for births in quad_births]

kurtosis_death = [kurtosis(deaths,bias=False) for deaths in quad_deaths]

def calc_percentiles(percentile_list, np_arr,column):
    percentiles_q1 = np.percentile(np_arr[0], percentile_list)
    percentiles_q2 = np.percentile(np_arr[1], percentile_list)
    percentiles_q3 = np.percentile(np_arr[2], percentile_list)

    percentiles = np.array([
        percentiles_q1,
        percentiles_q2,
        percentiles_q3]
        )
    percentiles = pd.DataFrame(
        percentiles,
        columns=[f"0_percentile_{int(i)}_{column}" for i in percentile_list]
            )
    percentiles[f"0_iqr_{column}"] =\
        percentiles[f"0_percentile_75_{column}"]\
            - percentiles[f"0_percentile_25_{column}"]
    return percentiles

percentile_list = [10*q for q in list(range(1,10))+[2.5,7.5]]

birth_percentiles = calc_percentiles(percentile_list, quad_births,'birth')
death_percentiles = calc_percentiles(percentile_list, quad_deaths,'death')

test_stats = pd.DataFrame(
    np.array([
        quadrants,
        num_points,
        total_persistence,
        avg_birth,
        avg_death,
        max_birth,
        min_birth,
        max_death,
        min_death,
        stdev_birth,
        stdev_death,
        skew_birth,
        skew_death,
        kurtosis_birth,
        kurtosis_death,
        points_less_eq_2,
        points_greater_2,
        pers_entropy
        ]).T,
    columns=[
        'quadrant',
        '0_num_points',
        '0_total_persistence',
        '0_avg_birth',
        '0_avg_death',
        '0_max_birth',
        '0_min_birth',
        '0_max_death',
        '0_min_death',
        '0_stddev_birth',
        '0_stddev_death',
        '0_skew_birth',
        '0_skew_death',
        '0_kurtosis_birth',
        '0_kurtosis_death',
        '0_num_points_less_eq_-2',
        '0_num_points_greater_-2',
        '0_pers_entropy'
        ]
        )
test_stats['name'] = 'test.tif'
test_stats['quadrant'] = test_stats['quadrant'].apply(lambda s: int(s))
test_stats = pd.concat(
    [test_stats,birth_percentiles,death_percentiles],
    axis=1
    )


stat_df = stats.quadrant_statistics(test_intervals, 0,filename,radius_split=-2)


# check shape of stats
assert stat_df.shape == test_stats.shape,\
    "stats dataframe incorrect shape"
for column in test_stats.columns:
    if column == '0_num_points_less_eq_-2':
        # assert (test_stats.dropna()[column] == stat_df.dropna()[column]).all(),\
        #     f"{column} doesn't match"
    elif column == '0_num_points_greater_-2':
        # assert (test_stats.dropna()[column] == stat_df.dropna()[column]).all(),\
        #     f"{column} doesn't match"
    elif column not in ['0_skew_birth','0_skew_death','0_kurtosis_birth','0_kurtosis_death']:
        assert (test_stats[column] == stat_df[column]).all(),\
            f"{column} doesn't match"

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

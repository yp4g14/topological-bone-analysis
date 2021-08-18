import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
import seaborn as sns
import mice

TPF_path = "C:/Users/yp4g14/Documents/PhDSToMI/PhDSToMI/data/TPF/2021_01_21_Time_15_34_no_overlap/persistence_intervals/"
filenames = [file for file in listdir(TPF_path) if isfile(join(TPF_path, file))]
intervals = []
for file in filenames:
    df = pd.read_csv(TPF_path+file, header=None, names=['birth','death'])
    df['filename'] = file[:-4]
    intervals.append(df)


int_df = pd.concat(intervals, axis=0)
quad_2 = int_df.loc[int_df['birth']<=0]
quad_2 = quad_2.loc[quad_2['death']>=0]
min = min(quad_2['birth'])



    keys, ko_truth, male_truth = mice.mice()
    ko = {True: 'ko', False: 'wt'}
    male = {True:'male', False:'female'}    
    
    stats = pd.read_csv(stats_path)
    stats = stats.drop('Unnamed: 0', axis=1)
    stats['tag'] = stats['name'].apply(lambda s: s.split('_'))
    stats['tag'] = stats['tag'].apply(lambda s: [i for i in s if i in keys][0])
    stats = stats.drop('name', axis=1)
    
    stats['ko_truth'] = stats['tag'].map(ko_truth)
    stats['male_truth'] = stats['tag'].map(male_truth)
    stats['type'] = stats['ko_truth'].map(ko)
    stats['sex'] = stats['male_truth'].map(male)
    stats['cat'] = stats['ko_truth'].astype(int) + 2*stats['male_truth'].astype(int)
    cat = {0: "wt, female", 1: "ko, female", 2: "wt, male", 3: "ko, male"}
    stats['cat_map'] = stats['cat'].map(cat)
    


sns.histplot(data=quad_2, x='birth', bins=100, binrange=[-5,0])
sns.histplot(data=quad_2, x='birth', bins=100, binrange=[-10,-5])
sns.histplot(data=quad_2, x='birth', bins=100, binrange=[min,-10])



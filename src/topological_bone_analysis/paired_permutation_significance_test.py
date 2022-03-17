# paired permutation significance testing

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt
import logging 
from itertools import combinations, product
from statsmodels.stats.multitest import multipletests


logger = logging.getLogger("significance")
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(stream_formatter)
logger.addHandler(console)
from matplotlib import rcParams
rcParams.update({'font.size': 12})

def paired_permutation_test(
    feat_df,
    feature_pairs,
    number_permutations=10000,
    significance=0.05,
    path=None, 
    title=None,
    plot_significant=False):

    feat_col1, feat_col2 = feature_pairs[0], feature_pairs[1]

    feat_df['difference'] = feat_df[feat_col1]-feat_df[feat_col2]
    num_samples = feat_df['difference'].count()
    # difference in features in true split
    true_diff = feat_df['difference'].mean()/feat_df['difference'].std()

    #Copy the all features list
    pool_sample = copy.copy(feat_df)
    
    #Initialize permutation distribution
    pool_distribution = []

    # permutation loop, calculates distribution
    for i in range(0,number_permutations):
        # generate random number in [0,1)        
        pool_sample['chance'] = np.array([random.random() for i in range(num_samples)])
        # if random number is >= 0.5, we want to swap the stats by multiplying the difference by -1
        pool_sample['swap'] = (pool_sample['chance']>=0.5).map({True:-1,False:1})
        pool_sample['shuffled_diff'] = pool_sample['difference']*pool_sample['swap']

        # calculate the new average shuffled difference (normalised by std dev)
        avg_shuffled_difference = pool_sample['shuffled_diff'].mean()/pool_sample['shuffled_diff'].std()
        # append to distribution
        pool_distribution.append(avg_shuffled_difference)

    # 2 tails
    tail_1 = len(np.where(pool_distribution > np.abs(true_diff))[0])
    tail_2 =  len(np.where(pool_distribution < -np.abs(true_diff))[0])
    p_val = (tail_1+tail_2)/number_permutations

    if plot_significant:
        if p_val <= significance:
            sns.histplot(pool_distribution, stat='probability', linewidth=0)
            plt.xlabel('Permuted difference')
            plt.ylabel('Proportion')
            ylims = plt.ylim()
            plt.vlines(
                true_diff,
                ymin=0,
                ymax=1,
                color='orange',
                label=str(true_diff)
                )
            plt.ylim(ylims)
            if title:
                plt.title(f"{title}")
            if path:
                plt.savefig(f"{path}permutation_hypothesis_tests/{title}.png")
                plt.tight_layout()
            plt.close()
    return p_val

if __name__=="__main__":
    TPF_stat_path = "D:/data/TPF/2021_01_21_Time_15_34_no_overlap/all_statistics_conf_None.csv"
    SHG_stat_path = "D:/data/SHG/2021_01_21_Time_15_12_no_overlap/all_statistics_conf_None.csv"
    TPF_custom_stat_path = "D:/data/TPF/2021_01_21_Time_15_34_no_overlap/number_pores_size.csv"
    SHG_custom_stat_path = "D:/data/SHG/2021_01_21_Time_15_12_no_overlap/number_pores_size.csv"
    TPF_stat_df = pd.read_csv(TPF_stat_path,index_col=0)
    SHG_stat_df = pd.read_csv(SHG_stat_path,index_col=0)
    TPF_custom_stat_df = pd.read_csv(TPF_custom_stat_path,index_col=0)
    SHG_custom_stat_df = pd.read_csv(SHG_custom_stat_path,index_col=0)

    tags = ['AE5.7', 'AE6.7','AF6','AJ1','AJ3','AO1','AA7.7','AB1','AB6.7','AF5','AO2','AA6']
    TPF_stat_df['tag'] = TPF_stat_df['name'].apply(lambda x: [i for i in x.split('_') if i in tags][0])
    SHG_stat_df['tag'] = SHG_stat_df['name'].apply(lambda x: [i for i in x.split('_') if i in tags][0])
    TPF_stat_df['patch_number'] = TPF_stat_df['name'].apply(lambda x: int(x.split('_')[-2]))
    SHG_stat_df['patch_number'] = SHG_stat_df['name'].apply(lambda x: int(x.split('_')[-2]))

    TPF_custom_stat_df['tag'] = TPF_custom_stat_df['name'].apply(lambda x: [i for i in x.split('_') if i in tags][0])
    SHG_custom_stat_df['tag'] = SHG_custom_stat_df['name'].apply(lambda x: [i for i in x.split('_') if i in tags][0])
    TPF_custom_stat_df['patch_number'] = TPF_custom_stat_df['name'].apply(lambda x: int(x.split('_')[-2]))
    SHG_custom_stat_df['patch_number'] = SHG_custom_stat_df['name'].apply(lambda x: int(x.split('_')[-2]))
    TPF_custom_stat_df = TPF_custom_stat_df.drop('name', axis=1)
    SHG_custom_stat_df = SHG_custom_stat_df.drop('name', axis=1)

    # merge on custom stats
    TPF_stat_df = TPF_stat_df.merge(TPF_custom_stat_df, on=['tag', 'patch_number','quadrant'], how='left')
    SHG_stat_df = SHG_stat_df.merge(SHG_custom_stat_df, on=['tag', 'patch_number','quadrant'], how='left')

    common_cols = ['tag', 'patch_number', 'quadrant']
    useful_cols = [
        '_num_points',
        '_avg_birth',
        '_avg_death',
        '_pers_entropy',
        '_q25_birth',
        '_q50_birth',
        '_iqr_birth',
        '_stddev_birth',
        '_q25_death',
        '_q50_death',
        '_q75_death',
        '_iqr_death',
        '_stddev_death'
        ]
    custom_cols = ['number_big_pores','number_small_pores']
    keys = ['0'+i for i in useful_cols]+['1'+i for i in useful_cols]+custom_cols
    # drop all columns not in keys or common_cols (used for joins)
    TPF_stat_df = TPF_stat_df[keys+common_cols]
    SHG_stat_df = SHG_stat_df[keys+common_cols]
    TPF_vals = ["TPF_"+col for col in keys]
    TPF_dict = dict(zip(keys,TPF_vals))
    SHG_vals = ["SHG_"+col for col in keys]
    SHG_dict = dict(zip(keys,SHG_vals))
    TPF_stat_df = TPF_stat_df.rename(TPF_dict, axis=1)
    SHG_stat_df = SHG_stat_df.rename(SHG_dict, axis=1)
    pairs = [('TPF_'+i,'SHG_'+i) for i in keys]
    all_stats = pd.merge(TPF_stat_df, SHG_stat_df, on=common_cols, how='outer')
    Q1 = all_stats.loc[all_stats['quadrant']==1]
    Q2 = all_stats.loc[all_stats['quadrant']==2]
    Q3 = all_stats.loc[all_stats['quadrant']==3]
    quad_stats = [Q1,Q2,Q3]
    pvals_Q1 = dict(zip(keys,[np.NaN for i in range(len(keys))]))
    pvals_Q2 = dict(zip(keys,[np.NaN for i in range(len(keys))]))
    pvals_Q3 = dict(zip(keys,[np.NaN for i in range(len(keys))]))
    pvals = [pvals_Q1, pvals_Q2, pvals_Q3]
    for quadrant in [2,3,1]:
        quad_df = quad_stats[quadrant-1]
        for i in range(len(pairs)):
            feat_df = quad_df[[pairs[i][0],pairs[i][1]]].dropna()
            if feat_df.shape[0] > 0:
                pval = paired_permutation_test(
                    feat_df,
                    pairs[i],
                    number_permutations=10000,
                    significance=0.05,
                    path=None, 
                    title=None,
                    plot_significant=False)
                pvals[quadrant-1][pairs[i][0][4:]] = pval
                print(quadrant, pairs[i], pval)

Q1_p = pd.DataFrame.from_dict(pvals_Q1, orient='index',columns=['p'])
Q1_p = Q1_p.reset_index(level=0).rename(columns={'index':'stat','p':'p'})
Q1_p['quadrant'] = 1
Q2_p = pd.DataFrame.from_dict(pvals_Q2, orient='index',columns=['p'])
Q2_p = Q2_p.reset_index(level=0).rename(columns={'index':'stat','p':'p'})
Q2_p['quadrant'] = 2
Q3_p = pd.DataFrame.from_dict(pvals_Q3, orient='index',columns=['p'])
Q3_p = Q3_p.reset_index(level=0).rename(columns={'index':'stat','p':'p'})
Q3_p['quadrant'] = 3

save_path = "D:/admin/bone_analysis_paper/TPF_vs_SHG_paired_permutation_tests/"

#save out all p vals unadjusted for quadrants 1,2,3
all_p = pd.concat([Q1_p,Q2_p,Q3_p]).dropna().reset_index(drop=True)
all_p.to_csv(save_path+'pvalues_paired_SHG_TPF_quadrants_1_2_3_main_stats.csv')

p_28 = pd.concat([Q1_p[13:-2], Q2_p[:13], Q2_p[-2:]])
p_28 =p_28.reset_index(drop=True)

from statsmodels.stats.multitest import multipletests

reject, adj_p, alphasidak, alphacBonf = multipletests(p_28['p'], method='fdr_bh')
adj_p_df = np.c_[np.array(p_28['stat']),adj_p]
adj_p_df = pd.DataFrame(adj_p_df, columns=['stat','adj_p'])

final = pd.merge(p_28,adj_p_df, on=['stat'],how='left')
final = final.reindex([13,16,14,18,17,19,20,15,22,21,23,24,25,26,27,0,3,1,5,4,6,7,2,9,8,10,11,12]).reset_index(drop=True)
final['adj_p_3dp'] = final['adj_p'].apply(lambda x: round(x, 3))
final['p_3dp'] = final['p'].apply(lambda x: round(x, 3))
final = final[['quadrant','stat','p','p_3dp','adj_p','adj_p_3dp']]
final.to_csv(save_path+'pvalues_paired_SHG_TPF_quadrants_1_2_benhoch_adjusted_for_paper_table.csv')

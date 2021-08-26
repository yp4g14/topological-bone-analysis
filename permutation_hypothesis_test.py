# randomisation significance testing

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt
import utils as ut
import logging 
from itertools import combinations, product
from statsmodels.stats.multitest import multipletests
from importlib import reload
reload(ut)

logger = logging.getLogger("significance")
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(stream_formatter)
logger.addHandler(console)
from matplotlib import rcParams
rcParams.update({'font.size': 12})

def permutation_test(
    feat_group1,
    feat_group2,
    metric_func=np.average,
    number_permutations=10000,
    significance=0.05,
    path=None, 
    title=None,
    plot_significant=False):
    """Permutation significance test

    Args:
        feat_group1 (array): feature group 1
        feat_group2 (array): features group 2
        metric_func (function, optional): How to assess the difference in 
            the groups. Defaults to np.average (mean).
        number_permutations (int, optional): Number of shuffles on pooled data.
            Defaults to 10000.
        significance (float, optional): significance level, 
            will only plot histplots if significant. Defaults to 0.05.
        path (string, optional): save location path as string. 
            Defaults to None.
        title (string, optional): histplot title. Defaults to None.

    Returns:
        p_val (float): pseudo p-value from permutation test.
    """
    # difference in features in true split
    diff_truths = np.abs(metric_func(feat_group1)-metric_func(feat_group2))

    # group all before shuffling
    feat_all = list(feat_group1) +  list(feat_group2)

    #Copy the all features list
    pool_sample = copy.copy(feat_all)
    
    #Initialize permutation distribution
    pool_distribution = []

    # Permutation loop
    for i in range(0,number_permutations):
    # Shuffle the copy of the all features set
        random.shuffle(pool_sample)
        # Compute permuted absolute difference of the two sampled distributions
        pool_distribution.append(
            np.abs(metric_func(pool_sample[0:len(feat_group1)])\
                 - metric_func(pool_sample[len(feat_group1):])))

    p_val = len(np.where(pool_distribution >= diff_truths)[0])\
        /number_permutations
    if plot_significant:
        if p_val <= significance:
            sns.histplot(pool_distribution, stat='probability', linewidth=0)
            plt.xlabel('Permuted difference')
            plt.ylabel('Proportion')
            ylims = plt.ylim()
            plt.vlines(
                diff_truths,
                ymin=0,
                ymax=1,
                color='orange',
                label=str(diff_truths)
                )
            plt.ylim(ylims)
            if title:
                plt.title(f"{title}")
            if path:
                plt.savefig(f"{path}permutation_hypothesis_tests/{title}.png")
                plt.tight_layout()
            plt.close()
    return p_val

def multiple_comparison_correction(
    p_vals_path,
    stats_list,
    logger,
    category_pair=('group_a','group_b'),
    alpha=0.05
    ):
    
    # read p_values
    p_values_df = pd.read_csv(p_vals_path, index_col=0)
    
    # restrict p_values to just the one category pair
    p_values_df = p_values_df.loc[
        (
            (p_values_df['category 0']==category_pair[0]) &
            (p_values_df['category 1']==category_pair[1])
        ) | (
            (p_values_df['category 0']==category_pair[1]) &
            (p_values_df['category 1']==category_pair[0])
        )
    ] 

    # restrict p_values to just the stats columns and quadrants
    p_values_df['include'] = p_values_df['stat_col'].apply(lambda s: s in stats_list)
    p_values_df = p_values_df.loc[p_values_df['include']].drop('include',axis=1)

    stats_consider_df = pd.DataFrame(
        np.array(list(product(stats_list,[1,2,3]))),
        columns=['stat_col','quadrant'])
    stats_consider_df['quadrant'] =stats_consider_df['quadrant'].astype(int)
    p_values_df = stats_consider_df.merge(
        p_values_df,
        how='left',
        on=['stat_col', 'quadrant'])
    p_values_df = p_values_df.dropna()
    m = p_values_df.shape[0]
    # adjust the p_values using benjamini-hochberg
    pvals=p_values_df["p"]
    (reject, adjusted_pvals,alphasidak,alphabf) = multipletests(
        pvals,
        alpha=alpha,
        method='fdr_bh',
        is_sorted=False,
        returnsorted=False)

    p_values_df["reject"] = reject
    p_values_df["adjusted_p"] = adjusted_pvals

    p_values_df = p_values_df.sort_values(["adjusted_p"])
    p_values_df.to_csv(
        f"{p_vals_path[:-4]}_adjusted_{m}_alpha{alpha}_{category_pair}.csv")
    return p_values_df

# # # format for use in boxplots:
# # p1 = p1[['stat_col','quadrant','adjusted_p']].rename({'adjusted_p':'SHG_KO'}, axis=1)
# # p2 = p2[['stat_col','quadrant','adjusted_p']].rename({'adjusted_p':'SHG_WT'}, axis=1)
# # p3 = p3[['stat_col','quadrant','adjusted_p']].rename({'adjusted_p':'SHG_male'}, axis=1)
# # p4 = p4[['stat_col','quadrant','adjusted_p']].rename({'adjusted_p':'SHG_female'}, axis=1)

# # pvals_cond = p1.merge(p2, on=['stat_col','quadrant'], how='outer')
# # pvals_sex = p3.merge(p4, on=['stat_col','quadrant'], how='outer')
# # SHG_pvals = pvals_cond.merge(pvals_sex, on=['stat_col','quadrant'], how='outer')

# # p5 = p5[['stat_col','quadrant','adjusted_p']].rename({'adjusted_p':'TPF_KO'}, axis=1)
# # p6 = p6[['stat_col','quadrant','adjusted_p']].rename({'adjusted_p':'TPF_WT'}, axis=1)
# # p7 = p7[['stat_col','quadrant','adjusted_p']].rename({'adjusted_p':'TPF_male'}, axis=1)
# # p8 = p8[['stat_col','quadrant','adjusted_p']].rename({'adjusted_p':'TPF_female'}, axis=1)

# # pvals_cond = p5.merge(p6, on=['stat_col','quadrant'], how='outer')
# # pvals_sex = p7.merge(p8, on=['stat_col','quadrant'], how='outer')
# # TPF_pvals = pvals_cond.merge(pvals_sex, on=['stat_col','quadrant'], how='outer')

# # all_pvals = TPF_pvals.merge(SHG_pvals, on=['stat_col', 'quadrant'])
# # save_path = "C:/Users/yp4g14/Documents/admin/bone_analysis_paper/adjusted_alpha0.05_p_values_28stats_SHG_TPEF.csv"
# # all_pvals.to_csv(save_path)

def group_comparison_permutation_test(
    name_maps,
    stat_df,
    save_path,
    logger,
    stats_list_to_test=None,
    metric_func=np.average,
    plot_significant=False,
    adjust_p_vals=True,
    alpha=0.05
    ):
    ut.directory(save_path)
    if plot_significant:
        ut.directory(save_path+"permutation_hypothesis_tests/")

    stat_df['category'] = stat_df['filename'].map(name_maps)
    categories = list(set(stat_df['category']))
    if len(categories)<2:
        logger.error(f"{len(categories)} categories given, expected 2 or more")

    pairs = list(combinations(categories, r=2))
    logger.info(f"Category combinations to test {pairs}")
    # pick feature column
    cols = list(stats.columns)
    ignore_cols = ['filename', 'quadrant','patch_number','category']
    if stats_list_to_test is not None:
        cols = stats_list_to_test
    else:
        cols = [u for u in cols if u not in ignore_cols]

    p_vals = []
    failed = []
    for category_pair in pairs:
        for quadrant in [1,2,3]:
            df = stat_df[stat_df['quadrant']==quadrant]
            cat0_allcols = df[df['category']==category_pair[0]]
            cat1_allcols = df[df['category']==category_pair[1]]
            for stat_col in cols:
                cat0 = cat0_allcols[[stat_col]]
                cat0 = np.asarray(cat0.dropna())
                cat1 = cat1_allcols[[stat_col]]
                cat1 = np.asarray(cat1.dropna())

                if (len(cat0) > 0) and (len(cat1)>0): 
                    test_name=f"{category_pair[0]}_{category_pair[1]}_Q{quadrant}_{stat_col}"
                    p = permutation_test(
                        cat0,
                        cat1,
                        metric_func,
                        number_permutations=10000,
                        path=save_path,
                        title=test_name,
                        plot_significant=plot_significant
                        )
                    p_vals.append(
                        [quadrant,
                        stat_col,
                        p,
                        categories[0],
                        categories[1],
                        10000]
                        )
                else:
                    failed.append([f"test skipped due to len 0 array, {category_pair} Q{quadrant} stat {stat_col}"])
            logger.info(f"Finished quadrant {quadrant} tests")

    p_vals = pd.DataFrame(
        p_vals,
        columns = ['quadrant', 'stat_col', 'p', 'category 0','category 1','number of permutations'])
    p_vals.to_csv(save_path+'p_values.csv')
    logger.info(f"p values complete see {save_path+'p_values.csv'}")

    if adjust_p_vals:
        if stats_list_to_test is None:
            logger.error(f"p_values could not be adjusted as stats_list_to_test not specified")
        adjusted_p_vals = multiple_comparison_correction(
            save_path+'p_values.csv',
            stats_list_to_test,
            logger,
            category_pair=('group_a','group_b'),
            alpha=alpha
            )
        logger.info(f"adjusted p values complete see {save_path}")
        return adjusted_p_vals
    else:
        return p_vals

if __name__ == "__main__":
    save_path = "D:/topological-bone-analysis/example/2021_08_26_Time_13_38/"    
    stats_path = save_path+"all_statistics.csv"
    name_maps = {'example_SHG_1.tif':'group_a', 'example_SHG_2.tif':'group_b'}
    stat_df=pd.read_csv(stats_path,index_col=0)
    group_comparison_permutation_test(
    name_maps,
    stat_df,
    save_path,
    logger,
    stats_list_to_test=['0_num_points','0_avg_birth','1_num_points'],
    metric_func=np.average,
    plot_significant=False,
    adjust_p_vals=True,
    alpha=0.05
    )
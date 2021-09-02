import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils as ut

def analysis_plots(
            df,
            filenames_map,
            plot_path,
            feature_cols
            ):
    if feature_cols is None:
        identifiers = ['filename', 'patch_number', 'quadrant']
        feature_cols = [col for col in df.columns if col not in identifiers]
    # initialise directories
    for quadrant in [1,2,3]:
        ut.directory(f"{plot_path}Q{quadrant}")

    # create category map from filenames dictionary
    df['category'] = df['filename'].map(filenames_map)
    for quadrant in range(1,4):
        stats = df.loc[df['quadrant']==quadrant]
        for col in feature_cols:
            single_stat = stats[['category', col]]
            single_stat = single_stat.dropna()
            if single_stat.shape[0] >= 2:
                sns.catplot(x='category', y=col, data=single_stat, dodge=True)
                plt.savefig(f"{plot_path}/Q{quadrant}/{col}.svg")
                plt.close()

                sns.boxplot(x='category', y=col, data=single_stat, dodge=True)
                plt.savefig(f"{plot_path}Q{quadrant}/{col}.svg")
                plt.close()
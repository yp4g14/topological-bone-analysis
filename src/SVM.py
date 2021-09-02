import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold,\
 RepeatedStratifiedKFold,LeaveOneGroupOut, GridSearchCV, LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn import metrics
from tqdm.auto import tqdm
def prepare_features(
    stats,
    feature_cols,
    filenames_map,
    strat_col=None,
):
    stats['category'] = stats['filename'].map(filenames_map)
    categories = list(set(stats['category']))
    if len(categories)!=2:
        print(f"Expected 2 categories, got {len(categories)}: {categories}")
    cat_map = dict(zip(categories,[0,1]))
    stats['category_int'] = stats['category'].map(cat_map)
    
    basic_cols = ['filename','patch_number','category','category_int']
    if strat_col is not None:
        basic_cols += [strat_col]

    features_quad2 = stats[stats['quadrant']==2]
    feature_cols_quad2 = [col for col in feature_cols if col[0]=='0']
    features_quad2 = features_quad2[basic_cols+feature_cols_quad2]

    features_quad1 = stats[stats['quadrant']==1]
    feature_cols_quad1 = [col for col in feature_cols if col[0]=='1']
    features_quad1 = features_quad1[basic_cols+feature_cols_quad1]

    features = pd.merge(
        features_quad1,
        features_quad2,
        on=basic_cols)
    features = features.dropna(axis=0)
    return features

def classification_one_v_one(
    df,
    save_path,
    logger,
    feature_cols,
    filenames_map,
    runs=100,
    strat_col=None,
    cross_val='stratkfold',
    param_grid_SVC = {'C': [1,2,3], 'kernel': ('rbf','linear')}
):
    features = prepare_features(
        df,
        feature_cols,
        filenames_map,
        strat_col=None
    )
    features.to_csv(save_path+'features.csv')

    df = features
    results = []
    params=[]
    for j in tqdm(range(runs),desc='runs'):
        X = np.asarray(df[feature_cols])
        y = np.asarray(df['category_int'])
        if strat_col is not None:
            groups = np.asarray(df[strat_col])
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        if cross_val == 'stratkfold':
            r_strat_kf= RepeatedStratifiedKFold(n_splits=10, n_repeats=1)
            # is this actually stratified????
            r_strat_kf_split = r_strat_kf.split(X,y)
            cv_split = r_strat_kf_split
        elif cross_val == 'kfold':
            r_kf= RepeatedKFold(n_splits=10, n_repeats=1)
            r_kf_split = r_kf.split(X,y)
            cv_split = r_kf_split
        elif cross_val == 'logo':
            logo = LeaveOneGroupOut()
            logo_split = logo.split(X,y, groups=groups)
            cv_split = logo_split
        else:
            logger.error(f"error in run {j} Cross Validation")

        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for train, test in cv_split:
            X_train.append(X[train])
            X_test.append(X[test])
            y_train.append(y[train])
            y_test.append(y[test])

        scores = []
        keys = list(param_grid_SVC.keys())
        num_keys=len(keys)
        chosen_model = dict(
            zip(param_grid_SVC.keys(),[[] for i in range(num_keys)])
            )
        
        for i in range(len(X_train)):
            # do an inner cross valdation grid search to select the best model
            # on the training set automatically. This leads to different models
            # per split
            clf = GridSearchCV(
                SVC(),
                param_grid=param_grid_SVC,
                cv=10)
            clf.fit(X_train[i],y_train[i])
            clf.cv_results_
            model = clf.best_estimator_
            for k in range(num_keys):
                chosen_model[keys[k]].append(clf.best_params_[keys[k]])
            prediction = model.predict(X_test[i])
            acc = metrics.accuracy_score(y_test[i],prediction)
            precision = metrics.precision_score(y_test[i],prediction) #, average=None) #, zero_division=0)
            recall = metrics.recall_score(y_test[i],prediction) #, average=None) #, zero_division=0)
            f1 = metrics.f1_score(y_test[i],prediction) #, average=None)
            scores.append([j, acc, precision, recall, f1])

        scores = np.array(scores).mean(axis=0)
        results.append(
            pd.DataFrame(
                scores.reshape(5,1).T,
                columns=['run','average accuracy','average precision','averag recall','average F1']
                )
            )
        split_params = pd.DataFrame.from_dict(chosen_model, orient='columns')
        split_params['run'] = j
        params.append(split_params)

    # summarise chosen model parameters and save to csv
    params = pd.concat(params).reset_index(drop=True)
    params = params.groupby(keys).count().rename(columns={'run':'total models'})
    params.to_csv(save_path+'SVC_chosen_parameters.csv')

    results = pd.concat(results)
    results.to_csv(save_path+'SVC_results.csv')
    return results

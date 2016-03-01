import os
import pandas as pd
import numpy as np
from datetime import date, timedelta as td
import time
from sklearn import cross_validation

from configuration import CONFIG
from utils import load_submission, load_train_df
from learning.date_shuffle_split import DateShuffleSplit, train_test_split
from learning.feature_engineering import FeatureFactory


def create_submission_file(estimator, cols, train_path=None, in_path=None, out_path=None):
    # Set defaults
    if out_path is None:
        out_path = os.path.join(CONFIG.results_path, "submission.txt")
    if train_path is None:
        train_path = CONFIG.preprocessed_train_path

    train_df = load_train_df(train_path, 50000)
    print("Number of training days:", len(set(train_df['DATE']))/48)
    submission_df = load_submission(in_path)
    for assignment in CONFIG.submission_assignments:
        t_df = train_df[train_df['ASS_ASSIGNMENT'] == assignment]
        sub_df = submission_df[submission_df['ASS_ASSIGNMENT'] == assignment]
        X, y, _, _ = get_cross_validation_parameters(t_df, cols)
        X_submission, _, _, _ = get_cross_validation_parameters(sub_df, cols, label='prediction')
        start = time.time()
        estimator.fit(X, y)
        print("Estimator fit in %i seconds." % (time.time() - start))
    submission_df['prediction'] = [int(i+0.5) for i in estimator.predict(X_submission)]
    submission_df.to_csv(out_path, sep='\t', index=None, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S.000')
    return submission_df


def get_cross_validation_parameters(df, columns, k_fold=5, out_path=None, label=None):
    """
    Get the parameters you need for sklearn cross-validation functions.

    Parameters
    ==========
    df: pd.Dataframe, the input data.
    columns: list, the features to be processed and/or kept.
    k_fold: The number of partitions of size 1/k to choose for cross-validation.
    out_path: str or None, path to a file in which X is saved, to avoid recomputing features.
        None not to save this file.

    Returns
    =======
    X: np.array, the data set.
    y: np.array, the corresponding labels.
    cv: An object to be used as a cross-validation generator.
    dates: np.array, the list of dates corresponding to the data set.
    """
    if label is None:
        label = 'CSPL_RECEIVED_CALLS'
    ff = FeatureFactory(df)
    dates = np.array(ff('full_date'))
    y = np.array(df[label])
    for column in columns:
        print(column)
        ff(column)
    ff.select_features(columns)
    df = ff.X
    if out_path is not None:
        df.to_csv(out_path)
    X = np.array(df)
    cv = DateShuffleSplit(dates, n_iter=k_fold, test_size=float(1)/k_fold)
    return X, y, cv, dates


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    df = load_train_df(CONFIG.preprocessed_train_path, chunksize=500000)
    submission_df = load_submission()
    estimator = LogisticRegression()
    k_fold = 5
    n_jobs = 3
    verbose = 0
    fit_params = None
    cols = ["WEEK_DAY", "TIME"]
    df = df[df['ASS_ASSIGNMENT'] == 'Téléphonie']
    X, y, cv, dates = get_cross_validation_parameters(df, cols, k_fold=k_fold)
    df = create_submission_file(estimator, cols)
    # print(max(df['prediction']))
    # ######################## Examples ######################## #
    # X_train, X_test, y_train, y_test = train_test_split(dates, X, y)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    # predict = cross_validation.cross_val_predict(estimator, X, y, cv=cv, n_jobs=n_jobs, verbose=verbose,
    #                                              fit_params=fit_params)
    # print(predict)
    score = cross_validation.cross_val_score(estimator, X, y, cv=cv, n_jobs=n_jobs, verbose=verbose,
                                             fit_params=fit_params)
    print(score)
    # score, perm_score, value = cross_validation.permutation_test_score(estimator, X, y, cv=cv, n_permutations=k_fold,
    #                                                                    n_jobs=n_jobs, verbose=verbose)
    # print(score)
    # print(perm_score)
    # print(value)

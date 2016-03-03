import os
import pandas as pd
import numpy as np
import scipy as sp
from datetime import date, timedelta as td
import time
import logging
from sklearn import cross_validation

from configuration import CONFIG
from utils import load_submission, load_train_df
from learning.date_shuffle_split import DateShuffleSplit, train_test_split
from learning.feature_engineering import FeatureFactory


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def create_submission_file(estimator, cols, weights=None, train_path=None, in_path=None, out_path=None):
    """
    Creates the submission.txt file.
    :param estimator:
    :param cols:
    :param weights:
    :param train_path:
    :param in_path:
    :param out_path:
    :return:
    """
    # Set defaults
    if out_path is None:
        out_path = os.path.join(CONFIG.results_path, "submission.txt")
    if train_path is None:
        train_path = CONFIG.preprocessed_train_path

    train_df = load_train_df(train_path, 50000)
    logger.info("Number of training days: %i" % int(len(set(train_df['DATE']))/48))
    submission_df = load_submission(in_path)
    predictions = {}
    for assignment in CONFIG.submission_assignments:
        logger.debug("*"*50)
        logger.debug(assignment)
        t_df = train_df[train_df['ASS_ASSIGNMENT'] == assignment]
        sub_df = submission_df[submission_df['ASS_ASSIGNMENT'] == assignment]
        X, y, _, _ = get_cross_validation_parameters(t_df, cols, weights=weights)
        X_submission, _, _, _ = get_cross_validation_parameters(sub_df, cols, weights=weights, label='prediction')
        try:
            start = time.time()
            estimator.fit(X, y)
            logger.debug("Estimator fit in %i seconds." % (time.time() - start))
            prediction = estimator.predict(X_submission)
        except ValueError as e:
            if str(e) == 'This solver needs samples of at least 2 classes in the data, ' \
                         'but the data contains only one class: 0.0':
                logger.warning('WARNING: Only 0 in prediction')
                prediction = [0]*len(sub_df.index)
            else:
                raise e
        for i, ind in enumerate(sub_df.index):
            try:
                predictions[ind] = int(prediction[i]+0.5)
            except ValueError as e:
                if str(e) == "cannot convert float NaN to integer":
                    logger.warning("cannot convert float NaN to integer")
                    predictions[ind] = 0
                else:
                    raise e
        logger.debug("Train max value: %i, Predicted max value: %i" % (max(y), max(prediction)))
    submission_df['prediction'] = [predictions[i] for i in submission_df.index]
    submission_df.to_csv(out_path, sep='\t', index=None, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S.000')
    return submission_df


def get_cross_validation_parameters(df, columns, k_fold=5, weights=None, out_path=None, label=None):
    """
    Get the parameters you need for sklearn cross-validation functions.

    Parameters
    ==========
    df: pd.Dataframe, the input data.
    columns: list, the features to be processed and/or kept.
    k_fold: The number of partitions of size 1/k to choose for cross-validation.
    weights: array-like, each column will be multiplied by its corresponding weight (e.g. impact on KNN)
    out_path: str or None, path to a file in which X is saved, to avoid recomputing features.
        None not to save this file.
    label: str, Name of the 'calls' column (CSPL_RECEIVED_CALLS or prediction)

    Returns
    =======
    X: np.array or sparse matrix, the data set.
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
        logger.debug(column)
        ff(column)
    if 'WEEK_DAY' in columns or 'WEEK_DAY_NAME' in columns or 'TIME' in columns:
        sparse_matrices = []
        if 'WEEK_DAY' in columns or 'WEEK_DAY_NAME' in columns:
            sparse_matrices.append(ff.get_sparse_matrix("WEEK_DAY_NAME"))
        if 'TIME' in columns:
            sparse_matrices.append(ff.get_sparse_matrix("TIME"))
        if set(columns).difference({'WEEK_DAY', 'WEEK_DAY_NAME', 'TIME'}):
            sparse_matrices.append(ff.select_features(set(columns).difference({'WEEK_DAY', 'WEEK_DAY_NAME', 'TIME'})))
        tup = tuple(sparse_matrices)
        X = sp.sparse.hstack(tup)

        # TODO: Trying to remove duplicates in X
        # Xdf = pd.DataFrame(X.todense())
        # ydf = pd.DataFrame(y, columns=['y'])
        # df = pd.concat((Xdf, ydf), axis=1)
        # grouped = df.groupby(list(range(X.shape[1])))
        # df_mean = grouped[['y']].mean().reset_index()
        # df_max = grouped[['y']].max().reset_index()
        # df = alpha*mean+(1-alpha)*max
        # X = np.array(df[list(range(X.shape[1]))])
        # y = np.array(df['y'])
    else:
        ff.select_features(columns)
        ff.apply_weights(weights)
        X = np.array(ff.X)
    if out_path is not None:
        ff.X.to_csv(out_path)
    cv = DateShuffleSplit(dates, n_iter=k_fold, test_size=float(1)/k_fold)
    return X, y, cv, dates


def cross_val_score(estimator, cols, k_fold, weights=None, scoring=None, n_jobs=1, verbose=0, fit_params=None,
                    chunksize=None):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    df = load_train_df(CONFIG.preprocessed_train_path)
    cum_score = np.zeros(k_fold)
    scores = {}
    for assignment in CONFIG.submission_assignments:
        t_df = df[df['ASS_ASSIGNMENT'] == assignment]
        if chunksize is not None:
            t_df = t_df.loc[t_df.index[:chunksize]]
        X, y, cv, dates = get_cross_validation_parameters(t_df, cols, k_fold=k_fold, weights=weights)
        score = cross_validation.cross_val_score(estimator, X, y, scoring=scoring, cv=cv,
                                                 n_jobs=n_jobs, verbose=verbose, fit_params=fit_params)
        logger.info("Score of %s: %s" % (assignment, str(np.mean(score))))
        scores[assignment] = np.mean(score)
        cum_score += score
    logger.info("Average score: %s" % str(cum_score/len(CONFIG.submission_assignments)))
    return scores


if __name__ == "__main__":
    from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor, NearestCentroid
    from sklearn.linear_model import TheilSenRegressor, LogisticRegression, RandomizedLogisticRegression, \
        ElasticNet, Ridge, SGDRegressor
    from sklearn import metrics
    from sklearn.feature_selection import SelectKBest, chi2
    import matplotlib.pyplot as plt
    pd.options.mode.chained_assignment = None  # Remove warnings

    logger.setLevel(logging.DEBUG)

    _df = load_train_df(CONFIG.preprocessed_train_path)
    _submission_df = load_submission()
    # _estimator = KNeighborsRegressor(n_neighbors=50, weights='distance')
    _estimator = RadiusNeighborsRegressor()
    _scoring = 'mean_squared_error'
    _k_fold = 3
    _n_jobs = 3
    _verbose = 0
    _fit_params = None
    _cols = ["TIME"]
    _weights = None
    # df = create_submission_file(_estimator, _cols)
    # print(max(df['prediction']))
    print(cross_val_score(_estimator, _cols, _k_fold, _weights, _scoring, _n_jobs, _verbose, _fit_params,
                          chunksize=10000))

    # ########################### Plot the scores of the feature selection #############################
    # ind = np.arange(len(_cols))
    # width = 0.35
    # for assignment in CONFIG.submission_assignments:
    #     _t_df = _df[_df['ASS_ASSIGNMENT'] == assignment]
    #     X, y, cv, dates = get_cross_validation_parameters(_t_df, _cols)
    #     select = SelectKBest(chi2, k='all')
    #     select.fit_transform(X, y)
    #     plt.bar(ind, select.scores_/max(select.scores_), width/2, color='blue')
    #     plt.bar(ind + width/2, 1 - select.pvalues_/max(select.pvalues_), width/2, color='red')
    #     plt.xticks(ind + width/2, tuple(_cols))
    #     plt.show()
    #     plt.clf()
    #     print(select.scores_)
    #     print(select.pvalues_)

    # ##################### Find the optimal choice of columns #####################
    # best_cols = dict(zip(CONFIG.submission_assignments, ['TIME']*len(CONFIG.submission_assignments)))
    # best_scores = None
    # _cols = []
    # _weights = [0.1]
    # for col in ["TIME", "WEEK_DAY", "YEAR", "WEEK_NUMBER"]:
    #     _cols += [col]
    #     if col != "TIME":
    #         _weights = [1] + _weights
    #     scores = cross_val_score(_estimator, _cols, _k_fold, _weights, _scoring, _n_jobs, _verbose, _fit_params,
    #                              chunksize=100000)
    #     if best_scores is None:
    #         best_scores = scores
    #     else:
    #         for k in best_scores.keys():
    #             if scores[k] > best_scores[k]:
    #                 best_scores[k] = scores[k]
    #                 best_cols[k] = col
    #     print(sum(best_scores.values())/len(best_scores))
    # print(best_cols)

    # ######################## Examples ######################## #
    # X_train, X_test, y_train, y_test = train_test_split(dates, X, y)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    # predict = cross_validation.cross_val_predict(estimator, X, y, cv=cv, n_jobs=n_jobs, verbose=verbose,
    #                                              fit_params=fit_params)
    # print(predict)

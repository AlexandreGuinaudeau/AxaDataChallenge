import os
import pandas as pd
import numpy as np
import scipy as sp
from datetime import date, timedelta as td
import time
import logging
from sklearn import cross_validation

from configuration import CONFIG
from utils import load_submission, load_train_df, load_weather_df, load_means_df
from learning.date_shuffle_split import DateShuffleSplit, train_test_split
from learning.feature_engineering import FeatureFactory


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def get_df(cols, load_from_temp, temp_path):
    weather_df = load_weather_df(CONFIG.preprocessed_meteo_path_complete)
    if not load_from_temp:
        logger.info('Loading train Dataframe...')
        train_df = load_train_df(CONFIG.preprocessed_train_path_means)
        logger.info('Loading weather Dataframe...')

        logger.info('Creating features...')
        ff = FeatureFactory(train_df, weather_df)
        for col in cols:
            logger.info('Creating %s feature...' % col)
            ff(col)
        if 'ASS_ASSIGNMENT' not in cols:
            cols = ['ASS_ASSIGNMENT'] + cols
        if 'DATE' not in cols:
            cols = ['DATE'] + cols
        if 'CSPL_RECEIVED_CALLS' not in cols:
            cols += ['CSPL_RECEIVED_CALLS']

        logger.info('Selecting features...')
        ff.select_features(cols)
        train_df = ff.X
        if temp_path is not None:
            train_df.to_csv(temp_path)
    else:
        assert temp_path is not None
        logger.info('Loading train Dataframe...')
        train_df = pd.read_csv(temp_path, encoding='latin-1', index_col=0, parse_dates=['DATE'])
    weather_df.reset_index(inplace=True)
    return train_df, weather_df


def get_cross_validation_parameters(df, columns, weather_df=None, k_fold=5, weights=None, out_path=None, label=None):
    """
    Get the parameters you need for sklearn cross-validation functions.

    Parameters
    ==========
    df: pd.Dataframe, the input data.
    columns: list, the features to be processed and/or kept.
    weather_df: pd.Dataframe, the weather data.
    k_fold: The number of partitions of size 1/k to choose for cross-validation.
    weights: dict, each column will be multiplied by its corresponding weight (e.g. impact on KNN)
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
    if weights is None:
        weights = {}

    df.reset_index(inplace=True, drop=True)
    ff = FeatureFactory(df, weather_df)
    dates = np.array(ff('full_date'))
    y = np.array(df[label]) - df['MEAN']
    for column in columns:
        logger.debug(column)
        ff(column)
    ff.select_features(columns)
    ff.apply_weights(weights)
    X = np.array(ff.X)
    if out_path is not None:
        ff.X.to_csv(out_path)
    cv = DateShuffleSplit(dates, n_iter=k_fold, test_size=float(1)/k_fold)
    return X, y, cv, dates


def create_submission_file(estimator, cols, weights=None, in_path=None, out_path=None, verbose=None,
                           load_from_temp=None, temp_path=None):
    """
    Creates the submission.txt file.
    """
    # Set defaults
    if verbose is None or verbose == 'WARNING':
        logger.setLevel(logging.WARNING)
    if verbose == 'INFO':
        logger.setLevel(logging.INFO)
    if verbose == 'DEBUG':
        logger.setLevel(logging.INFO)
    if out_path is None:
        out_path = os.path.join(CONFIG.results_path, "submission.txt")

    train_df, weather_df = get_df(cols, load_from_temp, temp_path)

    submission_df = load_submission(in_path)
    ff = FeatureFactory(submission_df.copy(), weather_df)
    for col in set(cols).union({'TIME', 'WEEKEND'}):
        ff(col)
    sub_df = ff.X

    predictions = {}

    for assignment in CONFIG.submission_assignments:
        logger.info('Preparing for submission: %s...' % assignment)
        t_df = train_df[train_df['ASS_ASSIGNMENT'] == assignment]
        X, y, _, _ = get_cross_validation_parameters(t_df, cols, weather_df=weather_df,
                                                     weights=weights)
        s_df = sub_df[sub_df['ASS_ASSIGNMENT'] == assignment]
        X_submission, _, _, _ = get_cross_validation_parameters(s_df, cols, weather_df=weather_df,
                                                                weights=weights, label='prediction')
        estimator.fit(X, y)
        predictions[assignment] = list(estimator.predict(X_submission))

    sub_df['raw_prediction'] = sub_df['ASS_ASSIGNMENT'].apply(lambda x: predictions[x].pop(0))
    submission_df['prediction'] = \
        sub_df.apply(lambda x: max(int(x['raw_prediction'] + x['MEAN'] + 0.5), 0), axis=1)
    submission_df.drop('MEAN', axis=1, inplace=True)

    submission_df.to_csv(out_path, sep='\t', index=None, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S.000')
    return submission_df


def cross_val_score(estimator, cols, k_fold, weights=None, scoring=None, n_jobs=1, verbose=0,
                    fit_params=None, chunksize=None, temp_path=None, load_from_temp=False):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    df, weather_df = get_df(cols, load_from_temp, temp_path)
    for assignment in CONFIG.submission_assignments:
        ass_df = df[df['ASS_ASSIGNMENT'] == assignment]
        # print(df['CSPL_RECEIVED_CALLS'].std())
        # print(np.corrcoef(df['CSPL_RECEIVED_CALLS'], [i**3 for i in t_df['NUMB_FROZEN_DEPT']])[0, 1])
        # print(np.corrcoef(df['CSPL_RECEIVED_CALLS'], [i**3 for i in t_df['NUMB_WET_DEPT']])[0, 1])

        X, y, cv, dates = get_cross_validation_parameters(ass_df, cols, weather_df=weather_df, k_fold=k_fold,
                                                          weights=weights)
        score = cross_validation.cross_val_score(estimator, X, y, scoring=scoring, cv=cv,
                                                 n_jobs=n_jobs, verbose=verbose, fit_params=fit_params)

        logger.info("Average score for %s: %s" % (assignment, str(np.mean(score))))


if __name__ == "__main__":
    from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor, NearestCentroid
    from sklearn.linear_model import TheilSenRegressor, LogisticRegression, RandomizedLogisticRegression, \
        ElasticNet, Ridge, SGDRegressor, ARDRegression, Perceptron, PassiveAggressiveRegressor, BayesianRidge, \
        OrthogonalMatchingPursuit, RANSACRegressor
    from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
    from sklearn.svm import SVR, LinearSVR
    from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.isotonic import IsotonicRegression

    from sklearn import metrics
    from sklearn.feature_selection import SelectKBest, chi2

    from learning.stacked_regressor import StackedRegression
    pd.options.mode.chained_assignment = None  # Remove warnings

    logger.setLevel(logging.DEBUG)

    # _df = load_train_df(CONFIG.preprocessed_train_path)
    _submission_df = load_submission()
    # _estimator = KNeighborsRegressor(n_neighbors=10, weights='distance')
    # _estimator = ARDRegression()
    _estimator = BayesianRidge()
    # _estimator = OrthogonalMatchingPursuit()

    _scoring = 'mean_squared_error'
    _k_fold = 3
    _n_jobs = 3
    _verbose = 0
    _fit_params = None
    _cols = ["NUMB_FROZEN_DEPT", "NUMB_FROZEN_DEPT"]
    _weights = None
    _temp_path = os.path.join(os.getcwd(), 'train.csv')

    # _stacked_estimator = StackedRegression()
    # df, weather_df = get_df(cols=_cols, load_from_temp=True, temp_path=_temp_path)
    # df = df[df['TIME'] >= 8]
    # df = df[df['WEEKEND'] == 0]
    # df = df[df['CUM_DAYS'] % 20 == 0]
    # df = df[df["ASS_ASSIGNMENT"].isin(['CAT', 'Tech. Axa', 'Téléphonie'])]
    # X, y, _, _ = get_cross_validation_parameters(df, _cols, weather_df)
    # _stacked_estimator.fit_stack(X, y)
    #
    # _estimator = _stacked_estimator

    df = create_submission_file(_estimator, _cols, verbose='INFO', load_from_temp=True,
                                temp_path=_temp_path)
    print(max(df['prediction']))
    # CONFIG.submission_assignments = ['CAT', 'Tech. Axa', 'Tech. Inter', 'Tech. Total', 'Téléphonie']
    # cross_val_score(_estimator, _cols, _k_fold, _weights, _scoring, _n_jobs, _verbose, _fit_params,
    #                 chunksize=10000, temp_path=_temp_path, load_from_temp=True)

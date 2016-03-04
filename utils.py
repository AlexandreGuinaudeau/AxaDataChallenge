import pandas as pd
from configuration import CONFIG


def load_submission(path=None):
    # Set defaults
    if path is None:
        path = CONFIG.submission_path_mean
    return pd.read_csv(path, sep='\t', parse_dates=[0])


def load_train_df(path, chunksize=None):
    dtype = {'ASS_ASSIGNMENT': str, 'CSPL_RECEIVED_CALLS': float}
    if chunksize is not None:
        for chunk in pd.read_csv(path, encoding='latin-1', index_col=0, chunksize=chunksize, dtype=dtype,
                                 parse_dates=['DATE']):
            return chunk
    return pd.read_csv(path, encoding='latin-1', index_col=0, dtype=dtype, parse_dates=['DATE'])


def load_weather_df(path, chunksize=None):
    dtype = {'ASS_ASSIGNMENT': str, 'CSPL_RECEIVED_CALLS': float}
    if chunksize is not None:
        for chunk in pd.read_csv(path, encoding='latin-1', index_col=0, chunksize=chunksize, dtype=dtype,
                                 parse_dates=['DATE']):
            return chunk
    return pd.read_csv(path, encoding='latin-1', index_col=0, dtype=dtype, parse_dates=['DATE'])


def load_means_df(path):
    return pd.read_csv(path, encoding='utf-8')

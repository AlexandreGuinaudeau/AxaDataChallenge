import pandas as pd
from configuration import CONFIG


def load_submission(path=None):
    # Set defaults
    if path is None:
        path = CONFIG.submission_path
    return pd.read_csv(path, sep='\t', parse_dates=[0])


def load_train_df(path, chunksize=None):
    dtype = {'ASS_ASSIGNMENT': str, 'YEAR': int, 'MONTH': int, 'DAY': int, 'WEEK_NUMBER': int, 'WEEKDAY': int,
             'DAY_OFF': int, 'TIME': float, 'CSPL_RECEIVED_CALLS': float}
    if chunksize is not None:
        for chunk in pd.read_csv(path, encoding='latin-1', index_col=0, chunksize=chunksize, dtype=dtype):
            return chunk
    return pd.read_csv(path, encoding='latin-1', index_col=0, dtype=dtype)
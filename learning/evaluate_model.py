import os
import pandas as pd
from datetime import date, timedelta as td

from configuration import CONFIG
from utils import load_submission, load_train_df


def create_submission_file(prediction_function, train_df, in_path=None, out_path=None):
    # Set defaults
    if out_path is None:
        out_path = os.path.join(in_path, "..", "submission.txt")
    df = load_submission(in_path)
    df['prediction'] = prediction_function(df.copy(), train_df)
    df.to_csv(out_path, sep='\t', index=None, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S.000')
    return df


def split_train_test(dates, remove_previous_days=None):
    """
    Creates train and test data sets, where the test set has only dates in `dates`.

    Parameters
    ==========
    dates: list of dates to be used in the testing files
    remove_previous_days: bool, whether the 2 days before a test date should be removed
    """
    # Set defaults
    if remove_previous_days is None:
        remove_previous_days = True
    dates_index = [(d.year, d.month, d.day) for d in dates]
    df = load_train_df(os.path.join(CONFIG.preprocessed_data_path, "train.csv"))
    df.set_index(['YEAR', 'MONTH', 'DAY'], inplace=True)
    test_df = df[df.index.isin(dates_index)]
    if remove_previous_days:
        dates_1 = [d - td(days=1) for d in dates]
        dates_2 = [d - td(days=2) for d in dates]
        dates.extend(dates_1)
        dates.extend(dates_2)
    dates_index = [(d.year, d.month, d.day) for d in dates]
    train_df = df[~df.index.isin(dates_index)]
    return train_df, test_df


def mean_square_error(expected_l, predicted_l):
    assert (len(expected_l) == len(predicted_l))
    diff = [(expected_l[i] - predicted_l[i])**2 for i in range(len(expected_l))]
    return sum(diff)/len(expected_l)


def evaluate_model(prediction_function, dates, remove_previous_days=None):
    train_df, test_df = split_train_test(dates, remove_previous_days=remove_previous_days)
    prediction = prediction_function(test_df.copy(), train_df)
    return mean_square_error(list(test_df['CSPL_RECEIVED_CALLS']), prediction)


if __name__ == "__main__":
    from learning.basic_model import predict
    # create_submission_file(predict)
    test_dates = [date(2012, 1, 17),
                  date(2012, 2, 1),
                  date(2012, 3, 5),
                  date(2012, 4, 2),
                  date(2012, 5, 26),
                  date(2012, 6, 25),
                  date(2012, 7, 8),
                  date(2012, 8, 7),
                  date(2012, 9, 6),
                  date(2012, 10, 10),
                  date(2012, 11, 19),
                  date(2012, 12, 14)]
    print(evaluate_model(predict, test_dates))

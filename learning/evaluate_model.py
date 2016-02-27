import os
import pandas as pd
from configuration import CONFIG


def load_submission():
    return pd.read_csv(CONFIG.submission_path, sep='\t', parse_dates=[0])


def create_submission_file(prediction_function):
    df = load_submission()
    df['prediction'] = prediction_function(df.copy())
    df.to_csv(os.path.join(CONFIG.results_path, "submission.txt"), sep='\t', index=None, encoding='utf-8',
              date_format='%Y-%m-%d %H:%M:%S.000')


def cross_validate(dates):
    """
    Evaluate the model by testing it on several dates and removing the 2 previous days in the training data
    (like in the submission file)

    Parameters
    ==========
    dates: The list of the dates for the testing
    """


if __name__ == "__main__":
    from learning.basic_model import predict
    create_submission_file(predict)

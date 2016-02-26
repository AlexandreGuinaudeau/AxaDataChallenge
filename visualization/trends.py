# Visualize the main trends: daily, weekly and yearly.
import os
import pandas as pd
import matplotlib.pyplot as plt

from configuration import CONFIG
import time


def parse(columns):
    # date_format='%Y-%m-%d %H:%M:%S.%f'
    # for df in pd.read_csv(CONFIG.train_file, sep=";", usecols=columns, parse_dates=[0], chunksize=10000):
    df = pd.read_csv(CONFIG.train_file, sep=";", usecols=columns, parse_dates=[0])
    df['TIME'] = df['DATE'].apply(lambda d: d.hour + float(d.minute)/60)
    df['WEEK_NUMBER'] = df['DATE'].apply(lambda d: d.isocalendar()[1])
    df['WEEKDAY'] = df['DATE'].apply(lambda d: d.isocalendar()[2])
    df.drop('DATE', axis=1, inplace=True)
    return df


def load_df(chunksize=None):
    if chunksize is not None:
        for chunk in pd.read_csv(train_path, encoding='latin-1', index_col=0, chunksize=chunksize):
            return chunk
    return pd.read_csv(train_path, encoding='latin-1', index_col=0)


def compare_calls(scale, out_path, assignments=None, remove_days_off=True):
    """
    Plot the number of calls to compare them.

    Parameters
    ==========
    scale: 'DAY', 'WEEK' or 'YEAR', calls are averaged on all smaller scales, and plotted for larger scales.
    out_path: str, folder in which figures should be saved.
    assignments: str or list of str, assignments to take into account.
        None to take all columns into account.
    remove_days_off: whether days off should be removed

    Example
    =======
    Week comparison: For each day of the week, take the average number of calls, then compare for each week of the year.
    """
    assert scale in ['DAY', 'WEEK', 'YEAR']
    df = load_df()
    if assignments is not None:
        if isinstance(assignments, str):
            assignments = [assignments]
        df = df[df["ASS_ASSIGNMENT"].isin(assignments)]
    else:
        assignments = set(df['ASS_ASSIGNMENT'])
    assignments = sorted(assignments)
    if remove_days_off:
        df = df[df["DAY_OFF"] == 0]
    df.drop("DAY_OFF", axis=1, inplace=True)

    grouped = df.groupby(["ASS_ASSIGNMENT", "WEEK_NUMBER", "WEEKDAY", "TIME"])
    df = grouped["CSPL_RECEIVED_CALLS"].sum().reset_index()
    df[["CSPL_RECEIVED_CALLS"]] = df[["CSPL_RECEIVED_CALLS"]].astype(float)
    if scale == 'DAY':
        for assignment in assignments:
            print(assignment)
            df_assignment = df[df['ASS_ASSIGNMENT'] == assignment]
            for day in range(366):
                print(day)
                df_day = df_assignment[df_assignment['WEEK_NUMBER'] == int(day/7 + 1)]
                df_day = df_day[df_day['WEEKDAY'] == day % 7]
                plt.plot(df_day['TIME'], df_day["CSPL_RECEIVED_CALLS"])
            plt.savefig(os.path.join(out_path, scale+"_"+assignment+".jpg"))
            plt.clf()
    if scale == 'WEEK':
        grouped = df.groupby(["ASS_ASSIGNMENT", "WEEK_NUMBER", "WEEKDAY"])
        df = grouped["CSPL_RECEIVED_CALLS"].mean().reset_index()
        for assignment in assignments:
            print(assignment)
            df_assignment = df[df['ASS_ASSIGNMENT'] == assignment]
            for week_number in range(53):
                df_week = df_assignment[df_assignment['WEEK_NUMBER'] == week_number]
                plt.plot(df_week['WEEKDAY'], df_week["CSPL_RECEIVED_CALLS"])
            plt.savefig(os.path.join(out_path, scale+"_"+assignment+".jpg"))
            plt.clf()
    if scale == 'YEAR':
        grouped = df.groupby(["ASS_ASSIGNMENT", "WEEK_NUMBER"])
        df = grouped["CSPL_RECEIVED_CALLS"].mean().reset_index()
        for assignment in assignments:
            print(assignment)
            df_assignment = df[df['ASS_ASSIGNMENT'] == assignment]
            plt.plot(df_assignment['WEEKDAY'], df_assignment["CSPL_RECEIVED_CALLS"])
        # plt.axis([0, 52, 0, 50])
        plt.savefig(os.path.join(out_path, scale+"_absolute_values.jpg"))
        plt.clf()


if __name__ == "__main__":
    # start = time.time()
    # df = parse(["DATE", "DAY_OFF", "ASS_ASSIGNMENT", "CSPL_RECEIVED_CALLS"])
    train_path = os.path.join(os.getcwd(), 'train.csv')
    visualization_path = os.path.join(os.getcwd(), 'visualization')
    compare_calls("WEEK", visualization_path, assignments="Téléphonie")

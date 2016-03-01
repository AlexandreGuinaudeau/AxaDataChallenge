# Visualize the main trends: daily, weekly and yearly.
import os
import pandas as pd
import matplotlib.pyplot as plt

from configuration import CONFIG
from utils import load_train_df
from learning.feature_engineering import FeatureFactory


def parse(df, remove_day_off=True):
    """
    Updates the dataframe by replacing the date by (year, month, day, day of the week, week of the year, time of the day

    Parameters
    ==========
    df: The input dataframe
    remove_day_off: Whether days off should be removed
    """
    # date_format='%Y-%m-%d %H:%M:%S.%f'
    df = df[df["ASS_ASSIGNMENT"].isin(CONFIG.submission_assignments)]
    if remove_day_off:
        df = df[df["DAY_OFF"] == 0]
        df.drop('DAY_OFF', axis=1, inplace=True)
    grouped = df.groupby(["ASS_ASSIGNMENT", "DATE", "DAY_OFF"])
    print("Grouped")
    df = grouped["CSPL_RECEIVED_CALLS"].sum().reset_index()
    print("Summed")
    df[["CSPL_RECEIVED_CALLS"]] = df[["CSPL_RECEIVED_CALLS"]].astype(float)
    return df


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
    if assignments is not None:
        if isinstance(assignments, str):
            assignments = [assignments]
        assert not set(assignments).difference(CONFIG.submission_assignments)
    else:
        assignments = CONFIG.submission_assignments

    df = load_train_df(CONFIG.preprocessed_train_path)
    df = df[df["ASS_ASSIGNMENT"].isin(assignments)]
    # if remove_days_off:
    #     df = df[df["DAY_OFF"] == 0]
    #     df.drop("DAY_OFF", axis=1, inplace=True)
    ff = FeatureFactory(df)
    for column in ["WEEK_NUMBER", "WEEK_DAY", "TIME"]:
        ff(column)
    df = ff.X

    if scale == 'DAY':
        for assignment in assignments:
            print(assignment)
            df_assignment = df[df['ASS_ASSIGNMENT'] == assignment]
            for day in range(366):
                df_day = df_assignment[df_assignment['WEEK_NUMBER'] == int(day/7 + 1)]
                df_day = df_day[df_day['WEEK_DAY'] == day % 7]
                plt.plot(df_day['TIME'], df_day["CSPL_RECEIVED_CALLS"])
            plt.savefig(os.path.join(out_path, scale+"_"+assignment+".jpg"))
            plt.clf()
    if scale == 'WEEK':
        grouped = df.groupby(["ASS_ASSIGNMENT", "WEEK_NUMBER", "WEEK_DAY"])
        df = grouped["CSPL_RECEIVED_CALLS"].mean().reset_index()
        for assignment in assignments:
            print(assignment)
            df_assignment = df[df['ASS_ASSIGNMENT'] == assignment]
            for week_number in range(53):
                df_week = df_assignment[df_assignment['WEEK_NUMBER'] == week_number]
                plt.plot(df_week['WEEK_DAY'], df_week["CSPL_RECEIVED_CALLS"])
            plt.savefig(os.path.join(out_path, scale+"_"+assignment+".jpg"))
            plt.clf()
    if scale == 'YEAR':
        grouped = df.groupby(["ASS_ASSIGNMENT", "WEEK_NUMBER"])
        df = grouped["CSPL_RECEIVED_CALLS"].mean().reset_index()
        for assignment in assignments:
            print(assignment)
            df_assignment = df[df['ASS_ASSIGNMENT'] == assignment]
            plt.plot(df_assignment['WEEK_DAY'], df_assignment["CSPL_RECEIVED_CALLS"])
        # plt.axis([0, 52, 0, 50])
        plt.savefig(os.path.join(out_path, scale+"_absolute_values.jpg"))
        plt.clf()


if __name__ == "__main__":
    # import time
    # start = time.time()
    # df = pd.read_csv(CONFIG.raw_train_path, sep=";",
    #                  usecols=["DATE", "DAY_OFF", "ASS_ASSIGNMENT", "CSPL_RECEIVED_CALLS"],
    #                  parse_dates=[0])
    # print("CSV file read in %i s" % (time.time() - start))
    # df = parse(df)
    # df.to_csv(CONFIG.preprocessed_train_path)
    # print("Dataframe parsed in %i s" % (time.time() - start))
    visualization_path = os.path.join(os.getcwd(), 'visualization')
    compare_calls("WEEK", visualization_path, assignments='Tech. Axa')

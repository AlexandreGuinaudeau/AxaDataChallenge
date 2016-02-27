# Visualize the main trends: daily, weekly and yearly.
import os
import pandas as pd
import matplotlib.pyplot as plt

from configuration import CONFIG
from utils import load_train_df


TRAIN_PATH = os.path.join(CONFIG.preprocessed_data_path, "train.csv")


def parse(df, remove_day_off=True):
    """
    Updates the dataframe by replacing the date by (year, month, day, day of the week, week of the year, time of the day

    Parameters
    ==========
    df: The input dataframe
    remove_day_off: Whether days off should be removed
    """
    # date_format='%Y-%m-%d %H:%M:%S.%f'
    if remove_day_off:
        df = df[df["DAY_OFF"] == 0]
        df.drop('DAY_OFF', axis=1, inplace=True)
    df['TIME'] = df['DATE'].apply(lambda d: d.hour + float(d.minute)/60)
    print("Added TIME")
    df['YEAR'] = df['DATE'].apply(lambda d: d.year)
    print("Added YEAR")
    df['MONTH'] = df['DATE'].apply(lambda d: d.month)
    print("Added MONTH")
    df['DAY'] = df['DATE'].apply(lambda d: d.day)
    print("Added DAY")
    df['WEEK_NUMBER'] = df['DATE'].apply(lambda d: d.isocalendar()[1])
    print("Added WEEK_NUMBER")
    df['WEEKDAY'] = df['DATE'].apply(lambda d: d.isocalendar()[2])
    print("Added WEEKDAY")
    df.drop('DATE', axis=1, inplace=True)
    if remove_day_off:
        grouped = df.groupby(["ASS_ASSIGNMENT", "YEAR", "MONTH", "DAY", "WEEK_NUMBER", "WEEKDAY", "TIME"])
    else:
        grouped = df.groupby(["ASS_ASSIGNMENT", "YEAR", "MONTH", "DAY", "WEEK_NUMBER", "WEEKDAY", "DAY_OFF", "TIME"])
    print("Grouped")
    df = grouped["CSPL_RECEIVED_CALLS"].sum().reset_index()
    print("Summed")
    df[["CSPL_RECEIVED_CALLS"]] = df[["CSPL_RECEIVED_CALLS"]].astype(float)
    df = df[df["ASS_ASSIGNMENT"].isin(CONFIG.relevant_assignments)]
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
    df = load_train_df(TRAIN_PATH)
    if assignments is not None:
        if isinstance(assignments, str):
            assignments = [assignments]
    else:
        assignments = CONFIG.relevant_assignments
    df = df[df["ASS_ASSIGNMENT"].isin(assignments)]
    if remove_days_off:
        df = df[df["DAY_OFF"] == 0]
    df.drop("DAY_OFF", axis=1, inplace=True)

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
    # import time
    # start = time.time()
    # df = pd.read_csv(CONFIG.train_path, sep=";", usecols=["DATE", "DAY_OFF", "ASS_ASSIGNMENT", "CSPL_RECEIVED_CALLS"],
    #                  parse_dates=[0])
    # print("CSV file read in %i s" % (time.time() - start))
    # df = parse(df)
    # df.to_csv(TRAIN_PATH)
    # print("Dataframe parsed in %i s" % (time.time() - start))
    visualization_path = os.path.join(os.getcwd(), 'visualization')
    compare_calls("WEEK", visualization_path, assignments='Téléphonie')

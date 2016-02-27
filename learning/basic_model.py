"""
Here, we simply predict using the mean of the values in the previous days, weeks or year.
"""
from datetime import date
from configuration import CONFIG
from visualization.trends import parse, load_df


def last_year(d):
    """
    Return a date that's 1 year before the date (or datetime)
    object `d`. Return the same calendar date (month and day) in the
    destination year, if it exists, otherwise use the following day
    (thus changing February 29 to March 1).
    """
    try:
        return d.replace(year=d.year - 1)
    except ValueError:
        return d + (date(d.year - 1, 1, 1) - date(d.year, 1, 1))


def predict(df):
    # df['DATE'] = df['DATE'].apply(last_year)
    df['DAY_OFF'] = 0
    df['CSPL_RECEIVED_CALLS'] = 0
    parsed_df = parse(df.copy())
    parsed_df = parsed_df[parsed_df["ASS_ASSIGNMENT"].isin(CONFIG.relevant_assignments)]
    train_df = load_df()
    train_df.drop("WEEK_NUMBER", axis=1, inplace=True)
    train_df = train_df.groupby(["ASS_ASSIGNMENT", "WEEKDAY", "TIME"])["CSPL_RECEIVED_CALLS"].mean().reset_index()
    train_df.set_index(['ASS_ASSIGNMENT', 'WEEKDAY', 'TIME'], inplace=True)
    prediction = None
    for week_number in sorted(set(parsed_df['WEEK_NUMBER'])):
        joined_df = parsed_df[parsed_df['WEEK_NUMBER'] == week_number]
        joined_df = joined_df[['ASS_ASSIGNMENT', 'WEEKDAY', 'TIME', 'CSPL_RECEIVED_CALLS']]
        joined_df.set_index(['ASS_ASSIGNMENT', 'WEEKDAY', 'TIME'], inplace=True)
        joined_df = joined_df.join(train_df, how='left', lsuffix="_empty")
        if prediction is None:
            prediction = [int(i+0.5) for i in joined_df['CSPL_RECEIVED_CALLS']]
        else:
            prediction.extend([int(i+0.5) for i in joined_df['CSPL_RECEIVED_CALLS']])
    return list(prediction)

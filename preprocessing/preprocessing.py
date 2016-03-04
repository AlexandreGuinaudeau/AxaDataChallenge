"""
reads train data and exports filtered data into the output file
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta as td
from configuration import CONFIG
from utils import load_train_df, load_means_df, load_weather_df
from learning import FeatureFactory


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def parse_train_as_df(in_path, out_path, useful_cols=None, verbose=0):
    """
    Removes from the file useless lines and columns.
    Computation time: 35s.
    """
    # date_format='%Y-%m-%d %H:%M:%S.%f'
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    if useful_cols is None:
        useful_cols = ["DATE", "DAY_OFF", "ASS_ASSIGNMENT", "CSPL_RECEIVED_CALLS", 'CSPL_INCOMPLETE']
    logger.debug('Reading csv file...')
    df = pd.read_csv(in_path,
                     sep=";",
                     usecols=useful_cols,
                     parse_dates=[0])

    logger.debug('Keeping useful assignments...')
    df = df[df["ASS_ASSIGNMENT"].isin(CONFIG.submission_assignments)]
    useful_cols.remove('CSPL_RECEIVED_CALLS')

    logger.debug('Removing weekends and days off...')
    # df = df[df["DATE"].apply(lambda x: x.isoweekday() < 6)]
    df = df[df["DAY_OFF"] == 0]
    df.drop('DAY_OFF', axis=1, inplace=True)
    useful_cols.remove('DAY_OFF')

    logger.debug('Removing incomplete data...')
    # df = df[df["DATE"].apply(lambda x: x.isoweekday() < 6)]
    df = df[df["CSPL_INCOMPLETE"] == 0]
    df.drop('CSPL_INCOMPLETE', axis=1, inplace=True)
    useful_cols.remove('CSPL_INCOMPLETE')

    logger.debug('Grouping and summing calls...')
    grouped = df.groupby(useful_cols)
    df = grouped["CSPL_RECEIVED_CALLS"].sum().reset_index()
    df[["CSPL_RECEIVED_CALLS"]] = df[["CSPL_RECEIVED_CALLS"]].astype(float)
    df.to_csv(out_path)
    logger.debug('Done.')


def complete_data_with_zeros(in_path, out_path=None):
    logger.debug('Loading Dataframe...')
    train_df = load_train_df(in_path)

    logger.debug('Generating empty Dataframe...')
    dates = sorted(set(train_df['DATE']))
    zero_date_df = pd.DataFrame({'DATE': dates})
    zero_date_df['ZERO'] = 0
    zero_ass_df = pd.DataFrame({'ASS_ASSIGNMENT': CONFIG.submission_assignments})
    zero_ass_df['ZERO'] = 0
    zero_df = zero_date_df.merge(zero_ass_df, how='left', on='ZERO')

    logger.debug('Completing Dataframe...')
    train_df = zero_df.merge(train_df, how='left', on=['DATE', 'ASS_ASSIGNMENT'])
    train_df['CSPL_RECEIVED_CALLS'].fillna(0, inplace=True)
    train_df.drop('ZERO', axis=1, inplace=True)

    logger.debug('Saving Dataframe...')
    if out_path is not None:
        train_df.to_csv(out_path)
    return train_df


def complete_with_means(df, out_path=None):
    logger.debug('Loading means Dataframe...')
    means = load_means_df(CONFIG.means_path)
    means.set_index(['ASS_ASSIGNMENT', 'TIME', 'WEEKEND'])

    logger.debug('Generating features...')
    ff = FeatureFactory(df.copy())
    ff('TIME')
    ff('WEEKEND')

    logger.debug('Joining Dataframes...')
    df['MEAN'] = pd.merge(ff.X, means, how='left', on=['ASS_ASSIGNMENT', 'TIME', 'WEEKEND'])['MEAN']
    logger.debug('Saving Dataframe...')
    if out_path is not None:
        df.to_csv(out_path, sep='\t', encoding='utf-8', index=None)
    return df


def parse_meteo_as_df(in_path, out_path=None, useful_cols=None, verbose=0):
    """
    Parsing meteo file
    """
    # print('Reading meteo...')
    if useful_cols is None:
        useful_cols = ['DATE', 'DEPT', 'MIN_TEMP', 'PRECIP']  # date, temperature_min, precipitations
    header_row = ['DATE', 'DEPT', 'CITY', 'MIN_TEMP', 'MAX_TEMP', 'WIND_DIR', 'PRECIP', 'HPA']

    df = pd.read_csv(in_path,
                     sep=",",
                     usecols=useful_cols,
                     parse_dates=[0],
                     date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M'),
                     names=header_row)
    # print('Done.')

    # print('Removing departments without digits...')
    df=df[df['DEPT'].str.isdigit()]
    # print('Done.')

    # print('Removing departments equal to 00...')
    df=df[df['DEPT']!='00']
    # print('Done.')

    # print('Formating dates.')
    df['DATE']=df.DATE.map(lambda x: x.date())
    # print('Dates formated.')

    if out_path is not None:
        df.to_csv(out_path)
    return df

# Number of departments where it has rained and where it has frozen
def preprocess_meteo1(df):
    grouped=df.groupby(['DATE','DEPT'])
    df1=grouped.agg({'MIN_TEMP':np.min,'PRECIP':np.max}).reset_index()
    df2=df1.groupby('DATE').agg({'MIN_TEMP':lambda x: pd.Series([(x <= 0).sum()]),'PRECIP':lambda x: pd.Series([(x > 0).sum()])})
    df2=df2.rename(columns = {'MIN_TEMP':'NUMB_FROZEN_DEPT'})
    df2=df2.rename(columns = {'PRECIP':'NUMB_WET_DEPT'})
    df2.to_csv(CONFIG.preprocessed_meteo1_path)
    print('meteo1 head:')
    print(df2.head())

# Average amount of rain and average lowest temperatures in each department.
def preprocess_meteo2(df):
    grouped=df.groupby(['DATE','DEPT'])
    df1=grouped.agg({'MIN_TEMP':np.mean,'PRECIP':np.mean}).reset_index()
    df1.to_csv(CONFIG.preprocessed_meteo2_path)
    print('meteo2 head:')
    print(df1.head())


# Booleans for each department where it has rained and where it has frozen
def preprocess_meteo3(df):
    grouped=df.groupby(['DATE','DEPT'])
    df1=grouped.agg({'MIN_TEMP':np.mean,'PRECIP':np.mean}).reset_index()
    df1['PRECIP']=df1['PRECIP'].apply(lambda x: x>0)
    df1['MIN_TEMP']=df1['MIN_TEMP'].apply(lambda x: x<0)
    df1.to_csv(CONFIG.preprocessed_meteo3_path)
    print('meteo3 head:')
    print(df1.head())

# Booleans for each department where average amount of rain is above 1mm and where it has frozen.
def preprocess_meteo4(df):
    grouped=df.groupby(['DATE','DEPT'])
    df1=grouped.agg({'MIN_TEMP':np.mean,'PRECIP':np.mean}).reset_index()
    df1['PRECIP']=df1['PRECIP'].apply(lambda x: x>0)
    df1['MIN_TEMP']=df1['MIN_TEMP'].apply(lambda x: x<0)
    df1.to_csv(CONFIG.preprocessed_meteo4_path)
    print('meteo4 head:')
    print(df1.head())

def complete_meteo_with_zeros(in_path, out_path=None):
    logger.debug('Loading Dataframe...')
    weather_df = load_weather_df(in_path)
    means = weather_df.mean()
    weather_df.reset_index(inplace=True)

    logger.debug('Generating empty Dataframe...')
    d1 = date(2011, 1, 1)
    d2 = date(2012, 12, 31)
    dates = [d1]
    while d1 < d2:
        d1 += td(days=1)
        dates.append(d1)
    zero_df = pd.DataFrame({'DATE': dates}, dtype=weather_df.dtypes['DATE'])

    logger.debug('Completing Dataframe...')
    weather_df = zero_df.merge(weather_df.copy(), how='left', on='DATE')
    weather_df['NUMB_FROZEN_DEPT'].fillna(means['NUMB_FROZEN_DEPT'], inplace=True)
    weather_df['NUMB_WET_DEPT'].fillna(means['NUMB_WET_DEPT'], inplace=True)
    weather_df.set_index('DATE', inplace=True)

    logger.debug('Saving Dataframe...')
    if out_path is not None:
        weather_df.to_csv(out_path)
    return weather_df


def create_means(grouped_train):
    means = {}
    for name, group in grouped_train:
        means[name] = group['CSPL_RECEIVED_CALLS'].mean()
    with open(os.path.join(CONFIG.preprocessed_data_path, "means.csv"), "w", encoding='utf-8') as out_f:
        out_f.write(",".join(["ASS_ASSIGNMENT", "TIME", "WEEKEND", "MEAN"]) + "\n")
        for name in sorted(means.keys()):
            ass, time, wd = name
            out_f.write(",".join([str(i) for i in [ass, time, wd, means[name]]]) + "\n")


def run(train_or_meteo=None, train_cols=None, meteo_cols=None, verbose=0):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def run(train_or_meteo=None, train_cols=None, meteo_cols=None, verbose=0):
    if train_or_meteo is None or train_or_meteo == 'train':
        parse_train_as_df(CONFIG.raw_train_path, CONFIG.preprocessed_train_path, useful_cols=train_cols,
                          verbose=verbose)
        logger.info('Saved train in csv file.')
        complete_data_with_zeros(CONFIG.preprocessed_train_path, CONFIG.preprocessed_train_path_zeros)
        logger.info('Saved completed train in csv file.')
        complete_with_means(load_train_df(CONFIG.preprocessed_train_path_zeros), CONFIG.means_path)
        logger.info('Saved train with means in csv file.')

    if train_or_meteo is None or train_or_meteo == 'meteo':
        print('Reading meteo file 1...')
        df1 = parse_meteo_as_df(CONFIG.raw_meteo_path1)
        print('Reading meteo file 2...')
        df2 = parse_meteo_as_df(CONFIG.raw_meteo_path2)
        print('Concatenating meteo files...')
        df = pd.concat([df1, df2])
        print('Meteo files concatenated. Running preprocessing...')

        # Number of departments where it has rained and where it has frozen
        print('Meteo1...')
        preprocess_meteo1(df)

        # Average amount of rain and average lowest temperatures in each department.
        print('Meteo2...')
        preprocess_meteo2(df)

        # Booleans for each department where it has rained and where it has frozen
        print('Meteo3...')
        preprocess_meteo3(df)

        # Booleans for each department where average amount of rain is above 1mm and where it has frozen.
        print('Meteo4...')
        preprocess_meteo4(df)

        return df

        # logger.debug('Meteo dataframes concatenated.')
        # logger.debug('Summing departments...')
        # df = df.groupby('DATE').agg({'MIN_TEMP': lambda x: pd.Series([(x <= 2).sum()]),
        #                                'PRECIP': lambda x: pd.Series([(x > 1).sum()])})
        # df = df.rename(columns={'MIN_TEMP': 'NUMB_FROZEN_DEPT'})
        # df = df.rename(columns={'PRECIP': 'NUMB_WET_DEPT'})
        # df[["NUMB_FROZEN_DEPT", 'NUMB_WET_DEPT']] = df[["NUMB_FROZEN_DEPT", "NUMB_WET_DEPT"]].astype(int)
        # df.to_csv(CONFIG.preprocessed_meteo_path)
        # logger.info('Saved meteo in csv file.')


if __name__ == "__main__":
    # parse_train_as_dict(CONFIG.raw_train_path, CONFIG.preprocessed_train_path)
    # parse_train_as_df(CONFIG.raw_train_path, CONFIG.preprocessed_train_path)
    # df1 = parse_meteo_as_df(CONFIG.raw_meteo_path1)
    # df2 = parse_meteo_as_df(CONFIG.raw_meteo_path2)
    # df = pd.concat([df1, df2])
    # print(df)
    # run('train', verbose=1)
    from utils import load_submission
    print(complete_with_means(load_submission(CONFIG.submission_path), CONFIG.submission_path_mean))
    pass

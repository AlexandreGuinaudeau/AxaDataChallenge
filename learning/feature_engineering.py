import inspect
from functools import partial
import time
from datetime import date
import pandas as pd
from scipy.sparse import spmatrix, csr_matrix
import numpy as np


def get_features(obj):
    return [i[0] for i in inspect.getmembers(obj) if not i[0].startswith("_")]


def column_method(name=None):
    def wrapper(func):
        def wrapped(*args):
            _lambda = partial(func, args[0])
            x = args[0].X
            if name is None:
                return x.apply(_lambda, axis='columns')
            col = args[0][name]
            return col.apply(_lambda)
        return wrapped
    return wrapper


class FeatureFactory:
    def __init__(self, x, weather_df=None):
        self.X = x
        self.assignments = {}
        self.weather_df = weather_df

    def __getitem__(self, item):
        item = item.upper()
        if item in self.features:
            return self.X[item]
        else:
            return self(item)

    @property
    def features(self):
        return list(self.X.columns)

    def __call__(self, name):
        lower_name = name.lower()
        name = name.upper()
        if name in self.features:
            return self.X[name]
        if lower_name not in get_features(self):
            raise NotImplementedError(lower_name)
        self.features.append(name)
        self.X[name] = getattr(self, lower_name)()
        return self.X[name]

    def select_features(self, features):
        wrong_features = set(features).difference(set(self.features))
        if wrong_features:
            raise KeyError(wrong_features)
        self.X = self.X.loc[:, features]
        return self.X

    def apply_weights(self, weights):
        for c in self.X.columns:
            if c in weights.keys():
                self.X[c].apply(lambda x: x*weights[c])

    def get_sparse_matrix(self, col):
        column = self[col]
        if col in ['WEEK_DAY', 'WEEK_DAY_NAME']:
            col_names = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
        elif col == 'TIME':
            col_names = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
                         9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0,
                         17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5]
        else:
            col_names = sorted(set(column))
        columns = []
        for c in col_names:
            try:
                columns.append(column == c)
            except TypeError as e:
                print(c)
                print(column)
                raise e
        df = pd.DataFrame(columns).transpose()
        return csr_matrix(df, dtype=np.int)

    @column_method('DATE')
    def year(self, x):
        return x.year

    @column_method('DATE')
    def month(self, x):
        return x.month

    @column_method('DATE')
    def day(self, x):
        return x.day

    @column_method('DATE')
    def cum_days(self, x):
        return int((x.date() - date(x.year, 1, 1)).days/10)

    @column_method('DATE')
    def full_date(self, x):
        return x.date()

    @column_method('DATE')
    def time(self, x):
        return x.hour + float(x.minute)/60

    @column_method('DATE')
    def week_day(self, x):
        return x.isocalendar()[2]

    @column_method('DATE')
    def week_day_name(self, x):
        return x.date().strftime("%A")

    @column_method('WEEK_DAY')
    def weekend(self, x):
        if x <= 5:
            return 0
        if x == 6:
            return 1
        return 2

    @column_method('DATE')
    def week_number(self, x):
        return x.isocalendar()[1]

    @column_method('ASS_ASSIGNMENT')
    def assignment(self, x):
        if x not in self.assignments.keys():
            self.assignments[x] = len(self.assignments)
        return self.assignments[x]

    def numb_frozen_dept(self):
        if 'FULL_DATE' not in self.features:
            self('FULL_DATE')
        self.X['FULL_DATE'] = pd.to_datetime(self['FULL_DATE'])
        df = pd.merge(self.X, self.weather_df, how='left', left_on='FULL_DATE', right_on='DATE')
        return df['NUMB_FROZEN_DEPT']

    def numb_wet_dept(self):
        if 'FULL_DATE' not in self.features:
            self('FULL_DATE')
        return pd.merge(self.X, self.weather_df, how='left', left_on='FULL_DATE', right_on='DATE')['NUMB_WET_DEPT']

    @column_method()  # Functions are 9 times more efficient when applied directly on the column X['DATE']
    def slow_example(self, x):
        return x['DATE'].hour + float(x['DATE'].minute)/60

import inspect
from functools import partial
import time
import pandas as pd


def get_features(obj):
    return [i[0] for i in inspect.getmembers(obj) if not i[0].startswith("_")]


def column_method(name=None):
    def wrapper(func):
        def wrapped(*args):
            _lambda = partial(func, args[0])
            x = args[0].X
            if name is None:
                return x.apply(_lambda, axis='columns')
            if name not in x.columns:
                raise KeyError(name)
            return x[name].apply(_lambda)
        return wrapped
    return wrapper


class FeatureFactory:
    def __init__(self, x):
        self.X = x
        self.assignments = {}

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
            print("INFO: %s is already a column of X." % name)
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
    def full_date(self, x):
        return x.date()

    @column_method('DATE')
    def time(self, x):
        return x.hour + float(x.minute)/60

    @column_method('DATE')
    def week_day(self, x):
        return x.isocalendar()[2]

    @column_method('DATE')
    def week_number(self, x):
        return x.isocalendar()[1]

    @column_method('ASS_ASSIGNMENT')
    def assignment(self, x):
        if x not in self.assignments.keys():
            self.assignments[x] = len(self.assignments)
        return self.assignments[x]

    @column_method()  # Functions are 9 times more efficient when applied directly on the column X['DATE']
    def slow_example(self, x):
        return x['DATE'].hour + float(x['DATE'].minute)/60


if __name__ == "__main__":
    from configuration import CONFIG
    from utils import load_train_df

    start = time.time()
    df = load_train_df(CONFIG.preprocessed_train_path)
    print("Dataframe loaded in %i seconds" % (time.time() - start))
    ff = FeatureFactory(df)
    for feature in {'YEAR', 'WEEK_NUMBER', 'WEEK_DAY', 'DAY', 'MONTH', 'TIME', 'FULL_DATE'}:
        # Features are created in roughly 8 seconds.
        ff(feature)
    ff.select_features(["ASS_ASSIGNMENT", "DATE", "YEAR", "MONTH", "DAY", "WEEK_NUMBER", "WEEK_DAY", "DAY_OFF", "TIME",
                        'FULL_DATE'])
    print(ff.features)

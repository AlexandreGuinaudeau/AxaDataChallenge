import time
import numpy as np
from sklearn.cross_validation import BaseShuffleSplit, check_random_state, check_array, indexable, _num_samples, chain, \
    safe_indexing
import warnings


class DateShuffleSplit(BaseShuffleSplit):
    """
    Like the ShuffleSplit, except we shuffle dates, and then select all the data corresponding to those dates.
    """

    def __init__(self, dates, n_iter=10, test_size=0.1, random_state=None):
        super(DateShuffleSplit, self).__init__(len(set(dates)), n_iter=n_iter, test_size=test_size,
                                               random_state=random_state)
        self.unique_dates = dict(zip(list(range(len(set(dates)))), sorted(set(dates))))
        self.dates = dates

    def _iter_indices(self):
        rng = check_random_state(self.random_state)

        # random partition
        permutation = rng.permutation(self.n)
        for i in range(self.n_iter):
            ind_test = permutation[i*self.n_test:(i+1)*self.n_test]
            dates_test = [self.unique_dates[i] for i in ind_test]
            dates_train = [self.unique_dates[i] for i in range(self.n) if i not in ind_test]
            yield [i for i, d in enumerate(self.dates) if d in dates_train], \
                  [i for i, d in enumerate(self.dates) if d in dates_test]

    def __repr__(self):
        return ('%s(%d, n_iter=%d, test_size=%s, '
                'random_state=%s)' % (
                    self.__class__.__name__,
                    self.n,
                    self.n_iter,
                    str(self.test_size),
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter


def train_test_split(dates, *arrays, **options):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    dtype = options.pop('dtype', None)
    if dtype is not None:
        warnings.warn("dtype option is ignored and will be removed in 0.18.",
                      DeprecationWarning)

    allow_nd = options.pop('allow_nd', None)
    allow_lists = options.pop('allow_lists', None)

    if allow_lists is not None:
        warnings.warn("The allow_lists option is deprecated and will be "
                      "assumed True in 0.18 and removed.", DeprecationWarning)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))
    if allow_nd is not None:
        warnings.warn("The allow_nd option is deprecated and will be "
                      "assumed True in 0.18 and removed.", DeprecationWarning)
    if allow_lists is False or allow_nd is False:
        arrays = [check_array(x, 'csr', allow_nd=allow_nd,
                              force_all_finite=False, ensure_2d=False)
                  if x is not None else x
                  for x in arrays]

    if test_size is None and train_size is None:
        test_size = 0.25
    arrays = indexable(*arrays)
    assert len(dates) == _num_samples(arrays[0]), "There should be as many dates as input samples."
    cv = DateShuffleSplit(dates, test_size=test_size, random_state=random_state)
    train, test = next(iter(cv))
    return list(chain.from_iterable((safe_indexing(a, train),
                                     safe_indexing(a, test)) for a in arrays))


train_test_split.__test__ = False  # to avoid a pb with nosetests

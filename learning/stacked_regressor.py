from sklearn.utils.validation import NotFittedError
from sklearn.linear_model.base import LinearModel, RegressorMixin, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ARDRegression, BayesianRidge
import numpy as np


class StackedRegression(LinearModel, RegressorMixin):
    def __init__(self, weights=None, cv_train_size=None):
        estimators = []
        estimators.append(KNeighborsRegressor(n_neighbors=10, weights='distance'))
        estimators.append(ARDRegression())
        # estimators.append(BayesianRidge())
        self.estimators = estimators
        self.stacker = LinearRegression()
        self.weights = weights if weights is not None else {}
        self.cv_train_size = cv_train_size if cv_train_size is not None else 0.7
        self._is_fitted = False

    def fit_stack(self, X, y):
        print('fitting')
        print(X.shape)
        n_train = int(X.shape[0] * self.cv_train_size)
        for estimator in self.estimators:
            estimator.fit(X[:n_train, :], y[:n_train])
        predictions = np.concatenate([np.matrix(estimator.predict(X[n_train:, :])).transpose()
                                      for estimator in self.estimators], axis=1)
        self.stacker.fit(predictions, y[n_train:])
        self._is_fitted = True
        print('fitted')
        print(self.stacker.residues_)

    def fit(self, X, y):
        if not self._is_fitted:
            raise NotFittedError('StackedRegression must call fit_stack before fit.')
        for estimator in self.estimators:
            estimator.fit(X, y)
        self._is_fitted = True

    def predict(self, X):
        predictions = np.concatenate([np.matrix(estimator.predict(X)).transpose()
                                      for estimator in self.estimators], axis=1)
        return self.stacker.predict(predictions)

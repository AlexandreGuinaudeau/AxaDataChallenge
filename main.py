import preprocessing
import learning
from utils import load_submission, load_train_df
from configuration import CONFIG


if __name__ == "__main__":
    # Create train and meteo preprocessed files
    preprocessing.run()
    # Define a model
    from sklearn.neighbors import KNeighborsRegressor

    _df = load_train_df(CONFIG.preprocessed_train_path)
    _submission_df = load_submission()
    _estimator = KNeighborsRegressor(n_neighbors=4, weights='distance')
    # estimator = LogisticRegression()
    _scoring = 'mean_squared_error'
    _k_fold = 3
    _n_jobs = 3
    _verbose = 0
    _fit_params = None
    _cols = ["YEAR", "WEEK_NUMBER", "WEEK_DAY", "TIME"]
    _weights = [1, 1, 1, 0.1]

    # Test the model
    print(learning.cross_val_score(_estimator, _cols, _k_fold, _weights, _scoring, _n_jobs, _verbose, _fit_params,
                                   chunksize=100000))

    # Create the corresponding submission file
    learning.create_submission_file(_estimator, _cols, weights=_weights)

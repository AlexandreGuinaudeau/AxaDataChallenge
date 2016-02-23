# Visualize the main trends: daily, weekly and yearly.
import pandas as pd
from configuration import CONFIG
import time


if __name__ == "__main__":
    start = time.time()
    for chunk in pd.read_csv(CONFIG.train_file, sep=";", chunksize=CONFIG.chunksize):
        print(chunk)
    print(time.time() - start)

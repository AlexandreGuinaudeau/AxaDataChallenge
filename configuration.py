import os


class _Config:
    def __init__(self):
        self.data_path = os.path.realpath(os.path.join(__file__, "..", "data"))
        self.train_file = os.path.join(self.data_path, "train_2011_2012.csv")
        self.meteo_file1 = os.path.join(self.data_path, "meteo_2011.csv")
        self.meteo_file2 = os.path.join(self.data_path, "meteo_2012.csv")

        self.chunksize = 10 ** 6


CONFIG = _Config()

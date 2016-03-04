import os
from datetime import date


class _Config:
    def __init__(self):
        self.root_path = os.path.realpath(os.path.join(__file__, ".."))

        self.raw_data_path = os.path.realpath(os.path.join(self.root_path, "raw_data"))
        self.raw_train_path = os.path.join(self.raw_data_path, "train_2011_2012.csv")
        self.raw_meteo_path1 = os.path.join(self.raw_data_path, "meteo_2011.csv")
        self.raw_meteo_path2 = os.path.join(self.raw_data_path, "meteo_2012.csv")

        self.preprocessed_data_path = os.path.join(self.root_path, "data")
        self.preprocessed_train_path = os.path.join(self.preprocessed_data_path, "train.csv")
        self.preprocessed_meteo_path = os.path.join(self.preprocessed_data_path, "meteo.csv")
        self.preprocessed_meteo1_path = os.path.join(self.preprocessed_data_path, "meteo1.csv")
        self.preprocessed_meteo2_path = os.path.join(self.preprocessed_data_path, "meteo2.csv")
        self.preprocessed_meteo3_path = os.path.join(self.preprocessed_data_path, "meteo3.csv")
        self.preprocessed_meteo4_path = os.path.join(self.preprocessed_data_path, "meteo4.csv")

        self.results_path = os.path.join(self.root_path, "results")
        self.submission_path = os.path.join(self.results_path, "empty_submission.txt")

        self.submission_assignments = ['CAT', 'CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique',
                                       'Gestion Amex', 'Gestion Assurances', 'Gestion Clients', 'Gestion DZ',
                                       'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Manager',
                                       'Mécanicien', 'Médical', 'Nuit', 'Prestataires', 'RENAULT', 'RTC',
                                       'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter',
                                       'Tech. Total', 'Téléphonie']
        self.submission_dates = [date(2012, 1, 3), date(2012, 2, 8), date(2012, 3, 12), date(2012, 4, 16),
                                 date(2012, 5, 19), date(2012, 6, 18), date(2012, 7, 22), date(2012, 8, 21),
                                 date(2012, 9, 20), date(2012, 10, 24), date(2012, 11, 26), date(2012, 12, 28)]


CONFIG = _Config()

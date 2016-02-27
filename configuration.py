import os


class _Config:
    def __init__(self):
        self.root_path = os.path.realpath(os.path.join(__file__, ".."))

        self.data_path = os.path.realpath(os.path.join(self.root_path, "data"))
        self.train_path = os.path.join(self.data_path, "train_2011_2012.csv")
        self.meteo_path1 = os.path.join(self.data_path, "meteo_2011.csv")
        self.meteo_path2 = os.path.join(self.data_path, "meteo_2012.csv")

        self.results_path = os.path.join(self.root_path, "results")
        self.submission_path = os.path.join(self.results_path, "empty_submission.txt")

        self.relevant_assignments = ['CAT', 'CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique',
                                     'Gestion Amex', 'Gestion Assurances', 'Gestion Clients', 'Gestion DZ',
                                     'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', 'Manager', 'Mécanicien',
                                     'Médical', 'Nuit', 'Prestataires', 'RENAULT', 'RTC', 'Regulation Medicale', 'SAP',
                                     'Services', 'Tech. Axa', 'Tech. Inter', 'Tech. Total', 'Téléphonie']


CONFIG = _Config()

"""
reads train data and exports filtered data into the output file
"""

import csv
import pandas as pd
#from configuration import CONFIG


def parse_as_df(in_path, out_path, useful_cols=None, remove_days_off=True):
    """
    Removes from the file useless lines and columns.
    Computation time: 35s.
    """
    # date_format='%Y-%m-%d %H:%M:%S.%f'
    if useful_cols is None:
        useful_cols = ["DATE", "DAY_OFF", "SPLIT_COD", "ACD_COD", "ASS_ASSIGNMENT", "CSPL_RECEIVED_CALLS"]
    df = pd.read_csv(in_path,
                     sep=";",
                     usecols=useful_cols,
                     parse_dates=[0])
    df = df[df["ASS_ASSIGNMENT"].isin(CONFIG.submission_assignments)]
    useful_cols.remove('CSPL_RECEIVED_CALLS')
    if remove_days_off:
        df = df[df["DAY_OFF"] == 0]
        df.drop('DAY_OFF', axis=1, inplace=True)
        useful_cols.remove('DAY_OFF')
    grouped = df.groupby(useful_cols)
    df = grouped["CSPL_RECEIVED_CALLS"].sum().reset_index()
    df[["CSPL_RECEIVED_CALLS"]] = df[["CSPL_RECEIVED_CALLS"]].astype(float)
    df.to_csv(out_path)

def parse_meteo_as_df(in_path, out_path, useful_cols=None):
    """
    Parsing meteo file
    """
    #date_format='%Y-%m-%d %H:%M:%S.%f'
    if useful_cols is None:
        useful_cols =['DATE','MIN_TEMP','PRECIP'] #date, temperature_min, precipitations
        header_row=['DATE','DEPT','CITY','MIN_TEMP','MAX_TEMP','WIND_DIR','PRECIP','HPA'] #because the meteo file index is missing  :(
    df = pd.read_csv(in_path,
                     sep=",",
                     usecols=useful_cols,
                     parse_dates=[0],
                     names=header_row)
    
    grouped = df.groupby('DATE') #group by date
    df = grouped["MIN_TEMP",'PRECIP'].mean().reset_index()
    df[["MIN_TEMP",'PRECIP']] = df[["MIN_TEMP","PRECIP"]].astype(float)
    df.to_csv(out_path)

def parse_as_dict(in_path, out_path):
    # with open ('test.csv', 'rb') as csvfile :  #test on a small file
    with open(in_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        useful_cols = [0, 7, 8, 12, 81]  # 'DATE','SPLIT_COD', 'ACD_COD','ASS_ASSIGNMENT','CSPL_RECEIVED_CALLS'
        nbCallsperDate = {}
        # les infos utiles
        """for row in reader:
            content =  list(row[i] for i in useful_cols)
            print content"""
        # regroupement par date
        for row in reader:
            if row[0] != 'DATE':
                if row[0] not in nbCallsperDate:
                    # add key with value row[81]= received calls
                    nbCallsperDate[row[0]] = int(row[81])
                else:
                    # add nb of received calls for this date
                    nbCallsperDate[row[0]] += int(row[81])
        # write in a new csv
    with open(out_path, 'w') as csvoutput:
        writer = csv.writer(csvoutput, delimiter=';')
        for x in nbCallsperDate:
            writer.writerow([x, nbCallsperDate[x]])


if __name__ == "__main__":
    # parse_as_dict(CONFIG.raw_train_path, CONFIG.preprocessed_train_path)
    #parse_as_df(CONFIG.raw_train_path, CONFIG.preprocessed_train_path)
    #parse_meteo_as_df("test.csv","out_meteo.csv")
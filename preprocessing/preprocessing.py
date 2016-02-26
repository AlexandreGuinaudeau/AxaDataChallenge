"""
reads train data and exports dates/calls into date.csv
"""

import csv

#with open ('test.csv', 'rb') as csvfile :  #test on a small file
with open ('train_2011_2012.csv', 'rb') as csvfile :
    reader = csv.reader(csvfile, delimiter = ';', quotechar = '|')
    useful_cols = [0,7,8,12,81] # 'DATE','SPLIT_COD', 'ACD_COD','ASS_ASSIGNMENT','CSPL_RECEIVED_CALLS'
    nbCallsperDate = {}
    #les infos utiles    
    """for row in reader:
        content =  list(row[i] for i in useful_cols)
        print content"""
    #regroupement par date 
    for row in reader:
        if row[0]!= 'DATE' :        
            if row[0] not in nbCallsperDate:
                #add key wtih value row[81]= received calls
                nbCallsperDate[row[0]]=int (row[81])
            else :
                #add nb of received calls for this date
                nbCallsperDate[row[0]]+=int (row[81])
    #write in a new csv
with open ('dates.csv', 'wb') as csvoutput:
    writer = csv.writer(csvoutput,delimiter = ';')
    for x in nbCallsperDate:
        writer.writerow([x,nbCallsperDate[x]])
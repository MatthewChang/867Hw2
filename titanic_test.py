from numpy import *
from plotBoundary import *
from SVM import *
from LR import *

import csv

name = 'titanic'
train = 'data/data_'+name+'_train.csv'

A = None
with open(train, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        row = [float(x) for x in row]
        if(A is None):
            A = np.matrix(row)
        else:
            A = np.concatenate((A,np.matrix(row)),axis=0)


print A.shape

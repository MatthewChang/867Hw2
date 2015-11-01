from numpy import *
from plotBoundary import *
from SVM import *
from LR import *
from Descent import *
from scipy import optimize

import csv

name = 'titanic'
train = 'data/data_'+name+'_train.csv'
test_set = 'data/data_'+name+'_test.csv'
eval_set = 'data/data_'+name+'_validate.csv'

A = None
A_test = None
A_eval = None
with open(train, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        row = [float(x) for x in row]
        if(A is None):
            A = np.matrix(row)
        else:
            A = np.concatenate((A,np.matrix(row)),axis=0)
print A.shape

with open(test_set, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        row = [float(x) for x in row]
        if(A_test is None):
            A_test = np.matrix(row)
        else:
            A_test = np.concatenate((A_test,np.matrix(row)),axis=0)
print A_test.shape

with open(eval_set, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        row = [float(x) for x in row]
        if(A_eval is None):
            A_eval = np.matrix(row)
        else:
            A_eval = np.concatenate((A_eval,np.matrix(row)),axis=0)
print A_eval.shape

X = A[:,0:-1].copy()
Y = A[:,-1:].copy()

X *= 1.0/1
line = lambda x,y: x*y.T
k = line
C,b = SVM(X,Y,k,1)
def eval_point(x):
    v = 0
    for coeff,xi,yi in C:
        v += coeff*k(x,xi)
    return v-b


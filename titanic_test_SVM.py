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

Xt = A_test[:,0:-1].copy()
Yt = A_test[:,-1:].copy()

Xv = A_eval[:,0:-1].copy()
Yv = A_eval[:,-1:].copy()

line = lambda x,y: x*y.T
k = line

min_error = 1
optimal_classifier = None
optimal_C = None;



def error_rate(X,Y,SV,b):
    def eval_point(x):
        v = 0
        for coeff,xi,yi in SV:
            v += coeff*k(x,xi)
        return v-b

    cor = 0
    for r in range(0,X.shape[0]):
        val = Y[r]*eval_point(X[r,:])
        if(val > 0):
            cor += 1
    return (1-1.0*cor/X.shape[0])

for C in np.linspace(0.5,2,20):
    SV,b = SVM(X,Y,k,C)
    rate = error_rate(Xt,Yt,SV,b)
    print rate
    if(rate < min_error):
        min_error = rate
        optimal_classifier = (SV,b)
        optimal_C = C

final_rate = error_rate(Xv,Yv,optimal_classifier[0],optimal_classifier[1])

def compute_W(SV):
    w = np.zeros(SV[0][1].shape)
    for coeff,xi,yi in SV:
        w += coeff*xi
    return w
W = compute_W(optimal_classifier[0])

print "Optimal C Value:" + str(optimal_C)
print "W vector:" + str(W)
print "Validation Error Rate:" + str(error_rate(Xv,Yv,optimal_classifier[0],optimal_classifier[1]))

W = np.matrix(W)
W_norm = W/np.linalg.norm(W,1)
print "Weight distrobution of W: " + str(np.abs(W_norm))

#Parker: okay here are the results
'''
Optimal C Value:1.0
W vector:[[  2.61652153e-06   2.57148352e-06  -5.18800505e-06   2.00000011e+00
   -7.11311543e-09   1.62789734e-08   1.09828388e-07  -4.57625049e-11
   -2.53898291e-06  -2.39254531e-06   4.93152822e-06]]
Validation Error Rate:0.21
Weight distrobution of W: [[  1.30826069e-06   1.28574169e-06   2.59400238e-06   1.00000000e+00
    3.55655752e-09   8.13948623e-09   5.49141907e-08   2.28812512e-11
    1.26949138e-06   1.19627259e-06   2.46576397e-06]]
'''
    


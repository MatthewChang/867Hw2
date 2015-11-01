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

X = A[:,0:-1]
Y = A[:,-1:]
X *= 1.0/100 #scale down X because of overflow errors
    
Xt = A_test[:,0:-1]
Yt = A_test[:,-1:]
Xt *= 1.0/100 #scale down X because of overflow errors

Xv = A_eval[:,0:-1]
Yv = A_eval[:,-1:]
Xv *= 1.0/100 #scale down X because of overflow errors

min_error = None
optimal_y = -1
for y in np.linspace(0,0.001,10): #y = 0 is clearly the best
    w = np.array([0]*12)
    f = lambda w: ELR_array(X,Y,w,y);
    res = optimize.fmin_bfgs(f,w)
    
    w_test = np.matrix(res).T
    error = NLL(Xt,Yt,w_test)

    error = LR_class_rate(Xt,Yt,w_test)
    print error
    if(min_error is None or error < min_error):
       min_error = error;
       optimal_y = y;
       optimal_w = w_test;
       
print min_error
print optimal_y
print optimal_w

print "Validataion NLL Error:" + str(NLL(Xv,Yv,optimal_w))
print "Validation Error Rate:" + str(LR_class_rate(Xv,Yv,optimal_w))
print "Optimal W vector:" + str(optimal_w)

W = optimal_w
W_norm = W/np.linalg.norm(W,1)
print "Weight distrobution of W: " + str(np.abs(W_norm))


'''
Validataion NLL Error:236.657899149
Validation Error Rate:0.206
Optimal W vector:[[ -7.10944224e+01]
 [ -5.24424977e+01]
 [  1.23537314e+02]
 [ -2.83868808e+02]
 [  2.22966804e+00]
 [  1.46804869e+01]
 [ -2.07853843e+01]
 [ -1.66356550e-01]
 [  1.48842427e+01]
 [ -3.33000886e+01]
 [  1.84201380e+01]
 [  9.81340208e-01]]
1.0
Weight distrobution of W: [[  1.11715047e-01]
 [  8.24061285e-02]
 [  1.94121795e-01]
 [  4.46060552e-01]
 [  3.50361480e-03]
 [  2.30683537e-02]
 [  3.26613553e-02]
 [  2.61406299e-04]
 [  2.33885278e-02]
 [  5.23264815e-02]
 [  2.89446981e-02]
 [  1.54204035e-03]]
'''



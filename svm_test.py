from numpy import *
from plotBoundary import *
from SVM import *
from LR import *

# parameters
name = 'stdev1'
print '======Training======'
# load data from csv files
train = loadtxt('data/data_'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

W,b = SVM(X,Y,lambda x,y: x*y.T)
print W,b
# Carry out training, primal and/or dual
### TODO ###
# Define the predictSVM(x) function, which uses trained parameters
### TODO ###

def predictSVM(x):
    v = W*np.matrix(x).T - b
    return v
    

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


print '======Validation======'
# load data from csv files
validate = loadtxt('data/data_'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()

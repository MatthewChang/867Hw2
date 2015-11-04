from numpy import *
from plotBoundary import *
from SVM import *
from LR import *

# parameters
name = 'stdev2'
print '======Training======'
# load data from csv files
train = loadtxt('data/data_'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

line = lambda x,y: x*y.T
quad = lambda x,y: (x*y.T + 1)**2
cube = lambda x,y: (x*y.T + 1)**4

def norm_pdf(x,var):
    return exp(-x**2/(2*var))/sqrt(2*pi*var)

def gen_gaussian_kernel(var):
    def k(x,y):
        z = x-y
        #print ((x-y)*(x-y).T)[0,0]
        return exp(-z*z.T/(2*var))
    return k


k = gen_gaussian_kernel(9)
k= line
C,b = SVM(X,Y,k,1)

def eval_point(x):
    v = 0
    for coeff,xi,yi in C:
        v += coeff*k(x,xi)
    return v 

def predictSVM(x):
    return eval_point(x)-b
    

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

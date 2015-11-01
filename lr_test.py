from numpy import *
from plotBoundary import *
from LR import *
from Descent import *;
# import your LR training code

# parameters
#data = 'ls'
print '======Training======'
# load data from csv files
name = "stdev2"
train = loadtxt('data/data_'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]
w = np.matrix("0;0;0")
y = 100;
print X.shape
print Y.shape
print np.max(X)
print np.min(X)
print w.shape
f = lambda w: ELR(X,Y,w,y);
g = lambda x: num_gradient(f,x,0.0001);
res = descent(f,g,0.001,w,0.001);
print res
print f(res)

#prediction
def predictLR(x):
	labels = [-1,1];
	best = 0;
	lab = 0;
	for l in labels:
		val = NLL(np.matrix(x),l,res)
		if val> best:
			best = val
			lab = l;
	return lab	

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0,0.5,1], title = 'LR Train')

print '======Validation======'
# load data from csv files
validate = loadtxt('data/data_'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
pl.show();

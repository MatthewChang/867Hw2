import numpy as np

def phi(X):
	n = np.ones((X.shape[0],1))
	return np.concatenate((X,n),axis=1)

def NLL(X,Y,w):
	val = phi(X)*w;
	ex = np.multiply(Y,val);
	return np.sum(np.log(1+np.exp(ex)));

def LR_class_rate(X,Y,w):
        val = phi(X)*w;
	ex = np.multiply(Y,val);
	return 1-(1.0*np.sum(ex < 0)/X.shape[0])
	
def ELR(X,Y,w,y):
	return NLL(X,Y,w) + y*w.T*w;

	
def ELR_array(X,Y,w,y):
        w = np.matrix(w).T
	return ELR(X,Y,w,y);
	

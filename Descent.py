import numpy as np;
import math;
from scipy.optimize import minimize, rosen, rosen_der

def descent(f,g,step,guess,thresh):
    x = guess;
    previous = None;
    #print x
    while(previous is None or abs(f(previous)-f(x)) > thresh):
		previous = x;
        #print x
        #print g(x)
		#print f(x)
		x = x - step*g(x);
    return x;

def num_gradient(f,x,step):
    x = x.astype('float64');
    res = np.zeros(x.shape)
    for i in range(x.shape[0]):
        forward = np.matrix(x,copy=True);
        forward[i] = forward[i] + step/2.0;
        back = np.matrix(x,copy=True);
        back[i] = back[i]- step/2.0;
        res[i] = (f(forward)-f(back))/step;
    return np.matrix(res);
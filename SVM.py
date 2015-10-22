import numpy as np
from cvxopt import matrix, solvers

def SVM(X,Y,k):
        X = np.matrix(X)
        P = np.zeros((X.shape[0],X.shape[0]))

        for r in range(0,X.shape[0]):
                for c in range(0,X.shape[0]):
                        val = Y[r]*Y[c]*k(X[r],X[c])
                        P[r][c] = val
        P = matrix(P)
        q = -matrix(np.ones((X.shape[0],1)))
        G = -matrix(np.eye(X.shape[0]))
        h = matrix(np.zeros((X.shape[0],1)))
        A = matrix(Y.T)
        b = matrix(np.zeros((1,1)))

        solution = solvers.qp(P, q, G, h, A, b)
        #solution = solvers.qp(P, q, G, h)
        
        sol = solution['x']
        W = np.zeros(X[0].shape);
        
        for i in range(0,X.shape[0]):
                W += X[i]*sol[i]*Y[i][0]
        nsv = 0;
        b = 0;
        for i in range(0,X.shape[0]):
                if(sol[i] > 0.0001):
                        nsv += 1
                        b += W*X[i].T - Y[i][0]
        b = b/nsv
        print nsv
        return (W,b)

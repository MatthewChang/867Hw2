import numpy as np
from cvxopt import matrix, solvers

def SVM(X,Y,k):
        X = np.matrix(X)
        P = np.zeros((X.shape[0],X.shape[0]))

        for r in range(0,X.shape[0]):
                for c in range(0,X.shape[0]):
                        val = Y[r]*Y[c]*k(X[r],X[c])
                        P[r][c] = 0.5*val
        P = matrix(P)
        q = -matrix(np.ones((X.shape[0],1)))
        G = -matrix(np.eye(X.shape[0]))
        #print G
        h = matrix(np.zeros((X.shape[0],1)))
        npA = np.diag(np.array(Y.T[0]))
        A = matrix(npA)
        b = matrix(np.zeros((X.shape[0],1)))
        solution = solvers.qp(P, q, G, h, A, b)
        print solution
        print solution['x']
        return np.matrix(solution['x'])

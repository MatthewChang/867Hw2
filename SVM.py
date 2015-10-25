import numpy as np
from cvxopt import matrix, solvers

def SVM(X,Y,k,C):
        X = np.matrix(X)
        P = np.zeros((X.shape[0],X.shape[0]))

        for r in range(0,X.shape[0]):
                for c in range(0,X.shape[0]):
                        val = Y[r]*Y[c]*k(X[r],X[c])
                        P[r][c] = val
        P = matrix(P)
        print P
        q = -matrix(np.ones((X.shape[0],1)))
        
        G1 = -np.eye(X.shape[0])
        G2 = np.eye(X.shape[0])        
        G = matrix(np.concatenate((G1,G2),axis=0))

        h1 = np.zeros((X.shape[0],1))
        h2 = C*np.ones((X.shape[0],1))       
        h = matrix(np.concatenate((h1,h2),axis=0))
        
        A = matrix(Y.T)
        b = matrix(np.zeros((1,1)))

        #solution = solvers.qp(P, q, G, h, A, b)
        solution = solvers.qp(P, q, matrix(G1), matrix(h1),A, b)
        
        sol = solution['x']
        W = np.zeros(X[0].shape);
        
        C = []
        for i in range(0,X.shape[0]):
                if(sol[i] > 0.0001):
                        C.append((sol[i]*Y[i][0],X[i],Y[i]))

        def eval_point(x):
            v = 0
            for coeff,xi,yi in C:
                v += coeff*k(x,xi)
            return v 

        b = 0;
        for coeff,xi,yi in C:
            b += (eval_point(xi)-yi)
        b /= len(C)
        
        return C,b

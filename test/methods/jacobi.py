import helper

np = None

def import_library(library = 'pypacho'):
    global np
    if library == 'pypacho':
        import pypacho
        np = pypacho
    else:
        import numpy
        np = numpy

import numpy as np

def norm(x):
    d = x.transpose()
    cross = d @ x
    return np.sqrt(float(cross))

def jacobi(A,b,x0,N=100,tol=0.005):    
    x = x0
    disp =  tol+1 
    D = A.diag()
    diagflat = D.diagflat()
    R = A - diagflat
    i = 0
    while disp > tol and i < N:
        xn = (b - R @ x) / D
        disp = norm(xn - x) / norm(xn)
        x = xn
        i +=1
    return x,i

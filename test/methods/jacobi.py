import helper
from numpy import ndarray
np = None

def import_library(library = 'pypacho'):
    global np
    if library == 'pypacho':
        import pypacho
        np = pypacho
    else:
        import numpy
        np = numpy


def jacobi(A,b,x0,N=100,tol=0.005):
    x = x0
    disp =  tol+1 
    D = np.diag(A)
    if isinstance(D,ndarray):
        D = D.reshape(x.shape)
    diagflat = np.diagflat(D)
    R = A - diagflat
    i = 0
    while disp > tol and i < N:
        xn = (b - R @ x) / D
        disp = np.linalg.norm(xn - x) / np.linalg.norm(xn)
        x = xn
        i +=1
    return x,i

from opencl_array import OpenCLArray
import numpy
import os
from time import time

def jacobi(A,b,x0,np,ln,N=100,tol=0.001):
    D = np.diag(A)
    Df = []
    if(isinstance(A,numpy.ndarray)):
        m,n = A.shape
        Df = np.diagflat(D).reshape(m,n)
        D = D.reshape(m,1)
    else:
        Df = np.diagflat(D)
    R = A - Df
    x = x0
    iter=0       
    error = tol+1
    while (error > tol)&(iter <= N):
        xn = (b - (R @ x)) / D
        error = ln.norm(x - xn)
        x = xn
        iter = iter + 1 
    if (iter > N):
        print("Se ha excedido el número de iteraciones. Procedimiento FALLIDO")
    else:
        return x
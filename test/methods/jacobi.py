from anarray import AnArray
from gpu_array import GpuArray
from opencl_array import OpenCLArray
from our_cuda import OurCuda
import numpy as np
import os
from time import time

""" def jacobi(A,b,x0,np,ln,N=100,tol=0.001):
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
        return x """
        

def norm(x):
    d = x.transpose()
    cross = d @ x
    del(d)
    return np.sqrt(cross.Matrix.get()[0,0])

def jacobi(A,b,x0,N=100,tol=0.005):
    
    x = x0

    disp =  tol+1
    
    D = A.diag()
    diagflat = D.diagflat()
    R = A - diagflat
    del(A)
    del(diagflat)
    i = 0
    while disp > tol and i < N:
        cross_Cuda = R @ x

        sub_Cuda = b - cross_Cuda

        del(cross_Cuda)

        divide_gpu =  sub_Cuda / D

        xn = divide_gpu
        
        disp = norm(x -xn)/norm(x)
        
        x = xn
        
        i +=1
    
    return x

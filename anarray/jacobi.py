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

import numpy as np
import linalg as ln

os.environ['PYOPENCL_CTX'] = '0:0'
OpenCLArray.set_enviroment()

n = 5000
A = np.random.rand(n,n).astype(np.float32)

A = (A + np.eye(n)*(n+1)).astype(np.float32)
x = np.random.randint(0,100,(n,1)).astype(np.float32)

a_gpu = OpenCLArray(n,n,None,A)
x_gpu = OpenCLArray(n,1,None,x)
b = a_gpu@x_gpu
B = A@x

x0 = np.ones((n,1),dtype=np.float32)*(n+1)
x0_gpu = OpenCLArray(n,1,None,x0)

t = time()
sol = jacobi(a_gpu,b,x0_gpu,ln,ln,N=5000,tol=0.01)
t_gpu = time() - t
t = time()
sol2 = jacobi(A,B,x0,np,np.linalg,N=5000,tol=0.01)
t_cpu = time() - t

print('error gpu')
print(np.linalg.norm(x - sol.to_numpy()))
print('Time:')
print(t_gpu)
print('error cpu')
print(np.linalg.norm(x - sol2))
print('Time')
print(t_cpu)
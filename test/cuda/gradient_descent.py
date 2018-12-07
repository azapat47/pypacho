import helper
from pypacho.cuda import OurCuda
from methods.gradient_descent import gradient_descent, gradient_descent2
import os
from time import time

import numpy as np

n = 500
A = np.random.rand(n,n).astype(np.float32)

A = (A + np.eye(n)*(n+1)).astype(np.float32)
x = np.random.randint(0,100,(n,1)).astype(np.float32)

a_gpu = OurCuda(n,n,A,None)
x_gpu = OurCuda(n,1,x,None)
b = a_gpu @ x_gpu
B = A @ x

x0 = np.ones((n,1),dtype=np.float32)*(n+1)
x0_gpu = OurCuda(n,1,x0,None)

alpha = 0.0001/n

t = time()
sol = gradient_descent2(a_gpu,b,alpha,x0_gpu,N=500,tol=0.001)
t_gpu = time() - t
print('error gpu')
print(np.linalg.norm(x - sol.Matrix.get()) / np.linalg.norm(x))
print('Time:')
print(t_gpu)

t = time()
sol2 = gradient_descent2(A,B,alpha,x0,N=500,tol=0.001)
t_cpu = time() - t
print('error cpu')
print(np.linalg.norm(x - sol2)/np.linalg.norm(x))
print('Time')
print(t_cpu)

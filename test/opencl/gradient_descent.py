import helper
from methods.gradient_descent import *
from pypacho.opencl import OpenCLArray
from time import time
import os
import numpy as np

os.environ['PYOPENCL_CTX'] = '0:0'
OpenCLArray.set_enviroment()
import_library('pypacho')

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

alpha = 0.0001/n

t = time()
sol = gradient_descent(a_gpu,b,alpha,x0_gpu,N=5000,tol=0.1)
t_gpu = time() - t

print('error gpu')
print(np.linalg.norm(x - sol.to_numpy()) / np.linalg.norm(x))
print('Time:')
print(t_gpu)
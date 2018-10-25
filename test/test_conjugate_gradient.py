from opencl_array import OpenCLArray
from methods.conjugate_gradient import conjugate_gradient
import os
from time import time

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
sol = conjugate_gradient(a_gpu,b,x0_gpu,ln,N=5000,error=1)
t_gpu = time() - t
t = time()
sol2 = conjugate_gradient(A,B,x0,np.linalg,N=5000,error=1)
t_cpu = time() - t

print('error gpu')
print(np.linalg.norm(x - sol.to_numpy()))
print('Time:')
print(t_gpu)
print('error cpu')
print(np.linalg.norm(x - sol2))
print('Time')
print(t_cpu)

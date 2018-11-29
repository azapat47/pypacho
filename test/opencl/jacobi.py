import helper
from methods.jacobi import *
from pypacho.opencl import OpenCLArray
from time import time
import os
import numpy

os.environ['PYOPENCL_CTX'] = '0:0'
OpenCLArray.set_enviroment()
import_library('pypacho')

n = 1000
A = numpy.random.rand(n,n).astype(numpy.float32)

A = (A + numpy.eye(n)*(n+1)).astype(numpy.float32)
x = numpy.random.rand(n,1).astype(numpy.float32)

a_gpu = OpenCLArray(n,n,None,A)
x_gpu = OpenCLArray(n,1,None,x)
b = a_gpu@x_gpu
B = A@x

x0 = numpy.zeros((n,1),dtype=numpy.float32)*(n+1)
x0_gpu = OpenCLArray(n,1,None,x0)

t = time()
sol = jacobi(a_gpu,b,x0_gpu, N=5000,tol=0.001)
t_gpu = time() - t

print('error gpu')
print(numpy.linalg.norm(x - sol.to_numpy()))
print('Time:')
print(t_gpu)

import helper
from methods.jacobi import *
from time import time
import os
import numpy

import_library('numpy')

n = 1000
A = numpy.random.rand(n,n).astype(numpy.float32)

A = (A + numpy.eye(n)*(n+1)).astype(numpy.float32)
x = numpy.random.rand(n,1).astype(numpy.float32)
B = A@x

x0 = numpy.zeros((n,1),dtype=numpy.float32)*(n+1)

t = time()
sol = jacobi(A,B,x0, N=5000,tol=0.001)
t = time() - t

print('error')
print(numpy.linalg.norm(x - sol))
print('Time:')
print(t)

import helper
from methods.conjugate_gradient import *
from time import time
import os
import numpy as np
import_library('numpy')

n = 1000
A = np.random.rand(n,n).astype(np.float32)

A = (A + np.eye(n)*(n+1)).astype(np.float32)
x = np.random.randint(0,100,(n,1)).astype(np.float32)
B = A@x

x0 = np.ones((n,1),dtype=np.float32)*(n+1)

t = time()
sol = conjugate_gradient(A,B,x0,N=5000,tol=0.1)
t = time() - t

print('error')
print(np.linalg.norm(x - sol) / np.linalg.norm(x))
print('Time:')
print(t)
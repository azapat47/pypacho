from anarray import AnArray
from gpu_array import GpuArray
from opencl_array import OpenCLArray
from our_cuda import OurCuda
from methods.jacobi import jacobi
import numpy as np
import os
from time import time

size = 30

A = np.random.randn(size,size).astype(np.float32)
a_Cuda = OurCuda(A.shape[0],A.shape[1],A,None)
asita = a_Cuda.dom()
a_Cuda.Matrix.gpudata.free()


xorigin = np.random.randn(size,1).astype(np.float32)

B = asita.Matrix.get() @ xorigin
b_Cuda = OurCuda(B.shape[0],B.shape[1],B,None)


x = np.ones((size,1),dtype=np.float32)
x_Cuda  = OurCuda(x.shape[0],x.shape[1],x,None)

tol = 0.5 * 10**(-5)

x_Cuda = jacobi(asita,b_Cuda,x_Cuda,50,tol)

print("**"*40)
print("error del metodo")
error_method = xorigin - x_Cuda.Matrix.get()
error = np.linalg.norm(error_method, ord=2)/np.linalg.norm(xorigin, ord=2)
print(error)

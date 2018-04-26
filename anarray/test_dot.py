from opencl_array import OpenCLArray
import os
import numpy as np
from time import time

m = 5000
n = 5000
p = 5000
        
OpenCLArray.set_enviroment()
A = np.random.randint(0,100,(m,n))
B = np.random.randint(0,100,(n,p))
a_gpu = OpenCLArray(m,n,None,A.astype(np.float32))
b_gpu = OpenCLArray(n,p,None,B.astype(np.float32))

t = time()
C = A@B
cpu_time = time() - t

t = time()
c = a_gpu @ b_gpu
gpu_time = time() - t

error = np.sum(C - c.to_numpy())

print("CPU TIME: ")
print(cpu_time)
print("GPU_TIME: ")
print(gpu_time)
print("Error: ")
print(error)

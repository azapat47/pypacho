from opencl_array import OpenCLArray
import os
import numpy as np
from time import time

m = 5000
n = 1
p = 5000
        
os.environ['PYOPENCL_CTX'] = '0:2'
OpenCLArray.set_enviroment()
A = np.random.rand(m,n).astype(np.float32)
a_gpu = OpenCLArray(m,n,None,A)

t = time()
C = np.linalg.norm(A)
cpu_time = time() - t

t = time()
c = a_gpu.norm()
gpu_time = time() - t

print("CPU TIME: ")
print(cpu_time)
print("Result")
print(C)
print("GPU_TIME: ")
print(gpu_time)
print("Result")
print(c)
print("proportion")
print(C/c * 100)

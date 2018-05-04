from our_cuda import OurCuda
import os
import numpy as np
from time import time

m = 5000
n = 5000
p = 5000
        
A = np.matrix('1 2 3; 1 2 3; 1 2 3').astype(np.float32)
B = np.matrix('1 2 3; 1 2 3; 1 2 3').astype(np.float32)
a_Cuda = OurCuda(A,False,len(A),len(A))
b_Cuda = OurCuda(B,False,len(B),len(B))

t = time()
C = A*B
cpu_time = time() - t

t = time()
c = a_Cuda.cross(b_Cuda)
gpu_time = time() - t

#error = np.sum(C - c.to_numpy())



print("CPU TIME: ")
print(cpu_time)
print(C)
print("GPU_TIME: ")
print(gpu_time)
print(c.Matrix.get())
print("Error: ")
print(C - c.Matrix.get())

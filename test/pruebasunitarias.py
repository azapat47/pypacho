import helper
from pypacho.cuda import OurCuda
from pypacho.opencl import OpenCLArray
import numpy as np
import os

os.environ["PYOPENCL_CTX"]='0'
OpenCLArray.set_enviroment()

size =5000

print("tamaño = ", size)

sizea = (1,size)
sizeb = (size,1)
A =  np.random.uniform(low=-1, high=1, size = sizea).astype(np.float32)
B =  np.random.uniform(low=-1, high=1, size = sizeb).astype(np.float32)
print("operacion = A @ B")
#print("operacion = A.transpose() @ B.transpose()")
print("size A = ", sizea)
print("size B = ", sizeb)
a_cu = OurCuda(A.shape[0],A.shape[1],A,None)
b_cu = OurCuda(B.shape[0],B.shape[1],B,None)

a_cl = OpenCLArray(A.shape[0],A.shape[1],None,A)
b_cl = OpenCLArray(B.shape[0],B.shape[1],None,B)

a_cu_t = a_cu @ b_cu
A_t = A @ B
a_cl_t = a_cl @ b_cl

#a_cu_t = a_cu.transpose() @ b_cu.transpose()
#A_t = A.transpose() @ B.transpose()
#a_cl_t = a_cl.transpose() @ b_cl.transpose()

print("numpy - cuda presicion simple")
print(A_t-a_cu_t.to_numpy())
print("numpy - opencl presicion simple")
print(A_t - a_cl_t.to_numpy())


A =  np.random.uniform(low=-1, high=1, size=sizea).astype(np.float64)
B =  np.random.uniform(low=-1, high=1, size=sizeb).astype(np.float64)
a_cu = OurCuda(A.shape[0],A.shape[1],A,None)
b_cu = OurCuda(B.shape[0],B.shape[1],B,None)

a_cl = OpenCLArray(A.shape[0],A.shape[1],None,A)
b_cl = OpenCLArray(B.shape[0],B.shape[1],None,B)

a_cu_t = a_cu @ b_cu
A_t = A @ B
a_cl_t = a_cl @ b_cl

#a_cu_t = a_cu.transpose() @ b_cu.transpose()
#A_t = A.transpose() @ B.transpose()
#a_cl_t = a_cl.transpose() @ b_cl.transpose()
print("numpy - cuda presicion doble")
print(A_t-a_cu_t.to_numpy())
print("numpy - opencl presicion doble")
print(A_t - a_cl_t.to_numpy())
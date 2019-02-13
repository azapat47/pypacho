import helper
from pypacho.cuda import OurCuda
from pypacho.opencl import OpenCLArray
import numpy as np
import os

os.environ["PYOPENCL_CTX"]='0'
OpenCLArray.set_enviroment()

size = 3

A = np.random.randn(size,size).astype(np.float32)
B = np.random.randn(size,size).astype(np.float32)
a_cu = OurCuda(A.shape[0],A.shape[1],A,None)
b_cu = OurCuda(B.shape[0],B.shape[1],B,None)

a_cl = OpenCLArray(A.shape[0],A.shape[1],None,A)
b_cl = OpenCLArray(B.shape[0],B.shape[1],None,B)

a_cu_t = a_cu @ b_cu 
#A_t = A @ B

#a_cu_t_2 = b_cu.transpose() @ a_cu.transpose() 
#A_t_2 = B.transpose() @ A.transpose()

a_cl_t = a_cl @ b_cl
A_t = A @ B

print(A_t-a_cl_t.to_numpy())
print(A_t-a_cu_t.to_numpy())

#print(A_t_2 - a_cu_t_2.Matrix.get())



'''
#print(A)
a_cl = OpenCLArray(A.shape[0],A.shape[1],None,A)
diag = a_cl.diag()
#print(diag.to_numpy())
diagflat = diag.diagflat()
print("M")
#print(diagflat.to_numpy())
diagnp = np.diag(A)
diagflatnp = np.diagflat(diagnp)

print(diag.to_numpy()-diagnp)
print(diagflat.to_numpy()-diagflatnp)
'''
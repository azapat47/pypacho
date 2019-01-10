import helper
from pypacho.cuda import OurCuda
from pypacho.opencl import OpenCLArray
from methods.jacobi import jacobi
from methods.jacobi import import_library as ja_import
from methods.conjugate_gradient import conjugate_gradient
from methods.conjugate_gradient import import_library as cg_import
from methods.gradient_descent import gradient_descent, gradient_descent2
from methods.gradient_descent import import_library as gd_import
import numpy as np
OpenCLArray.set_enviroment()
size = 100
A = np.random.randn(size,size).astype(np.float32)
B = np.random.randn(size,size).astype(np.float32)
x = np.ones((size,1)).astype(np.float32)

a_cl = OpenCLArray(A.shape[0],A.shape[1],None,A)
x_cl = OpenCLArray(x.shape[0],x.shape[1],None,x)
a_cu = OurCuda(A.shape[0],A.shape[1],A,None)
b_cu = OurCuda(B.shape[0],B.shape[1],B,None)
x_cu = OurCuda(x.shape[0],x.shape[1],x,None)

sumcu = a_cu+b_cu
sumnp = A+B

print(x_cl)



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
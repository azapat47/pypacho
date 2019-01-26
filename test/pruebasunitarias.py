import helper
from pypacho.cuda import OurCuda
import numpy as np

size = 3

A = np.random.randn(size,size).astype(np.float32)
B = np.random.randn(size,size).astype(np.float32)
a_cu = OurCuda(A.shape[0],A.shape[1],A,None)
b_cu = OurCuda(B.shape[0],B.shape[1],B,None)

a_cu_t = a_cu - b_cu
A_t = A - B
print(A_t - a_cu_t.Matrix.get())



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
from pypacho.anarray import GpuArray, AnArray
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import numpy as np


#class NativeCuda(AnArray,GpuArray):
class NativeCuda:
  #Se tiene que cambiar a self, n, m, Matrix, host=None
    def __init__(self,Matrix,ready,n,m):
        self.n = n
        self.m = m
        if ready:
            self.Matrix = Matrix
        else:
            self.Matrix = gpuarray.to_gpu(Matrix)
            
    
    def __del__(self):
        del(self.Matrix)
        
    def add(self,cudita):
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        c_gpu = self.Matrix + cudita.Matrix
        return NativeCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)
      
    def subtract(self,cudita):
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        c_gpu = self.Matrix - cudita.Matrix
        return NativeCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)
      
    def multiply(self,cudita):
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        c_gpu = self.Matrix * cudita.Matrix
        return NativeCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)
      
    def divide(self,cudita):
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        c_gpu = self.Matrix / cudita.Matrix
        return NativeCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)
      
    #Modifican Matriz original
    def iadd(self,cudita):
        self.Matrix += cudita.Matrix
      
    def isubtract(self,cudita):
        self.Matrix -= cudita.Matrix
               
    def positive(self):
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        c_gpu = abs(self.Matrix)
        return NativeCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)
      
    def negative(self):
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        c_gpu = -self.Matrix
        return NativeCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)
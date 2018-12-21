from pypacho.anarray import GpuArray, AnArray
from pypacho.cuda import kernel
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import numpy as np


class OurCuda(AnArray,GpuArray):
    kernel_code_template = None
    kernelBin = None

    #Se tiene que cambiar a self, n, m, Matrix, host=None
    def __init__(self,n,m,Matrix = None,GpuMatrix = None):

        with open(kernel.get_path()) as file:
                self.kernel_code_template = file.read()
        self.kernelBin = compiler.SourceModule(self.kernel_code_template)
        self.n = n
        self.m = m
        if GpuMatrix is None:
            self.Matrix=gpuarray.to_gpu(Matrix)
        else:
            self.Matrix = GpuMatrix
    
    def __del__(self):
        self.Matrix.gpudata.free()

    def id(self):
        return str(hex(id(self.Matrix)))

    def meminfo(self, kernel):
        shared=kernel.shared_size_bytes
        regs=kernel.num_regs
        local=kernel.local_size_bytes
        const=kernel.const_size_bytes
        mbpt=kernel.max_threads_per_block
        print("""=MEM=\nLocal:%d,\nShared:%d,\nRegisters:%d,\nConst:%d,\nMax Threads/B:%d"""%(local,shared,regs,const,mbpt))
    
    def __str__(self):
        return str(self.Matrix)
    
    #def get(self, A):
    #    pycuda.driver.memcpy_dtoh(A,self.Matrix)

    def add(self,cudita):
        c_gpu = self.Matrix + cudita.Matrix
        return OurCuda(self.n,self.m,None,c_gpu)
    
    def subtract(self,cudita):
        c_gpu = self.Matrix - cudita.Matrix
        return OurCuda(self.n,self.m,None,c_gpu)

    def multiply(self,cudita):
        if not isinstance(cudita,OurCuda):
            c_gpu = cudita * self.Matrix
        else:
            c_gpu = self.Matrix * cudita.Matrix
        return OurCuda(self.n,self.m,None,c_gpu)

    def divide(self,cudita):
        c_gpu = self.Matrix / cudita.Matrix
        return OurCuda(self.n,self.m,None,c_gpu)

    def mod(self,cudita):
        Mod = self.kernelBin.get_function("Mod")
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Mod(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.n,self.m,None,c_gpu)
      
    #Modificar la matriz original
    
    def iadd(self,cudita):
        Suma = self.kernelBin.get_function("ISuma")
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1

        Suma(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, block = (32, 32, 1), grid = (grid_size,grid_size,1))
    
    def isubtract(self,cudita):
        Resta = self.kernelBin.get_function("IResta")
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Resta(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, block = (32, 32, 1), grid = (grid_size,grid_size,1))

    def imultiply(self,cudita):
        Multi = self.kernelBin.get_function("IMulti")
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Multi(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, block = (32, 32, 1), grid = (grid_size,grid_size,1))

    def idivide(self,cudita):
        Divide = self.kernelBin.get_function("IDivide")
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Divide(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, block = (32, 32, 1), grid = (grid_size,grid_size,1))

    def imod(self,cudita):
        Mod = self.kernelBin.get_function("IMod")
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Mod(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, block = (32, 32, 1), grid = (grid_size,grid_size,1))

    #falta negative y positive

    def dot(self, cudita):
        Cross = self.kernelBin.get_function("Cross")
        #c_gpu = gpuarray.empty((self.n, cudita.m), np.float32)
        x = np.zeros((self.n, cudita.m),dtype=np.float32)
        c_gpu = gpuarray.to_gpu(x)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Cross(np.int32(MATRIX_SIZE),np.int32(self.m), np.int32(cudita.m),self.Matrix, cudita.Matrix, c_gpu, block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.n,cudita.m,None,c_gpu)
    
    #Better than Dom
    def domself(self):
        Dom = self.kernelBin.get_function("Dom1")
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 1024):
            grid_size = (MATRIX_SIZE//1024) + 1
        else:
            grid_size = 1
        Dom(np.int32(MATRIX_SIZE), self.Matrix ,block = (1, 1024, 1), grid = (1,grid_size,1))
    
    def dom(self):
        Dom = self.kernelBin.get_function("Dom2")
        MATRIX_SIZE = self.n
        b_gpu = gpuarray.empty((self.n, self.m), np.float32)
        if(MATRIX_SIZE > 1024):
            grid_size = (MATRIX_SIZE//1024) + 1
        else:
            grid_size = 1
        Dom(np.int32(MATRIX_SIZE), self.Matrix, b_gpu ,block = (1, 1024, 1), grid = (1,grid_size,1))
        return OurCuda(self.n,self.m,None,b_gpu)
    
    def transpose(self):
        Transpose = self.kernelBin.get_function("Transpose")
        MATRIX_SIZE = self.n
        x = np.zeros((self.m, self.n),dtype=np.float32)
        b_gpu = gpuarray.to_gpu(x)
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Transpose(np.int32(self.m),np.int32(self.n), self.Matrix, b_gpu ,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.m,self.n,None,b_gpu)

    def diag(self):
        MATRIX_SIZE = self.n
        #b_gpu = gpuarray.empty((self.n, 1), np.float32)
        x = np.zeros((self.n,1),dtype=np.float32)
        b_gpu = gpuarray.to_gpu(x)
        if(MATRIX_SIZE > 1024):
            grid_size = (MATRIX_SIZE//1024) + 1
        else:
            grid_size = 1
        Diag = self.kernelBin.get_function("Diag")
        Diag(np.int32(MATRIX_SIZE), self.Matrix, b_gpu ,block = (1, 1024, 1), grid = (1,grid_size,1))
        return OurCuda(1,MATRIX_SIZE,None,b_gpu)

    def diagflat(self):
        MATRIX_SIZE = self.m
        x = np.zeros((MATRIX_SIZE,MATRIX_SIZE),dtype=np.float32)
        #b_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)
        b_gpu = gpuarray.to_gpu(x)
        if(MATRIX_SIZE > 1024):
            grid_size = (MATRIX_SIZE//1024) + 1
        else:
            grid_size = 1
        DiagFlat = self.kernelBin.get_function("DiagFlat")
        DiagFlat(np.int32(MATRIX_SIZE), self.Matrix, b_gpu ,block = (1, 1024, 1), grid = (1,grid_size,1))
        return OurCuda(MATRIX_SIZE,MATRIX_SIZE,None,b_gpu)

    def __float__(self):
        return float(self.Matrix.get()[0,0])
    
    def to_numpy(self):
        mat = self.Matrix.get()
        return mat
    
    def norm(self):
        at = self.transpose()
        n2 = []
        if(self.n != 1):
            n2 = at @ self
        else:
            n2 = self @ at
        return float(np.sqrt(n2.to_numpy()))
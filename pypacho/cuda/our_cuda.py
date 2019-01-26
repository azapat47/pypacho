from pypacho.anarray import GpuArray, AnArray
from pypacho.cuda import kernel
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import numpy as np


class OurCuda(AnArray,GpuArray):
    kernel_code_template = None
    kernelBin = None
    kernel_code_template_d = None
    kernelBin_d = None
    doble = False
    transp = False

    #Se tiene que cambiar a self, n, m, Matrix, host=None
    def __init__(self,n,m,Matrix = None,GpuMatrix = None, Transp = False):

        self.n = n
        self.m = m
        self.transp = Transp
        if GpuMatrix is None:
            if Matrix.dtype == 'float64':
                self.doble = True
            else:
                self.doble = False

            self.Matrix=gpuarray.to_gpu(Matrix)
        else:
            if GpuMatrix.dtype == 'float64':
                self.doble = True
            else:
                self.doble = False

            self.Matrix = GpuMatrix
            
        if self.doble:
            #doble
            with open(kernel.get_path_d()) as file:
                    self.kernel_code_template_d = file.read()
            self.kernelBin_d = compiler.SourceModule(self.kernel_code_template_d)
        else:
            #simple
            with open(kernel.get_path()) as file:
                    self.kernel_code_template = file.read()
            self.kernelBin = compiler.SourceModule(self.kernel_code_template)
        
    
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

    def add_b(self,cudita):
        c_gpu = self.Matrix + cudita.Matrix
        return OurCuda(self.n,self.m,None,c_gpu)
    
    def add(self,cudita):
        if self.doble:
            Add = self.kernelBin_d.get_function("Suma")
            c_gpu = gpuarray.empty((self.n, self.m), np.float64)
        else:
            Add = self.kernelBin.get_function("Suma")
            c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Add(np.int32(self.transp), np.int32(cudita.transp), np.int32(self.n), np.int32(self.m),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.n,self.m,None,c_gpu)
    
    
    def subtract(self,cudita):
        if self.doble:
            Sub = self.kernelBin_d.get_function("Resta")
            c_gpu = gpuarray.empty((self.n, self.m), np.float64)
        else:
            Sub = self.kernelBin.get_function("Resta")
            c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Sub(np.int32(self.transp), np.int32(cudita.transp), np.int32(self.n), np.int32(self.m),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.n,self.m,None,c_gpu)

    def multiply(self,cudita):
        if not isinstance(cudita,OurCuda):
            c_gpu = cudita * self.Matrix
        else:
            if self.doble:
                Mul = self.kernelBin_d.get_function("Multi")
                c_gpu = gpuarray.empty((self.n, self.m), np.float64)
            else:
                Mul = self.kernelBin.get_function("Multi")
                c_gpu = gpuarray.empty((self.n, self.m), np.float32)
            MATRIX_SIZE = self.n
            if(MATRIX_SIZE > 32):
                grid_size = (MATRIX_SIZE//32) + 1
            else:
                grid_size = 1
            Mul(np.int32(self.transp), np.int32(cudita.transp), np.int32(self.n), np.int32(self.m),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.n,self.m,None,c_gpu)

    def divide(self,cudita):
        if self.doble:
            Div = self.kernelBin_d.get_function("Divide")
            c_gpu = gpuarray.empty((self.n, self.m), np.float64)
        else:
            Div = self.kernelBin.get_function("Divide")
            c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Div(np.int32(self.transp), np.int32(cudita.transp), np.int32(self.n), np.int32(self.m),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.n,self.m,None,c_gpu)
      
    #falta negative y positive

    def dot(self, cudita):
        if self.doble:
            Cross = self.kernelBin_d.get_function("Cross")
            x = np.zeros((self.n, cudita.m),dtype=np.float64)
        else:
            Cross = self.kernelBin.get_function("Cross")
            x = np.zeros((self.n, cudita.m),dtype=np.float32)
        #c_gpu = gpuarray.empty((self.n, cudita.m), np.float32)
        c_gpu = gpuarray.to_gpu(x)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Cross(np.int32(self.transp), np.int32(cudita.transp), np.int32(MATRIX_SIZE),np.int32(self.m), np.int32(cudita.m),self.Matrix, cudita.Matrix, c_gpu, block = (32, 32, 1), grid = (grid_size,grid_size,1))
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
    
    def transpose_b(self):
        if self.doble:
            Transpose = self.kernelBin_d.get_function("Transpose")
            x = np.zeros((self.m, self.n),dtype=np.float64)
        else:
            Transpose = self.kernelBin.get_function("Transpose")
            x = np.zeros((self.m, self.n),dtype=np.float32)
        MATRIX_SIZE = self.n
        b_gpu = gpuarray.to_gpu(x)
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Transpose(np.int32(self.m),np.int32(self.n), self.Matrix, b_gpu ,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.m,self.n,None,b_gpu)

    def transpose(self):
        #b_gpu = self.Matrix.copy()
        return OurCuda(self.m,self.n,None,self.Matrix.copy(),True)

    def diag(self):
        if self.doble:
            Diag = self.kernelBin_d.get_function("Diag")
            x = np.zeros((self.n,1),dtype=np.float64)
        else:
            Diag = self.kernelBin.get_function("Diag")
            x = np.zeros((self.n,1),dtype=np.float32)
        
        b_gpu = gpuarray.to_gpu(x)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 1024):
            grid_size = (MATRIX_SIZE//1024) + 1
        else:
            grid_size = 1
        Diag(np.int32(MATRIX_SIZE), self.Matrix, b_gpu ,block = (1, 1024, 1), grid = (1,grid_size,1))
        return OurCuda(1,MATRIX_SIZE,None,b_gpu)

    def diagflat(self):
        MATRIX_SIZE = self.m
        if self.doble:
            DiagFlat = self.kernelBin_d.get_function("DiagFlat")
            x = np.zeros((MATRIX_SIZE,MATRIX_SIZE),dtype=np.float64)
        else:
            DiagFlat = self.kernelBin.get_function("DiagFlat")
            x = np.zeros((MATRIX_SIZE,MATRIX_SIZE),dtype=np.float32)
        #b_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)
        b_gpu = gpuarray.to_gpu(x)
        if(MATRIX_SIZE > 1024):
            grid_size = (MATRIX_SIZE//1024) + 1
        else:
            grid_size = 1
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
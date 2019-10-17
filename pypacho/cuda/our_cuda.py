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
    npType = None
    transp = False

    #Se tiene que cambiar a self, n, m, Matrix, host=None
    def __init__(self,n,m,Matrix = None,GpuMatrix = None, Transp = False, Bin = None):

        self.n = n
        self.m = m
        if Transp:
            self.transp = not self.transp
        if GpuMatrix is None:
            if Matrix.dtype == 'float64':
                self.npType = np.float64
            else:
                self.npType = np.float32
            self.Matrix=gpuarray.to_gpu(Matrix)
        else:
            if GpuMatrix.dtype == 'float64':
                self.npType = np.float64
            else:
                self.npType = np.float32
            self.Matrix = GpuMatrix
        if Bin is None:
            if self.npType == np.float64:
                #doble
                with open(kernel.get_path_d()) as file:
                        self.kernel_code_template = file.read()
                #self.kernelBin_d = compiler.SourceModule(self.kernel_code_template_d)
                self.kernelBin = compiler.SourceModule(self.kernel_code_template)
            else:
                #simple
                with open(kernel.get_path()) as file:
                        self.kernel_code_template = file.read()
                self.kernelBin = compiler.SourceModule(self.kernel_code_template)
        else:
            self.kernelBin = Bin        
    
    def __del__(self):
        pass
        #self.Matrix.gpudata.free()

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
    
    def add(self,cudita):
        Add = self.kernelBin.get_function("Suma")
        c_gpu = gpuarray.empty((self.n, self.m), self.npType)
        MATRIX_SIZE = max(self.n,self.m)
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Add(np.int32(self.transp), np.int32(cudita.transp), np.int32(self.n), np.int32(self.m),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.n,self.m,None,c_gpu,Bin=self.kernelBin)
    
    
    def subtract(self,cudita):
        Sub = self.kernelBin.get_function("Resta")
        c_gpu = gpuarray.empty((self.n, self.m), self.npType)
        MATRIX_SIZE = max(self.n,self.m)
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Sub(np.int32(self.transp), np.int32(cudita.transp), np.int32(self.n), np.int32(self.m),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.n,self.m,None,c_gpu,Bin=self.kernelBin)

    def multiply(self,cudita):
        if not isinstance(cudita,OurCuda):
            c_gpu = cudita * self.Matrix
        else:
            Mul = self.kernelBin.get_function("Multi")
            c_gpu = gpuarray.empty((self.n, self.m), self.npType)
            MATRIX_SIZE = max(self.n,self.m)
            if(MATRIX_SIZE > 32):
                grid_size = (MATRIX_SIZE//32) + 1
            else:
                grid_size = 1
            Mul(np.int32(self.transp), np.int32(cudita.transp), np.int32(self.n), np.int32(self.m),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.n,self.m,None,c_gpu,Bin=self.kernelBin)

    def divide(self,cudita):
        Div = self.kernelBin.get_function("Divide")
        c_gpu = gpuarray.empty((self.n, self.m), self.npType)    
        MATRIX_SIZE = max(self.n,self.m)
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Div(np.int32(self.transp), np.int32(cudita.transp), np.int32(self.n), np.int32(self.m),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(self.n,self.m,None,c_gpu,Bin=self.kernelBin)
    
    def dot(self, cudita):
        MATRIX_SIZE = max(self.n,self.m,cudita.n,cudita.m)
        if(MATRIX_SIZE > 32):
            if MATRIX_SIZE % 32 == 0:
                sum = 0
            else:
                sum = 1
            grid_size_32 = (MATRIX_SIZE//32)+sum
        else:
            grid_size_32 = 1

        if(MATRIX_SIZE > 1024):
            if MATRIX_SIZE % 1024 == 0:
                sum = 0
            else:
                sum = 1
            grid_size_1024 = (MATRIX_SIZE//1024)+sum
        else:
            grid_size_1024 = 1
        c_gpu = gpuarray.empty((self.n, cudita.m), self.npType)
        if self.n == cudita.n and self.m == cudita.m:
            Cross = self.kernelBin.get_function("matrixMul")
            Cross(self.Matrix, cudita.Matrix, c_gpu,np.int32(self.n),np.int32(self.m),np.int32(cudita.n),np.int32(cudita.m),np.int32(self.n),np.int32(cudita.m), np.int32(self.transp), np.int32(cudita.transp),block = (32, 32, 1), grid = (grid_size_32,grid_size_32,1))
        elif self.n != 1 and cudita.m == 1:
            Cross = self.kernelBin.get_function("MatDotVec")
            Cross(self.Matrix, cudita.Matrix, c_gpu,np.int32(self.n),np.int32(self.m),np.int32(cudita.n),np.int32(cudita.m),np.int32(self.n),np.int32(cudita.m), np.int32(self.transp), np.int32(cudita.transp),block = (1, 32, 1), grid = (1,grid_size_32,1))                
        elif self.n == 1 and cudita.m == 1:
            Cross = self.kernelBin.get_function("vec_dot")
            Cross(self.Matrix, cudita.Matrix, c_gpu, np.int32(MATRIX_SIZE),np.int32(self.transp), np.int32(cudita.transp),block = (1024, 1, 1), grid = (grid_size_1024,1,1))
        else:
            Cross = self.kernelBin.get_function("DOT")
            Cross(np.int32(self.transp), np.int32(cudita.transp), np.int32(self.n),np.int32(self.m), np.int32(cudita.m),self.Matrix, cudita.Matrix, c_gpu, block = (32, 32, 1), grid = (grid_size_32,grid_size_32,1))
            
        return OurCuda(self.n,cudita.m,None,c_gpu,Bin=self.kernelBin)    

    def transpose(self):
        return OurCuda(self.m,self.n,None,self.Matrix,True,Bin=self.kernelBin)

    def diag(self):
        Diag = self.kernelBin.get_function("Diag")
        b_gpu = gpuarray.empty((self.n, 1), self.npType)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 1024):
            grid_size = (MATRIX_SIZE//1024) + 1
        else:
            grid_size = 1
        Diag(np.int32(MATRIX_SIZE), self.Matrix, b_gpu ,block = (1, 1024, 1), grid = (1,grid_size,1))
        return OurCuda(1,MATRIX_SIZE,None,b_gpu,Bin=self.kernelBin)

    def diagflat(self):
        MATRIX_SIZE = max(self.n,self.m)
        DiagFlat = self.kernelBin.get_function("DiagFlat")
        b_gpu = gpuarray.zeros((MATRIX_SIZE,MATRIX_SIZE),dtype=self.npType)
        if(MATRIX_SIZE > 1024):
            grid_size = (MATRIX_SIZE//1024) + 1
        else:
            grid_size = 1
        DiagFlat(np.int32(MATRIX_SIZE), self.Matrix, b_gpu ,block = (1, 1024, 1), grid = (1,grid_size,1))
        return OurCuda(MATRIX_SIZE,MATRIX_SIZE,None,b_gpu,Bin=self.kernelBin)

    def __float__(self):
        return float(self.Matrix.get()[0,0])
    
    def to_numpy(self):
        mat = self.Matrix.get()
        if self.transp:
            mat = mat.transpose()
        return mat
    
    def norm_b(self):
        at = self.transpose()
        n2 = []
        if(self.n != 1):
            n2 = at @ self
        else:
            n2 = self @ at
        return float(np.sqrt(n2.to_numpy()))

    def norm(self):
        MATRIX_SIZE = max(self.n,self.m)
        x = np.zeros(1,dtype=self.npType)
        c_gpu = driver.mem_alloc(x.nbytes)
        if(MATRIX_SIZE > 1024):
            if MATRIX_SIZE % 1024 == 0:
                sum = 0
            else:
                sum = 1
            grid_size = (MATRIX_SIZE//1024)+sum
        else:
            grid_size = 1
        Cuadratic = self.kernelBin.get_function("cuadratic_sum")
        Cuadratic(self.Matrix, c_gpu, np.int32(MATRIX_SIZE),
        block = (1024, 1, 1), grid = (grid_size,1,1))
        driver.memcpy_dtoh(x,c_gpu)
        return float(np.sqrt(x))
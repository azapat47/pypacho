from gpu_array import GpuArray
from anarray import AnArray
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import numpy as np

class OurCuda(AnArray,GpuArray):

    kernel_code_template = """
    __global__ void Suma(int size, float *a, float *b, float *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] + b[idx + (idy*size)];
        }
    }
    
    __global__ void ISuma(int size, float *a, float *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] + b[idx + (idy*size)];
        }
    }

    __global__ void Resta(int size, float *a, float *b, float *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] - b[idx + (idy*size)];
        }
    } 
    
    __global__ void IResta(int size, float *a, float *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] - b[idx + (idy*size)];
        }
    }
    

    __global__ void Multi(int size, float *a, float *b, float *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] * b[idx + (idy*size)];
        }
    }
    
    __global__ void IMulti(int size, float *a, float *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] * b[idx + (idy*size)];
        }
    }

    __global__ void Divide(int size, float *a, float *b, float *c)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            c[idx + (idy*size)] =  a[idx + (idy*size)] / b[idx + (idy*size)];
        }
    }
    
    __global__ void IDivide(int size, float *a, float *b)
    {
        const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint idy = threadIdx.y + blockDim.y * blockIdx.y;
        if(idx < size && idy < size){
            a[idx + (idy*size)] =  a[idx + (idy*size)] / b[idx + (idy*size)];
        }
    }

    __global__ void Cross(int size, float *a, float *b, float *c)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size && ty < size){
            float Pvalue = 0;
            for (int k = 0; k < size; ++k) {
                float Aelement = a[ty * size + k];
                float Belement = b[k * size + tx];
                Pvalue += Aelement * Belement;
            }
            c[ty * size + tx] = Pvalue;
        }
    }

    // dominante modificando la matriz original
    __global__ void Dom1(int size, float *a)
    {
        int ty = threadIdx.y + blockDim.y * blockIdx.y;
        float Pvalue = 0;
        if(ty < size){
            for (int i = 0; i < size; ++i) {
                Pvalue += abs(a[ty * size + i]);
            }
        a[ty * size + ty] = Pvalue + 2000.0;
        }
    }

    // dominante guardando el resultado en otra matriz
    __global__ void Dom2(int size, float *a, float *b)
    {
        int ty = threadIdx.y + blockDim.y * blockIdx.y;
        float Pvalue = 0;
        if(ty < size){
            for (int i = 0; i < size; ++i) {
                Pvalue += abs(a[ty * size + i]);
                b[ty * size + i] = a[ty * size + i];
            }
            b[ty * size + ty] = Pvalue + 2000.0; 
        }
    }
    
    __global__ void Transpose(int size, const float *a, float *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size && ty < size){
            b[ty + (tx*size)] =  a[tx + (ty*size)];
        }
    }
    
    __global__ void neg(int size, float * a, float *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size && ty < size){
            b[tx + (ty*size)] =  a[tx + (ty*size)] * -1;
        }
    }
    
    __global__ void absolute(int size, float *a, float *b)
    {
        const uint tx = threadIdx.x + blockDim.x * blockIdx.x;
        const uint ty = threadIdx.y + blockDim.y * blockIdx.y;
        if(tx < size && ty < size){
            b[tx + (ty*size)] =  abs(a[tx + (ty*size)]);
        }
    }
    """
    kernelBin = compiler.SourceModule(kernel_code_template)
    
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
        Suma = self.kernelBin.get_function("Suma")
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1

        Suma(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)
    
    def subtract(self,cudita):
        Resta = self.kernelBin.get_function("Resta")
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Resta(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)

    def multiply(self,cudita):
        Multi = self.kernelBin.get_function("Multi")
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Multi(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)

    def divide(self,cudita):
        Divide = self.kernelBin.get_function("Divide")
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Divide(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)

    def mod(self,cudita):
        Mod = self.kernelBin.get_function("Mod")
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1
        Mod(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, c_gpu,block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)
      
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

    def cross(self, cudita):
        Cross = self.kernelBin.get_function("Cross")
        c_gpu = gpuarray.empty((self.n, self.m), np.float32)
        MATRIX_SIZE = self.n
        if(MATRIX_SIZE > 32):
            grid_size = (MATRIX_SIZE//32) + 1
        else:
            grid_size = 1

        Cross(np.int32(MATRIX_SIZE),self.Matrix, cudita.Matrix, c_gpu, block = (32, 32, 1), grid = (grid_size,grid_size,1))
        return OurCuda(c_gpu,True,MATRIX_SIZE,MATRIX_SIZE)
    
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
        return OurCuda(b_gpu,True,MATRIX_SIZE,MATRIX_SIZE)
    
    def transpose(self):
        Transpose = self.kernelBin.get_function("Transpose")
        MATRIX_SIZE = self.n
        b_gpu = gpuarray.empty((self.n, self.m), np.float32)
        if(MATRIX_SIZE > 1024):
            grid_size = (MATRIX_SIZE//1024) + 1
        else:
            grid_size = 1
        Transpose(np.int32(MATRIX_SIZE), self.Matrix, b_gpu ,block = (1, 1024, 1), grid = (1,grid_size,1))
        return OurCuda(b_gpu,True,MATRIX_SIZE,MATRIX_SIZE)

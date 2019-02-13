from pypacho.anarray import GpuArray, AnArray
from pypacho.opencl import kernel
import pyopencl
import numpy

class OpenCLArray(AnArray,GpuArray):
    ctx = None
    queue = None
    mf = None
    prg = None
    block_size = None
    ready = False

    def set_enviroment(block = None, options = '-Werror', kernel_params = None):
        if(not OpenCLArray.ready):
            with open(kernel.get_path()) as file:
                KERNEL_CODE = file.read()
            OpenCLArray.ctx = pyopencl.create_some_context()
            OpenCLArray.queue = pyopencl.CommandQueue(OpenCLArray.ctx,
                                        properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            OpenCLArray.block_size = block
            if(kernel_params is None):
                kernel_params = ''
            if(options is None):
                options = []
            OpenCLArray.prg = pyopencl.Program(OpenCLArray.ctx, KERNEL_CODE,
            ).build(options=options)
            OpenCLArray.mf = pyopencl.mem_flags
            OpenCLArray.ready = True

    def __init__(self,m,n,buf=None,host=None, dtype=None):
        self.m = m
        self.n = n
        self.shape = (m,n)
        if(host is None):
            self.buf = buf
            if dtype is None:
                self.dtype = numpy.dtype(numpy.float32)
            else:
                self.dtype = numpy.dtype(dtype)

        else:
            self.buf = pyopencl.Buffer\
                    (self.ctx,self.mf.READ_ONLY |self.mf.COPY_HOST_PTR, hostbuf=host)
            self.dtype = host.dtype
        self.nbytes = self.m * self.n * self.dtype.itemsize
            
    def __del__(self):
        self.buf.release()

        
    def transpose(self):
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, size=self.nbytes)
        grid = (self.m *self.n,)
        if self.dtype == numpy.float32:
            cl_function = self.prg.transpose
        elif self.dtype == numpy.float64:
            cl_function = self.prg.double_transpose
            
        cl_function(self.queue, grid, self.block_size,
                           c_buf, self.buf, numpy.uint32(self.m), numpy.uint32(self.n))
        return OpenCLArray(self.n,self.m,c_buf,None,self.dtype)


    def add(self,B):
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, self.nbytes)
        grid = (self.m *self.n,)
        if self.dtype == numpy.float32:
            cl_function = self.prg.add
        elif self.dtype == numpy.float64:
            cl_function = self.prg.double_add

        cl_function(self.queue, grid, self.block_size,
                           self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None,self.dtype)
    
    def subtract(self,B):
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, self.nbytes)
        grid = (self.m *self.n,)
        if self.dtype == numpy.float32:
            cl_function = self.prg.subtract
        elif self.dtype == numpy.float64:
            cl_function = self.prg.double_subtract

        cl_function(self.queue, grid, self.block_size,
                           self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None,self.dtype)
    
    def multiply(self,B):
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, self.nbytes)
        grid = (self.m *self.n,)
        
        if(not isinstance(B,OpenCLArray)):
            if self.dtype == numpy.float32:
                cl_function = self.prg.scalar_mult
                cl_function(self.queue, grid, self.block_size,
                                    self.buf,numpy.float32(B),c_buf)
            elif self.dtype == numpy.float64:
                cl_function = self.prg.double_scalar_mult
                cl_function(self.queue, grid, self.block_size,
                                    self.buf,numpy.float64(B),c_buf)
        else:
            if self.dtype == numpy.float32:
                cl_function = self.prg.multiply
            elif self.dtype == numpy.float64:
                cl_function = self.prg.double_multiply

            cl_function(self.queue, grid, self.block_size,
                               self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None,self.dtype)
    
    def divide(self,B):
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, self.nbytes)
        grid = (self.m *self.n,)
        if self.dtype == numpy.float32:
            cl_function = self.prg.divide
        elif self.dtype == numpy.float64:
            cl_function = self.prg.double_divide
        
        cl_function(self.queue, grid, self.block_size,
                           self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None,self.dtype)
    
    def dot_b(self,B):        
        if(self.n == 1 and self.m != 1 and B.n == 1  and B.m != 1):
            return B.transpose().dot(self)
        else:
            nbytes = self.m * B.n * self.dtype.itemsize
            c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, nbytes)
            grid = (self.m *B.n,)
            if self.dtype == numpy.float32:
              cl_function = self.prg.dot_matrix
            elif self.dtype == numpy.float64:
                cl_function = self.prg.double_dot_matrix

            cl_function(self.queue, grid, self.block_size,self.buf, B.buf, c_buf,
                         numpy.uint32(self.m),
                         numpy.uint32(self.n),
                         numpy.uint32(B.n))
            return OpenCLArray(self.m,B.n,c_buf,None,self.dtype)
    
    def dot(self,B):        
        if(self.n == 1 and self.m != 1 and B.n == 1  and B.m != 1):
            return B.transpose().dot(self)
        else:
            nbytes = self.m * B.n * self.dtype.itemsize
            c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, nbytes)
            if self.dtype == numpy.float32:
              cl_function = self.prg.dot_matrix2
            elif self.dtype == numpy.float64:
                cl_function = self.prg.dot_matrix2
            
            MATRIX_SIZE = max(self.n,B.m)
            if(MATRIX_SIZE > 32):
                if MATRIX_SIZE % 32 == 0:
                    sum = 0
                else:
                    sum = 1
                grid_size = (self.n//32) + sum
            else:
                grid_size = 1
            grid = (grid_size*32, grid_size*32)
            #grid = (self.m, B.n)
            block = 16
            cl_function(self.queue, grid, (block, block), 
                         numpy.uint32(self.n),
                         numpy.uint32(self.m),
                         numpy.uint32(B.n),
                         numpy.uint32(B.m),
                         numpy.uint32(block),
                         self.buf, B.buf, c_buf,
                         pyopencl.LocalMemory(self.dtype.itemsize * block * block),
                         pyopencl.LocalMemory(self.dtype.itemsize * block * block))
            return OpenCLArray(self.m,B.n,c_buf,None,self.dtype)

    def negative(self):
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, self.nbytes)
        grid = (self.m *self.n,)
        if self.dtype == numpy.float32:
            cl_function = self.prg.negative
        elif self.dtype == numpy.float64:
            cl_function = self.prg.double_negative
        
        cl_function(self.queue, grid, self.block_size,
                           c_buf, self.buf)
        return OpenCLArray(self.m,self.n,c_buf,None,self.dtype)

    def sqrt(self):
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, self.nbytes)
        grid = (self.m *self.n,)
        if self.dtype == numpy.float32:
            cl_function = self.prg.sqrt_
        elif self.dtype == numpy.float64:
            cl_function = self.prg.double_sqrt_
        cl_function(self.queue, grid, self.block_size,
                           c_buf, self.buf)
        return OpenCLArray(self.m,self.n,c_buf,None,self.dtype)

    def diag(self):
        nbytes = self.m * self.dtype.itemsize
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, nbytes)
        grid = (self.m,)
        if self.dtype == numpy.float32:
            cl_function = self.prg.diag
        elif self.dtype == numpy.float64:
            cl_function = self.prg.double_diag
        
        cl_function(self.queue, grid, self.block_size,
                           self.buf, c_buf, numpy.uint32(self.m))
        return OpenCLArray(1,self.m,c_buf,None,self.dtype)

    def diagflat(self):
        nbytes = self.n * self.n * self.dtype.itemsize
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, nbytes)
        grid = (self.n *self.n,)
        if self.dtype == numpy.float32:
            cl_function = self.prg.diagflat
        elif self.dtype == numpy.float64:
            cl_function = self.prg.double_diagflat

        cl_function(self.queue, grid, self.block_size,
                           self.buf, c_buf, numpy.uint32(self.n))
        return OpenCLArray(self.n,self.n,c_buf,None,self.dtype)
    
    def norm(self):
        at = self.transpose()
        n2 = []
        if(self.m != 1):
            n2 = at @ self
        else:
            n2 = self @ at
        return float(numpy.sqrt(n2.to_numpy()))

    def __float__(self):
        return float(self.to_numpy())

    def to_numpy(self):
        C = numpy.zeros((self.m*self.n),dtype=self.dtype)
        pyopencl.enqueue_copy(self.queue, C, self.buf)
        return C.reshape(self.m,self.n)

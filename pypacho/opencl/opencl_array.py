from pypacho.anarray import GpuArray, AnArray
from pypacho.opencl import kernel
import pyopencl
import numpy
import math

class OpenCLArray(AnArray,GpuArray):
    ctx = None
    queue = None
    mf = None
    prg = None
    block_size = None
    ready = False
    max_block_size = None

    def set_enviroment(block = None, options = '-Werror', kernel_params = None):
        if(not OpenCLArray.ready):
            with open(kernel.get_path()) as file:
                KERNEL_CODE = file.read()
            OpenCLArray.ctx = pyopencl.create_some_context()
            OpenCLArray.queue = pyopencl.CommandQueue(OpenCLArray.ctx,
                                        properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            OpenCLArray.block_size = block
            OpenCLArray.max_block_size = OpenCLArray.queue.device.max_work_group_size
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
        assert self.n == B.m, "Matrix dimentions must match"
        if(self.m == 1 and B.n == 1):
            return self.vecdot(B)
        elif(B.n == 1):
            return self.matrixvec(B)
        else:
            nbytes = self.m * B.n * self.dtype.itemsize
            c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, nbytes)
            if self.dtype == numpy.float32:
              cl_function = self.prg.dot_matrix
            elif self.dtype == numpy.float64:
                cl_function = self.prg.double_dot_matrix
            
            block = int(numpy.sqrt(self.max_block_size))
            blockx = min(block, self.m)
            blocky = min(block, self.n)

            grid = (math.ceil(self.m / blockx) * blockx, math.ceil(self.n / blocky) * blocky)
            cl_function(self.queue, grid, (blockx, blocky), 
                         numpy.uint32(self.m),
                         numpy.uint32(self.n),
                         numpy.uint32(B.n),
                         numpy.uint32(blockx),
                         self.buf, B.buf, c_buf,
                         pyopencl.LocalMemory(self.dtype.itemsize * block * block),
                         pyopencl.LocalMemory(self.dtype.itemsize * block * block))
            return OpenCLArray(self.m,B.n,c_buf,None,self.dtype)

    def matrixvec(self, B):
        size = self.m
        nbytes = self.dtype.itemsize
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, size*nbytes)
        block = int(numpy.sqrt(self.max_block_size))
        blockx = min(block, self.m)
        blocky = min(block, self.n)

        #grid = (math.ceil(self.m / blockx) * blockx, math.ceil(self.n / blocky) * blocky)
        grid = (blockx, math.ceil(self.n / blocky) * blocky)
        print('grid: ', grid)
        print('block: ', (blockx, blocky))
        if self.dtype == numpy.float32:
            cl_function = self.prg.matrix_vec
        elif self.dtype == numpy.float64:
            cl_function = self.prg.double_matrix_vec

        cl_function(self.queue, grid, (blockx, blocky),
                            self.buf, B.buf, c_buf, 
                            pyopencl.LocalMemory(blockx*blocky*nbytes),
                            pyopencl.LocalMemory(blocky*nbytes), 
                            numpy.int32(self.n), numpy.int32(self.m))
        return OpenCLArray(self.m,1,c_buf,None,self.dtype)

    def vecdot(self, B):
        size = max(self.m, self.n)
        nbytes = self.dtype.itemsize
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, nbytes)
        block_size = min(self.max_block_size, size) 
        block = (block_size, 1)
        grid_size = self.m*self.n
        grid = (math.ceil(grid_size / block_size) * block_size, 1)
        print('grid: ', grid)
        print('block: ', block)
        if self.dtype == numpy.float32:
            cl_function = self.prg.vec_dot
        elif self.dtype == numpy.float64:
            cl_function = self.prg.double_vec_dot
        
        cl_function(self.queue, grid, block,
                            self.buf, B.buf, c_buf, 
                            pyopencl.LocalMemory(block_size*nbytes), 
                            numpy.int32(size))
        return OpenCLArray(1,1,c_buf,None,self.dtype)

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

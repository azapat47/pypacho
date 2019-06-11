from pypacho.anarray import GpuArray, AnArray
from pypacho.opencl import kernel
import pyopencl
import numpy
import math
import os

current_dir = os.path.abspath(os.path.dirname(__file__))

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
                options = ['-Werror', '-I', kernel.get_dir()]
            options = ['-w', '-I', kernel.get_dir()]
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
        size = self.m * self.n
        grid = (size,)
        dtype = self.dtype
        if self.dtype == numpy.float32:
            if B.dtype == numpy.float32:
                cl_function = self.prg.add_float_float
            elif B.dtype == numpy.float64:
                cl_function = self.prg.add_float_double
                dtype = numpy.dtype(numpy.float64)
            elif B.dtype == numpy.int32:
                cl_function = self.prg.add_float_int

        elif self.dtype == numpy.float64:
            if B.dtype == numpy.float32:
                cl_function = self.prg.add_double_float
            elif B.dtype == numpy.float64:
                cl_function = self.prg.add_double_double
            elif B.dtype == numpy.int32:
                cl_function = self.prg.add_double_int

        elif self.dtype == numpy.int32:
            if B.dtype == numpy.float32:
                cl_function = self.prg.add_int_float
                dtype = numpy.dtype(numpy.float32)
            elif B.dtype == numpy.float64:
                cl_function = self.prg.add_int_double
                dtype = numpy.dtype(numpy.float64)
            elif B.dtype == numpy.int32:
                cl_function = self.prg.add_int_int

        nbytes = size * dtype.itemsize
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, nbytes)
        cl_function(self.queue, grid, self.block_size,
                           self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None,dtype)
    
    def subtract(self,B):
        size = self.m * self.n
        grid = (size,)
        dtype = self.dtype
        if self.dtype == numpy.float32:
            if B.dtype == numpy.float32:
                cl_function = self.prg.subtract_float_float
            elif B.dtype == numpy.float64:
                cl_function = self.prg.subtract_float_double
                dtype = numpy.dtype(numpy.float64)
            elif B.dtype == numpy.int32:
                cl_function = self.prg.subtract_float_int

        elif self.dtype == numpy.float64:
            if B.dtype == numpy.float32:
                cl_function = self.prg.subtract_double_float
            elif B.dtype == numpy.float64:
                cl_function = self.prg.subtract_double_double
            elif B.dtype == numpy.int32:
                cl_function = self.prg.subtract_double_int

        elif self.dtype == numpy.int32:
            if B.dtype == numpy.float32:
                cl_function = self.prg.subtract_int_float
                dtype = numpy.dtype(numpy.float32)
            elif B.dtype == numpy.float64:
                cl_function = self.prg.subtract_int_double
                dtype = numpy.dtype(numpy.float64)
            elif B.dtype == numpy.int32:
                cl_function = self.prg.subtract_int_int

        nbytes = size * dtype.itemsize
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, nbytes)
        cl_function(self.queue, grid, self.block_size,
                           self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None,dtype)
    
    def multiply(self,B):
        size = self.m * self.n
        grid = (size,)
        dtype = self.dtype
        B_buf = None
        if isinstance(B, OpenCLArray):
            B_buf = B.buf

        if self.dtype == numpy.float32:
            if not isinstance(B, OpenCLArray):
                b_numpy = numpy.array(B)
                B_buf = b_numpy
                if b_numpy.dtype == numpy.float32:
                    cl_function = self.prg.scalar_mult_float_float
                elif b_numpy.dtype == numpy.float64:
                    cl_function = self.prg.scalar_mult_float_double
                    dtype = numpy.dtype(numpy.float64)
                elif b_numpy.dtype == numpy.int32:
                    cl_function = self.prg.scalar_mult_float_int

            elif B.dtype == numpy.float32:
                cl_function = self.prg.multiply_float_float
            elif B.dtype == numpy.float64:
                cl_function = self.prg.multiply_float_double
                dtype = numpy.dtype(numpy.float64)
            elif B.dtype == numpy.int32:
                cl_function = self.prg.multiply_float_int

        elif self.dtype == numpy.float64:

            if not isinstance(B, OpenCLArray):
                b_numpy = numpy.array(B)
                B_buf = b_numpy
                if b_numpy.dtype == numpy.float32:
                    cl_function = self.prg.scalar_mult_double_float
                elif b_numpy.dtype == numpy.float64:
                    cl_function = self.prg.scalar_mult_double_double
                    dtype = numpy.dtype(numpy.float64)
                elif b_numpy.dtype == numpy.int32:
                    cl_function = self.prg.scalar_mult_double_int

            elif B.dtype == numpy.float32:
                cl_function = self.prg.multiply_double_float
            elif B.dtype == numpy.float64:
                cl_function = self.prg.multiply_double_double
            elif B.dtype == numpy.int32:
                cl_function = self.prg.multiply_double_int

        elif self.dtype == numpy.int32:
            if not isinstance(B, OpenCLArray):
                b_numpy = numpy.array(B)
                B_buf = b_numpy
                if b_numpy.dtype == numpy.float32:
                    cl_function = self.prg.scalar_mult_int_float
                elif b_numpy.dtype == numpy.float64:
                    cl_function = self.prg.scalar_mult_int_double
                    dtype = numpy.dtype(numpy.float64)
                elif b_numpy.dtype == numpy.int32:
                    cl_function = self.prg.scalar_mult_int_int

            elif B.dtype == numpy.float32:
                cl_function = self.prg.multiply_int_float
                dtype = numpy.dtype(numpy.float32)
            elif B.dtype == numpy.float64:
                cl_function = self.prg.multiply_int_double
                dtype = numpy.dtype(numpy.float64)
            elif B.dtype == numpy.int32:
                cl_function = self.prg.multiply_int_int

        nbytes = size * dtype.itemsize
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, nbytes)
        cl_function(self.queue, grid, self.block_size,
                           self.buf, B_buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None,dtype)
    
    def divide(self,B):
        size = self.m * self.n
        grid = (size,)
        dtype = self.dtype
        if self.dtype == numpy.float32:
            if B.dtype == numpy.float32:
                cl_function = self.prg.divide_float_float
            elif B.dtype == numpy.float64:
                cl_function = self.prg.divide_float_double
                dtype = numpy.dtype(numpy.float64)
            elif B.dtype == numpy.int32:
                cl_function = self.prg.divide_float_int

        elif self.dtype == numpy.float64:
            if B.dtype == numpy.float32:
                cl_function = self.prg.divide_double_float
            elif B.dtype == numpy.float64:
                cl_function = self.prg.divide_double_double
            elif B.dtype == numpy.int32:
                cl_function = self.prg.divide_double_int

        elif self.dtype == numpy.int32:
            if B.dtype == numpy.float32:
                cl_function = self.prg.divide_int_float
                dtype = numpy.dtype(numpy.float32)
            elif B.dtype == numpy.float64:
                cl_function = self.prg.divide_int_double
                dtype = numpy.dtype(numpy.float64)
            elif B.dtype == numpy.int32:
                cl_function = self.prg.divide_int_int

        nbytes = size * dtype.itemsize
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, nbytes)
        cl_function(self.queue, grid, self.block_size,
                           self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None,dtype)
    
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

from gpu_array import GpuArray
from anarray import AnArray
import pyopencl
import numpy

class OpenCLArray(AnArray,GpuArray):
    ctx = None
    queue = None
    mf = None
    prg = None
    block_size = None
    ready = False

    def set_enviroment(block = None, options = None, kernel_params = None):
        if(not OpenCLArray.ready):
            with open('kernel.c') as file:
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

    def __init__(self,m,n,buf=None,host=None):
        self.m = m
        self.n = n
        if(host is None):
            self.buf = buf
        else:
            self.buf = pyopencl.Buffer\
                    (self.ctx,self.mf.READ_WRITE |self.mf.COPY_HOST_PTR, hostbuf=host)


    def transpose(self):
        C = numpy.zeros((self.m*self.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        
        self.prg.transpose(self.queue, C.shape, self.block_size,
                           c_buf, self.buf, numpy.uint32(self.m), numpy.uint32(self.n))
        return OpenCLArray(self.m,self.n,c_buf,None)


    def add(self,B):
        C = numpy.zeros((self.m*self.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        
        self.prg.add(self.queue, C.shape, self.block_size,
                           self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None)
    
    def subtract(self,B):
        C = numpy.zeros((self.m*self.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        
        self.prg.subtract(self.queue, C.shape, self.block_size,
                           self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None)
    
    def multiply(self,B):
        C = numpy.zeros((self.m*self.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        
        self.prg.multiply(self.queue, C.shape, self.block_size,
                           self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None)
    
    def divide(self,B):
        C = numpy.zeros((self.m*self.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        
        self.prg.divide(self.queue, C.shape, self.block_size,
                           self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None)
    
    def dot(self,B):
        C = numpy.zeros((self.m*B.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        
        self.prg.dot(self.queue, C.shape, self.block_size,self.buf, B.buf, c_buf,
                     numpy.uint32(self.m),
                     numpy.uint32(self.n),
                     numpy.uint32(B.n))
        return OpenCLArray(self.m,B.n,c_buf,None)

    def mod(self,B):
        C = numpy.zeros((self.m*self.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        
        self.prg.mod(self.queue, C.shape, self.block_size,
                           self.buf, B.buf,c_buf)
        return OpenCLArray(self.m,self.n,c_buf,None)

    def negative(self):
        C = numpy.zeros((self.m*self.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        
        self.prg.negative(self.queue, C.shape, self.block_size,
                           c_buf, self.buf)
        return OpenCLArray(self.m,self.n,c_buf,None)
    
    def to_numpy(self):
        C = numpy.zeros((self.m*self.n),dtype=numpy.float32)
        r = numpy.empty_like(C)
        pyopencl.enqueue_copy(self.queue, r, self.buf)
        return r.reshape(self.m,self.n)

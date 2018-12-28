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

    def set_enviroment(block = None, options = None, kernel_params = None):
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

    def __init__(self,m,n,buf=None,host=None):
        self.m = m
        self.n = n
        self.shape = (m,n)
        if(host is None):
            self.buf = buf
        else:
            self.buf = pyopencl.Buffer\
                    (self.ctx,self.mf.READ_WRITE |self.mf.COPY_HOST_PTR, hostbuf=host)

            
    def __del__(self):
        self.buf.release()

        
    def transpose(self):
        # Posible optimización calculando los bytes sin crear la matriz numpy
        C = numpy.zeros((self.m*self.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        
        self.prg.transpose(self.queue, C.shape, self.block_size,
                           c_buf, self.buf, numpy.uint32(self.m), numpy.uint32(self.n))
        return OpenCLArray(self.n,self.m,c_buf,None)


    def add(self,B):
        # Posible optimización calculando los bytes sin crear la matriz numpy
        C = numpy.zeros((self.m*self.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)

        # Agregar excepcion tamaño de la matriz b = tamaño matriz a
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
        
        if(not isinstance(B,OpenCLArray)):
            self.prg.scalar_mult(self.queue, C.shape, self.block_size,
                                self.buf,numpy.float32(B),c_buf)
        else:
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
        
        if(self.n == 1 and self.m != 1 and B.n == 1  and B.m != 1):
            return B.transpose().dot(self)
        else:

            self.prg.dot_matrix(self.queue, C.shape, self.block_size,self.buf, B.buf, c_buf,
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

    def sqrt(self):
        C = numpy.zeros((self.m*self.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        self.prg.sqrt_(self.queue, C.shape, self.block_size,
                           c_buf, self.buf)
        return OpenCLArray(self.m,self.n,c_buf,None)

    def diag(self):
        C = numpy.zeros((self.m), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        self.prg.diag(self.queue, C.shape, self.block_size,
                           self.buf, c_buf, numpy.uint32(self.m))
        return OpenCLArray(1,self.m,c_buf,None)

    def diagflat(self):
        C = numpy.zeros((self.n*self.n), dtype=numpy.float32)
        c_buf = pyopencl.Buffer(self.ctx,self.mf.READ_WRITE, C.nbytes)
        self.prg.diagflat(self.queue, C.shape, self.block_size,
                           self.buf, c_buf, numpy.uint32(self.n))
        return OpenCLArray(self.n,self.n,c_buf,None)
    
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
        C = numpy.zeros((self.m*self.n),dtype=numpy.float32)
        pyopencl.enqueue_copy(self.queue, C, self.buf)
        return C.reshape(self.m,self.n)
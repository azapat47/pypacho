from opencl_array import OpenCLArray
import numpy
import pyopencl

def norm(A):
    return A.norm()

def diag(A):
    C = numpy.zeros((A.m*1), dtype=numpy.float32)
    c_buf = pyopencl.Buffer(A.ctx,A.mf.READ_WRITE, C.nbytes)
    A.prg.diag(A.queue, C.shape, A.block_size,
                        A.buf, c_buf,numpy.int32(A.m))
    return OpenCLArray(A.m,1,c_buf,None)

def diagflat(A):
    C = numpy.zeros((A.m*A.m), dtype=numpy.float32)
    c_buf = pyopencl.Buffer(A.ctx,A.mf.READ_WRITE, C.nbytes)
    A.prg.diagflat(A.queue, (A.m,1), A.block_size,
                        c_buf, A.buf,numpy.int32(A.m))
    return OpenCLArray(A.m,A.m,c_buf,None)
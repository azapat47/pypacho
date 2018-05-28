from opencl_array import OpenCLArray
from our_cuda import OurCuda
import numpy
import pyopencl

def norm(A):
    return A.norm()

def diag(A):
    return A.diag()

def diagflat(A):
    return A.diagflat()
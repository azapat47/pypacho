import helper
from pypacho.anarray import AnArray
from pypacho.opencl import OpenCLArray
from pypacho.cuda import OurCuda
from methods.jacobi import jacobi
from methods.conjugate_gradient import conjugate_gradient
from methods.gradient_descent import gradient_descent, gradient_descent2
import numpy as np
import os
from time import time




def generate(size):
    A = np.random.randn(size,size).astype(np.float32)
    x = np.random.randn(size,1).astype(np.float32)
    

    return A,B,x

def turn_dominant(Matriz):
    pass

def main(size,delta):
    A_np,B_np,xorigin_np = generate(size)
    # Create CUDA array
    # Create OpenCL array
    test(jacobi,A,B,xorigin,N,tol,alpha=0)

# Generate

# Test Jacobi
def test(method, *args):
    # Initial Time
    method(*args)
    # Final Time
    print("**"*40)
    print("Error: ", end="")
    error_method = xorigin - x_Cuda.Matrix.get()
    error = np.linalg.norm(error_method, ord=2)/np.linalg.norm(xorigin, ord=2)
    print(error)


# Test Descendent

# Test Conjugate

main(str(sys.argv[1]),str(sys.argv[2]))

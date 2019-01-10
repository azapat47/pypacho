import helper
from pypacho.anarray import AnArray
from pypacho.cuda import OurCuda
from pypacho.opencl import OpenCLArray
from methods.jacobi import jacobi
from methods.jacobi import import_library as ja_import
from methods.conjugate_gradient import conjugate_gradient
from methods.conjugate_gradient import import_library as cg_import
from methods.gradient_descent import gradient_descent, gradient_descent2
from methods.gradient_descent import import_library as gd_import
import numpy as np
import os
from time import time
import collections
import pandas as pd
import json
import sys



# Modulo 2
def generate(size):
    A = np.random.randn(size,size).astype(np.float64)
    xv = np.random.randn(size,1).astype(np.float64)
    turn_dominant(A)
    B = A @ xv
    return A,B,xv

def turn_dominant(Matriz, delta = 0):
    for i in range(0, Matriz.shape[0]):
      sum = 0
      for j in range(0,Matriz.shape[1]):
        sum += abs(Matriz[i][j])
      Matriz[i][i] = sum + delta

      
# Arguments: EXIT_CODE: specify if any exit code is required       
def usage(ec=None):
  print("Usage - optirun python3 laboratory.py [args] ")
  print("Arguments: Iterable of Glob_params [trys per size ,initial size, delta, how many sizes will be tested,tolerance, iteretions per method]") 
  print("           Iterable of plataforms  [cuda, opencl, numpy] Plataforms, boolean value, in that order.")
  print("           Iterable of Methods     [jacobi, GD, CG] Methods, boolean value, in that order.")
  print('Example of call from console: $ optirun python3 laboratory.py "[1,10,10,5,100,0.001]" "[1,0,0]" "[1,0,0]"')
  if(ec is not None):
    exit(ec)

    
#Arguments: Iterable Glob_params [trys per size ,initial size, delta, how many sizes will be tested, iteretions per method,tolerance] The number of trys, the size of the matrix and the delta of growth. The first matrix is always with a size of delta. 
#           Iterable plataforms  [cuda, opencl, numpy] Plataforms, boolean values, in that order.
#           Iterable Methods     [jacobi, GD, CG] Methods, boolean values, in that order.
#Example of call from console: $ optirun python3 laboratory.py "[1,10,10,5,100,0.001]" "[1,0,0]" "[1,0,0]"
#Modulo 1
def main(Glob_params, plataforms, methods):
    #Params check
    if(not isinstance(Glob_params, collections.Iterable) or not isinstance(plataforms, collections.Iterable) or not isinstance(methods, collections.Iterable)):
      print("Bad type in any arguments. Use Lists")
      usage(1)
    if(len(Glob_params) != 6 or len(plataforms) != 3 or len(methods)!=3):
      print("Bad number of elements in any param")
      usage(1)
    integrity_glob_params = all(isinstance(i, int)
                               for i in Glob_params[:-1]) and isinstance(Glob_params[5],float)
    integrity_plataforms = all((isinstance(i, int) and (i == 0 or i == 1)) 
                               for i in plataforms)
    integrity_methods = all(isinstance(i, int) and (i == 0 or i == 1)
                            for i in methods)
    
    if(not integrity_glob_params or not integrity_plataforms or not integrity_methods):
      print("Bad type in any element of params' list")
      usage(1)
     
    print("LAB: ***Starting***")
    # Creationg pandas DataFrame
    fat_panda = pd.DataFrame(columns=["platform", "method", "size", "iterations", "time", "accuracy","dispersion" ])
    fat_panda["size"] = fat_panda["size"].astype(int)
    fat_panda["iterations"] = fat_panda["iterations"].astype(int)
    fat_panda["iterations"] = fat_panda["time"].astype(float)
    fat_panda["accuracy"] = fat_panda["accuracy"].astype(float)
    fat_panda["dispersion"] = fat_panda["dispersion"].astype(float)
    ini_size = Glob_params[1]
    delta = Glob_params[2]
    n = Glob_params[3]
    iter_meth = Glob_params[4]
    tolerance = Glob_params[5]
    #For every matrix size
    for matrix_size in range(ini_size, ini_size+(delta*n), delta): 
      print("LAB: Matrix size", matrix_size, end=" | ")
      print(str(int(matrix_size/delta)) + '/' + str(n))
      # Number Of trys  
      for tr in range(Glob_params[0]): 
        print("  LAB: Trying number", tr, end='... ')
        df = runner(matrix_size, iter_meth, tolerance, 0.001, plataforms[0], plataforms[1], plataforms[2], methods[0],  methods[1],  methods[2])
        fat_panda = pd.concat([fat_panda,df], ignore_index=True)
        print("DONE")
    
    print(fat_panda)
    fat_panda.to_csv("out.csv", sep=',')


        
# Test Any method
# The arguments must be [Method, Original X, and specific method args]
# modulo 3
def test(method, xv, *args):
    # Initial Time
    t_start = time()
    x_meth,iter,disp = method(*args)
    t = time() - t_start
    # Final Time
    if not isinstance(x_meth,np.ndarray):
      x = x_meth.to_numpy()
    else:
      x = x_meth
    #Error of the method
    error_method = xv - x
    error = np.linalg.norm(error_method, ord=2)/np.linalg.norm(xv, ord=2)
    
    return iter,t,error,disp


# Test Descendent

# Test Conjugate

def runner(size=100, N=100, tol=0.001, alpha=0.001,
      cuda=False,opencl=False,numpy=False,
      jaco=False, grad_descent=False, conj_grad=False):
  A,B,xv = generate(size)
  x_ini = np.ones(xv.shape).astype(np.float32)
  platform = []
  method = []
  Size = []
  iterations = []
  times = []
  errors = []
  dispersions = []
  #generate CPU
  if cuda == True:
    #create CUDA
    a_cuda = OurCuda(A.shape[0],A.shape[1],A,None)
    x_cuda = OurCuda(x_ini.shape[0],x_ini.shape[1],x_ini,None)
    b_cuda = OurCuda(B.shape[0],B.shape[1],B,None)
    if jaco == True:
      ja_import("pypacho")
      iter,time,error,dispersion = test(jacobi, xv, a_cuda, b_cuda,x_cuda, N, tol)
      platform.append("cuda")
      method.append("jacobi")
      Size.append(size)
      iterations.append(iter)
      times.append(time)
      errors.append(error)
      dispersions.append(dispersion)

    if grad_descent == True:
      gd_import("pypacho")
      iter,time,error,dispersion = test(gradient_descent2, xv, a_cuda, b_cuda, alpha, x_cuda, N, tol)
      platform.append("cuda")
      method.append("GD")
      Size.append(size)
      iterations.append(iter)
      times.append(time)
      errors.append(error)
      dispersions.append(dispersion)

    if conj_grad == True:
      cg_import("pypacho")
      iter,time,error,dispersion = test(conjugate_gradient, xv, a_cuda, b_cuda,x_cuda, N, tol)
      platform.append("cuda")
      method.append("CG")
      Size.append(size)
      iterations.append(iter)
      times.append(time)
      errors.append(error)
      dispersions.append(dispersion)

    # delete CUDA
    del(a_cuda)
    del(b_cuda)
    del(x_cuda)
  if opencl == True:
  #create opencl
    OpenCLArray.set_enviroment()
    a_cl = OpenCLArray(A.shape[0],A.shape[1],None,A)
    x_cl = OpenCLArray(x_ini.shape[0],x_ini.shape[1],None,x_ini)
    b_cl = OpenCLArray(B.shape[0],B.shape[1],None,B)
    

    if jaco == True:
      ja_import("pypacho")
      iter,time,error,dispersion = test(jacobi, xv, a_cl, b_cl,x_cl, N, tol)
      platform.append("opencl")
      method.append("jacobi")
      Size.append(size)
      iterations.append(iter)
      times.append(time)
      errors.append(error)
      dispersions.append(dispersion)
    if grad_descent == True:
      gd_import("pypacho")
      iter,time,error,dispersion = test(gradient_descent2, xv, a_cl, b_cl, alpha, x_cl, N, tol)
      platform.append("opencl")
      method.append("GD")
      Size.append(size)
      iterations.append(iter)
      times.append(time)
      errors.append(error)
      dispersions.append(dispersion)
    if conj_grad == True:
      cg_import("pypacho")
      iter,time,error,dispersion = test(conjugate_gradient, xv, a_cl, b_cl,x_cl, N, tol)
      platform.append("opencl")
      method.append("CG")
      Size.append(size)
      iterations.append(iter)
      times.append(time)
      errors.append(error)
      dispersions.append(dispersion)
    # delete openCL
    del(a_cl)
    del(b_cl)
    del(x_cl)
  if numpy == True:
    #exec numpy
    if jaco == True:
      ja_import("numpy")
      iter,time,error,dispersion = test(jacobi, xv, A, B,x_ini, N, tol)
      platform.append("numpy")
      method.append("jacobi")
      Size.append(size)
      iterations.append(iter)
      times.append(time)
      errors.append(error)
      dispersions.append(dispersion)
    if grad_descent == True:
      gd_import("numpy")
      iter,time,error,dispersion = test(gradient_descent2, xv, A, B, alpha, x_ini, N, tol)
      platform.append("numpy")
      method.append("GD")
      Size.append(size)
      iterations.append(iter)
      times.append(time)
      errors.append(error)
      dispersions.append(dispersion)
    if conj_grad == True:
      cg_import("numpy")
      iter,time,error,dispersion = test(conjugate_gradient, xv, A, B, x_ini, N, tol)
      platform.append("numpy")
      method.append("CG")
      Size.append(size)
      iterations.append(iter)
      times.append(time)
      errors.append(error)
      dispersions.append(dispersion)
  #delete cpu
  data = { 'platform': platform , 'method': method, 'size': Size,
  'iterations': iterations, 'time': times, 'accuracy' : errors, 'dispersion' : dispersions }
  dataframe = pd.DataFrame(data)
  return dataframe

  
  
#Example of call from console: $ python lab.py "[1,1,10]" "[1,0,0]" "[1,0,0]"

if __name__ == '__main__':
    if(len(sys.argv)!=4): usage(0)
    os.environ["PYOPENCL_CTX"]='0'
    main(json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3]))
import helper
from pypacho.opencl import OpenCLArray
from pypacho.anarray import AnArray
from pypacho.cuda import OurCuda
from methods.jacobi import jacobi
from methods.jacobi import import_library as ja_import
from methods.conjugate_gradient import conjugate_gradient
from methods.conjugate_gradient import import_library as cg_import
from methods.gradient_descent import gradient_descent, gradient_descent2
from methods.gradient_descent import import_library as gd_import
import pycuda.driver as drv
import numpy as np
import os
from time import time
import collections.abc as collections
import pandas as pd
import json
import sys



# Modulo 2
def generate(size,npFloatType):
  A = np.random.uniform(low=-10000, high=10000, size=(size,size)).astype(npFloatType)
  xv = np.random.uniform(low=-10000, high=10000, size=(size,1)).astype(npFloatType)
  turn_dominant(A)
  B = A @ xv
  return A,B,xv

def turn_dominant(Matriz, delta = 0):
    for i in range(0, Matriz.shape[0]):
      sum = 0
      for j in range(0,Matriz.shape[1]):
        sum += abs(Matriz[i][j])
      Matriz[i][i] = sum + delta

def turn_dominant2(Matriz):
    for i in range(0, Matriz.shape[0]):
      delta = float(np.random.uniform(0,Matriz.shape[0]))
      Matriz[i][i] = Matriz.shape[0] + delta

      
# Arguments: EXIT_CODE: specify if any exit code is required       
def usage(ec=None):
  print("Usage - python3 laboratory.py [args] ")
  print("Arguments: Iterable of Glob_params [trys per size ,initial size, delta, how many sizes will be tested,tolerance, iteretions per method]") 
  print("           Iterable of plataforms  [cuda, opencl, numpy] Plataforms, boolean value, in that order.")
  print("           Iterable of Methods     [jacobi, GD, CG] Methods, boolean value, in that order.")
  print("           Double or float flag    false|true boolean value, if true the lab is going to run in double presicion otherwise in float")
  print('Example of call from console: $ python3 laboratory.py "[1,10,10,5,100,0.001]" "[1,0,0]" "[1,0,0]" "false"')
  if(ec is not None):
    exit(ec)

    
#Arguments: Iterable Glob_params [trys per size ,initial size, delta, how many sizes will be tested, iteretions per method,tolerance] The number of trys, the size of the matrix and the delta of growth. The first matrix is always with a size of delta. 
#           Iterable plataforms  [cuda, opencl, numpy] Plataforms, boolean values, in that order.
#           Iterable Methods     [jacobi, GD, CG] Methods, boolean values, in that order.
#           Double or float flag false|true boolean value, if true the lab is going to run in double presicion otherwise in float
#Example of call from console: $ python3 laboratory.py "[1,10,10,5,100,0.001]" "[1,0,0]" "[1,0,0]" "false"
#Modulo 1
def main(Glob_params, plataforms, methods, double = False):
    assert isinstance(Glob_params, collections.Iterable), "Global params must be List, is " + str(type(Glob_params))
    assert isinstance(plataforms, collections.Iterable),  "Plaforms params must be List, is " + str(type(plataforms))
    assert isinstance(methods, collections.Iterable),  "Methods params must be List, is " + str(type(methods))
    assert len(Glob_params)==6, "Global Params must be 6 of length, but is " + str(len(Glob_params))
    assert len(plataforms)==3, "Platforms params must be 3 of length, but is " + str(len(plataforms))
    assert len(methods)==3, "Methods params must be 3 of length, but is " + str(len(plataforms))

    #Params values check
    integrity_glob_params = all(isinstance(i, int)
                               for i in Glob_params[:-1]) and isinstance(Glob_params[5],float)
    integrity_plataforms = all((isinstance(i, int) and (i == 0 or i == 1)) 
                               for i in plataforms)
    integrity_methods = all(isinstance(i, int) and (i == 0 or i == 1)
                            for i in methods)
    
    if(not integrity_glob_params or not integrity_plataforms or not integrity_methods):
      print("Bad type in any element of params' list")
      usage(1)
      
    doubleorfloat = ""  
    if(double):
      doubleorfloat = "double"
    else:
      doubleorfloat = "float"
    print("LAB: ***Starting in "+ doubleorfloat +" ***")
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
      print(str(int(((matrix_size-ini_size)/delta)+1)) + '/' + str(n))
      # Number Of trys  
      for tr in range(Glob_params[0]): 
        print("  LAB: Trying number", tr+1, end='... ')
        df = runner(matrix_size, iter_meth, tolerance,0.001, plataforms[0], plataforms[1], plataforms[2], methods[0],  methods[1],  methods[2], double)
        fat_panda = pd.concat([fat_panda,df], ignore_index=True)
        print("DONE")
    
    print(fat_panda)
    fat_panda.to_csv("out_" + doubleorfloat + ".csv", sep=',')


        
# Test Any method
# The arguments must be [Method, Original X, and specific method args]
# modulo 3
def test(method, xv, *args):
    # Time cuda
    if(isinstance(args[0],OurCuda)):
      start = drv.Event()
      end = drv.Event()
      start.record()
      start.synchronize()
      x_meth,iter,disp = method(*args)
      end.record() 
      end.synchronize() 
      t = start.time_till(end)*1e-3
    # Time OpenCL and Numpy
    else:  
      t_start = time()
      x_meth,iter,disp = method(*args)
      t = time() - t_start
    # Answer
    if not isinstance(x_meth,np.ndarray):
      x = x_meth.to_numpy()
    else:
      x = x_meth
    #Error of the method
    error_method = xv - x
    error = np.linalg.norm(error_method, ord=2)/np.linalg.norm(xv, ord=2)
    
    return iter,t,error,disp

def runner(size=100, N=100, tol=0.001, alpha=0.001,
      cuda=False,opencl=False,numpy=False,
      jaco=False, grad_descent=False, conj_grad=False, double = False):
  if double:
    npFloatType = np.float64
  else:
    npFloatType = np.float32
  A,B,xv = generate(size,npFloatType)
  x_ini = np.ones(xv.shape).astype(npFloatType)
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

if __name__ == '__main__':
  if(not(len(sys.argv)==4 or len(sys.argv)==5)): usage(0)
  os.environ["PYOPENCL_CTX"]='0'
  if len(sys.argv)==4:
    main(json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3]))
  else:
    main(json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3]),json.loads(sys.argv[4]))

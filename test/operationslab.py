import helper
from pypacho.cuda import OurCuda
from pypacho.opencl import OpenCLArray
import numpy
import pypacho
import os
from time import time
import collections
import pandas as pd
import json
import sys

def generate(size_n,size_m):
    A = numpy.random.randn(size_n,size_m).astype(numpy.float32)
    turn_dominant(A)
    return A

def turn_dominant(Matriz, delta = 0):
    for i in range(0, Matriz.shape[0]):
      sum = 0
      for j in range(0,Matriz.shape[1]):
        sum += abs(Matriz[i][j])
      Matriz[i][i] = sum + delta
      
def usage(ec=None):
  print("Usage - optirun python3 op_laboratory.py [args] ")
  print("Arguments: Iterable of Glob_params [trys per size ,initial_size_a, initial_size_b, delta, how many sizes will be tested]") 
  print("           Iterable of plataforms  [cuda, opencl, numpy] Plataforms, boolean value, in that order.")
  print("           Iterable of Methods     [+, -, @, /,  *, transp, norm] Methods, boolean value, in that order.")
  print('Example of call from console: $ optirun python3 operationslab.py "[1,10,10,10,5]" "[1,0,0]" "[1,0,0,0,0,0,0]"')
  if(ec is not None):
    exit(ec)
    

def main(Glob_params, plataforms, methods):
    #Params check
    if(not isinstance(Glob_params, collections.Iterable) or not isinstance(plataforms, collections.Iterable) or not isinstance(methods, collections.Iterable)):
      print("Bad type in any arguments. Use Lists")
      usage(1)
    if(len(Glob_params) != 5 or len(plataforms) != 3 or len(methods)!=7):
      print("Bad number of elements in any param")
      usage(1)
    integrity_glob_params = all(isinstance(i, int)
                               for i in Glob_params)
    integrity_plataforms = all((isinstance(i, int) and (i == 0 or i == 1)) 
                               for i in plataforms)
    integrity_methods = all(isinstance(i, int) and (i == 0 or i == 1)
                            for i in methods)
    
    if(not integrity_glob_params or not integrity_plataforms or not integrity_methods):
      print("Bad type in any element of params' list")
      usage(1) 
    print("LAB: ***Starting***")
    # Creationg pandas DataFrame
    fat_panda = pd.DataFrame(columns=["platform", "method", "size_a", "size_b", "time"])
    fat_panda["size_a"] = fat_panda["size_a"].astype(int)
    fat_panda["size_b"] = fat_panda["size_b"].astype(int)
    ini_size_a = Glob_params[1]
    ini_size_b = Glob_params[2]
    delta = Glob_params[3]
    n = Glob_params[4]
    #For every matrix size_a
    for matrix_size in range(ini_size_a, ini_size_a+(delta*n), delta): 
      print("LAB: Matrix size", matrix_size, end=" | ")
      print(str(int(matrix_size/delta)) + '/' + str(n))
      # Number Of trys  
      for tr in range(Glob_params[0]): 
        print("  LAB: Trying number", tr, end='... ')
        df = runner(matrix_size, matrix_size, matrix_size, matrix_size, plataforms[0], plataforms[1], plataforms[2], methods[0],  methods[1],  methods[2], methods[3],  methods[4],  methods[5],  methods[6])
        fat_panda = pd.concat([fat_panda,df], ignore_index=True)
        print("DONE")
    
    print(fat_panda)
    fat_panda.to_csv("out_op.csv", sep=',')  
    
    
    
    
    
def test(method, *args):
    # Initial Time
    t_start = time()
    result = method(*args)
    t = time() - t_start
    # Final Time
    return t

def runner(size_n_a,size_m_a,size_n_b,size_m_b,
      cuda=False,opencl=False,numpy=False,
      suma=False, resta=False, punto=False, divi=False,
      multi=False, transp=False, norma=False):
  A = generate(size_n_a,size_m_a)
  B = generate(size_n_b,size_m_b)
  platform = []
  method = []
  Size = []
  Size_b = [] 
  times = []
  #generate CPU
  def add(A, B):
    return A + B
  def sub(A, B):
    return A - B
  def dot(A, B):
    return A @ B
  def div(A, B):
    return A / B
  def mul(A,B):
    return A * B
  def transpose(A):
    return A.transpose()
  def norm(A):
    return np.linalg.norm(A)
  if cuda == True:
    np = pypacho
    #create CUDA
    a_cuda = OurCuda(A.shape[0],A.shape[1],A,None)
    b_cuda = OurCuda(B.shape[0],B.shape[1],B,None)
    if suma == True:
      time = test(add, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("suma")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if resta == True:
      time = test(add, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("resta")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if punto == True:
      time = test(add, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("punto")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)  
    if divi == True:
      time = test(add, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("division")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if multi == True:
      time = test(add, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("multiplicacion")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if transp == True:
      time = test(add, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("transpuesta")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if norm == True:
      time = test(norma, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("norma")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    # delete CUDA
    del(a_cuda)
    del(b_cuda)
  if opencl == True:
    np = pypacho
  #create opencl
    OpenCLArray.set_enviroment()
    a_cl = OpenCLArray(A.shape[0],A.shape[1],None,A)
    b_cl = OpenCLArray(B.shape[0],B.shape[1],None,B)

    if suma == True:
      time = test(add, a_cl, b_cl)
      platform.append("cl")
      method.append("suma")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if resta == True:
      time = test(add, a_cl, b_cl)
      platform.append("cl")
      method.append("resta")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if punto == True:
      time = test(add, a_cl, b_cl)
      platform.append("cl")
      method.append("punto")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)  
    if divi == True:
      time = test(add, a_cl, b_cl)
      platform.append("cl")
      method.append("division")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if multi == True:
      time = test(add, a_cl, b_cl)
      platform.append("cl")
      method.append("multiplicacion")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if transp == True:
      time = test(add, a_cl, b_cl)
      platform.append("cl")
      method.append("transpuesta")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if norm == True:
      time = test(norma, a_cl, b_cl)
      platform.append("cl")
      method.append("norma")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    # delete opencl
    del(a_cl)
    del(b_cl)
    
  if numpy == True:
    np = numpy
    #exec numpy
    if suma == True:
      time = test(add, A, B)
      platform.append("numpy")
      method.append("suma")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if resta == True:
      time = test(add, A, B)
      platform.append("numpy")
      method.append("resta")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if punto == True:
      time = test(add, A, B)
      platform.append("numpy")
      method.append("punto")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)  
    if divi == True:
      time = test(add, A, B)
      platform.append("numpy")
      method.append("division")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if multi == True:
      time = test(add, A, B)
      platform.append("numpy")
      method.append("multiplicacion")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if transp == True:
      time = test(add, A, B)
      platform.append("numpy")
      method.append("transpuesta")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
    if norm == True:
      time = test(norma, A, B)
      platform.append("numpy")
      method.append("norma")
      Size.append(size_n_a)
      Size_b.append(size_n_b)
      times.append(time)
  #delete cpu
  data = { 'platform': platform , 'method': method, 'size_a': Size,
  'size_b': Size_b, 'time': times,}
  dataframe = pd.DataFrame(data)
  return dataframe


if __name__ == '__main__':
    if(len(sys.argv)!=4): usage(0)
    os.environ["PYOPENCL_CTX"]='0'
    main(json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3]))

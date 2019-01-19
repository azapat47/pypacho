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

np = None

def generate(size_n,size_m):
    A = numpy.random.randn(size_n,size_m).astype(numpy.float32)
    #turn_dominant(A)
    return A

def generate_a(size_n,size_m):
    A = numpy.random.randn(size_n,size_m).astype(numpy.float32)
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
  print('Example of call from console: $ optirun python3 operationslab.py "[1,10,10,10,10,10,5]" "[1,0,0]" "[1,0,0,0,0,0,0]"')
  if(ec is not None):
    exit(ec)
    

def main(Glob_params, plataforms, methods):
    #Params check
    if(not isinstance(Glob_params, collections.Iterable) or not isinstance(plataforms, collections.Iterable) or not isinstance(methods, collections.Iterable)):
      print("Bad type in any arguments. Use Lists")
      usage(1)
    if(len(Glob_params) != 7 or len(plataforms) != 3 or len(methods)!=7):
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
    fat_panda = pd.DataFrame(columns=["platform", "method", "size_a_n", "size_a_m", "size_b_n", "size_b_m", "time"])
    fat_panda["size_a_n"] = fat_panda["size_a_n"].astype(int)
    fat_panda["size_a_m"] = fat_panda["size_a_m"].astype(int)
    fat_panda["size_b_n"] = fat_panda["size_b_n"].astype(int)
    fat_panda["size_b_m"] = fat_panda["size_b_m"].astype(int)
    ini_size_a_n = Glob_params[1]
    ini_size_a_m = Glob_params[2]
    ini_size_b_n = Glob_params[3]
    ini_size_b_m = Glob_params[4]
    if(methods[2] == 1 and ini_size_a_m != ini_size_b_n):
      print("Incompatible sizes for a dot operation")
      usage(1) 
    delta = Glob_params[5]
    n = Glob_params[6]
    an = 0
    am = 0
    bn = 0
    bm = 0
    #For every matrix size_a
    for matrix_size in range(ini_size_a_m, ini_size_a_m+(delta*n), delta): 
      print("LAB: Matrix size", matrix_size, end=" | ")
      print(str(int(matrix_size/delta)) + '/' + str(n))
      # Number Of trys  
      if(ini_size_a_n == 1):
        an = 1
      else:
        an = matrix_size
      if(ini_size_a_m == 1):
        am = 1
      else:
        am = matrix_size
      if(ini_size_b_n == 1):
        bn = 1
      else:
        bn = matrix_size
      if(ini_size_b_m == 1):
        bm = 1
      else:
        bm = matrix_size
      for tr in range(Glob_params[0]): 
        print("  LAB: Trying number", tr, end='... ')
        df = runner(an, am, bn, bm, plataforms[0], plataforms[1], plataforms[2], methods[0],  methods[1],  methods[2], methods[3],  methods[4],  methods[5],  methods[6])
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

def runner(an,am,bn,bm,
      cuda=False,opencl=False,numpy=False,
      suma=False, resta=False, punto=False, divi=False,
      multi=False, transp=False, norma=False):
  global np
  if(an == 1 or am == 1):
    A = generate_a(an,am)
  else:
    A = generate(an,am)
  if(bn == 1 or bm == 1):
    B = generate_a(bn,bm)
  else:
    B = generate(bn,bm)
  x = generate_a(an,1)
  platform = []
  method = []
  Size_a_n = []
  Size_a_m = []
  Size_b_n = [] 
  Size_b_m = []
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
    import pypacho
    np = pypacho
    #create CUDA
    a_cuda = OurCuda(A.shape[0],A.shape[1],A,None)
    b_cuda = OurCuda(B.shape[0],B.shape[1],B,None)
    x_cuda = OurCuda(x.shape[0],x.shape[1],x,None)
    if suma == True:
      time = test(add, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("suma")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if resta == True:
      time = test(sub, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("resta")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if punto == True:
      time = test(dot, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("punto")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)  
    if divi == True:
      time = test(div, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("division")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if multi == True:
      time = test(mul, a_cuda, b_cuda)
      platform.append("cuda")
      method.append("multiplicacion")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if transp == True:
      time = test(transpose, a_cuda)
      platform.append("cuda")
      method.append("transpuesta")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if norma == True:
      time = test(norm, x_cuda)
      platform.append("cuda")
      method.append("norma")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    # delete CUDA
    del(a_cuda)
    del(b_cuda)
  if opencl == True:
    import pypacho
    np = pypacho
  #create opencl
    OpenCLArray.set_enviroment()
    a_cl = OpenCLArray(A.shape[0],A.shape[1],None,A)
    b_cl = OpenCLArray(B.shape[0],B.shape[1],None,B)
    x_cl = OpenCLArray(x.shape[0],x.shape[1],None,x)

    if suma == True:
      time = test(add, a_cl, b_cl)
      platform.append("cl")
      method.append("suma")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if resta == True:
      time = test(sub, a_cl, b_cl)
      platform.append("cl")
      method.append("resta")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if punto == True:
      time = test(dot, a_cl, b_cl)
      platform.append("cl")
      method.append("punto")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)  
    if divi == True:
      time = test(div, a_cl, b_cl)
      platform.append("cl")
      method.append("division")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if multi == True:
      time = test(mul, a_cl, b_cl)
      platform.append("cl")
      method.append("multiplicacion")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if transp == True:
      time = test(transpose, a_cl)
      platform.append("cl")
      method.append("transpuesta")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if norma == True:
      time = test(norm, x_cl)
      platform.append("cl")
      method.append("norma")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    # delete opencl
    del(a_cl)
    del(b_cl)
    
  if numpy == True:
    import numpy
    np = numpy
    #exec numpy
    if suma == True:
      time = test(add, A, B)
      platform.append("numpy")
      method.append("suma")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if resta == True:
      time = test(sub, A, B)
      platform.append("numpy")
      method.append("resta")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if punto == True:
      time = test(dot, A, B)
      platform.append("numpy")
      method.append("punto")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)  
    if divi == True:
      time = test(div, A, B)
      platform.append("numpy")
      method.append("division")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if multi == True:
      time = test(mul, A, B)
      platform.append("numpy")
      method.append("multiplicacion")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if transp == True:
      time = test(transpose, A)
      platform.append("numpy")
      method.append("transpuesta")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
    if norma == True:
      time = test(norm, x)
      platform.append("numpy")
      method.append("norma")
      Size_a_n.append(an)
      Size_a_m.append(am)
      Size_b_n.append(bn)
      Size_b_m.append(bm)
      times.append(time)
  #delete cpu
  data = { 'platform': platform , 'method': method, 'size_a_n': Size_a_n,
  'size_a_m': Size_a_m, 'size_b_n': Size_b_n, 'size_b_m': Size_b_m,'time': times,}
  dataframe = pd.DataFrame(data)
  return dataframe


if __name__ == '__main__':
    if(len(sys.argv)!=4): usage(0)
    os.environ["PYOPENCL_CTX"]='0'
    main(json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3]))

from laboratory import main as lab
from operationslab import main as op
import os
import sys
import contextlib
import json


def limits(precision=False):
    doubleorfloat = ""
    if(precision):
      doubleorfloat = "double"
    else:
      doubleorfloat = "float"
    print("Running limits lab with "+ doubleorfloat +" precision...\n")
    valLimits = {}
    ini_size = 12000
    step = 1000
    plat = [0,0,0]
    method = [0,0,0]
    plat_names = ["cuda","opencl","numpy"]
    method_names = ["jacobi","GD","CG"]
    for i in range(2):
        valLimits[plat_names[i]] = {}
        plat = [0,0,0]
        plat[i] = 1
        for j in range(3):
            method = [0,0,0]
            method[j] = 1
            size = ini_size
            step = 1000
            ## Look for at least minimum value
            find_minimum=True
            while(find_minimum):
               try:
                    print("testing size (minimum): " + str(size) + " with " + method_names[j] + " in " + plat_names[i])
                    with contextlib.redirect_stdout(None):
                        lab([1,size,size,1,100,0.0000001],plat,method,precision)
                        find_minimum = False
                        if(size<ini_size):
                            step=step/10
                        size = size+step
               except Exception as inst:
                    size = size - step
                    if(size==0):
                        valLimits[plat_names[i]][method_names[j]]=size
                        break
            while not find_minimum:
                try:
                    print("testing size: " + str(size) + " with " + method_names[j] + " in " + plat_names[i])
                    with contextlib.redirect_stdout(None):
                        lab([1,size,size,1,100,0.0000001],plat,method,precision)
                        size = size + step
                except Exception as inst:
                    size = size - step
                    if (step <= 10):
                        valLimits[plat_names[i]][method_names[j]]=size
                        break
                    else:
                        step = step/10
                        size = size + step
                    
    print(valLimits)
            
def usage(ec=None):
  print("Calculates the maximun size linear system ecuation problem (Ax=b) that could be solve in this computer")
  print("Usage - python3 limits_lab.py [args] ")
  print("Arguments: Precision: [double|simple]. Default value: simple")
  print('Example of call from console: $python3 limits_lab double')
  if(ec is not None):
    exit(ec)        


if __name__ == '__main__':
    len_params = len(sys.argv)
    if(len_params>2): usage(0)
    if(len_params==2):
      ps = sys.argv[1]
      if(ps == "double"):  
        limits(precision=True)
      elif(ps == "simple"):
        limits()
      else:
         print("Argument '"+ ps+ "' is not valid\n") 
         usage(1)  
    else:
       limits()

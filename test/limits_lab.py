from laboratory import main as lab
from operationslab import main as op
import os
import sys
import contextlib
import json
import time

# Iterative Binary Search Function 
def binarySearch(r,plat,method,precision): 
    delta = 100
    l = 0
    realVal = 0
    while l <= r: 
        mid = l + (r - l)/2 
        try:
            print("Trying minimum... " + str(mid) + " [" + str(l) + " , "+ str(r) + "]...", end="")
            print("Safe wait for memory allocation... ", end="")
            time.sleep(60)
            print("End Wait")
            with contextlib.redirect_stdout(None):
              lab([1,int(mid),int(mid),1,100,0.0000001],plat,method,precision)
            print(" Done")
            realVal = mid
            l = mid + delta    
        except Exception as inst:
            print(" Failed")
            print("Catched ERROR:")
            print(inst)
            r = mid - delta  
    return int(realVal)

def limits(precision=False):
    doubleorfloat = ""
    if(precision):
      doubleorfloat = "double"
    else:
      doubleorfloat = "float"
    print("Running limits lab with "+ doubleorfloat +" precision...\n")
    valLimits = {}
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
            print("LAB: Testing " + method_names[j] + " in " + plat_names[i], end="... ")
            size = binarySearch(50000,plat,method,precision)
            valLimits[plat_names[i]][method_names[j]]=size
            print("" + str(size))
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

from laboratory import main as lab
from operationslab import main as op
import os
import sys
import contextlib
import json
import time
import argparse

# Iterative Binary Search Function 
def binarySearch(r,plat,method,precision): 
    delta = 100
    l = 0
    realVal = 0
    while l <= r: 
        mid = l + (r - l)/2 
        try:
            print("Trying minimum... " + str(mid) + " [" + str(l) + " , "+ str(r) + "]...", end="")
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

def limits(method,doubleorfloat):
    precision= doubleorfloat=='double'
    print("Running limits lab with "+ doubleorfloat +" precision...\n")
    valLimits = {}
    plat = [0,0,0]
    methods = [0,0,0]
    plat_names = ["cuda","opencl","numpy"]
    method_names = ["Jacobi","GD","CG"]
    for i in range(1):
        valLimits[plat_names[i]] = {}
        plat = [0,0,0]
        plat[i] = 1
        methods = [0,0,0]
        methods[method_names.index(method)] = 1
        print("LAB: Testing " + method + " in " + plat_names[i] + "...")
        size = binarySearch(50000,plat,methods,precision)
        valLimits[plat_names[i]][method]=size
        print("" + str(size))
    print(valLimits)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates the maximum size of a linear system ecuation problem (Ax=b) that could be solve in this computer')
    parser.add_argument("--precision", default="simple", type=str, help="Generated system will use double or simple precision. Default simple",  choices=["simple","double"])
    parser.add_argument("--method", required=True, type=str, help="Method to be executed", choices=["Jacobi","GD","CG"])
    args = parser.parse_args()
    os.environ["PYOPENCL_CTX"]='0'
    limits(args.method,args.precision)

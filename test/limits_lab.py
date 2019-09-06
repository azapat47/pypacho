from laboratory import main as lab
from operationslab import main as op
import os
import sys
import contextlib


def limits():
    valLimits = {}
    ini_size = 14000
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
            while True:
                try:
                    print("testing size: " + str(size) + " with " + method_names[j] + " in " + plat_names[i])
                    with contextlib.redirect_stdout(None):
                        lab([1,size,size,1,100,0.0000001],plat,method)
                        size = size + 1000
                except Exception as inst:
                    valLimits[plat_names[i]][method_names[j]]=size
                    break
    print(valLimits)
            
        


if __name__ == '__main__':
    limits()
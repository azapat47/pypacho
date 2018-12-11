import helper

np = None

def import_library(library = 'pypacho'):
    global np
    if library == 'pypacho':
        import pypacho
        np = pypacho
    else:
        import numpy
        np = numpy

import numpy as np

def norm(x):
    d = x.transpose()
    cross = d @ x
    return np.sqrt(float(cross))

def conjugate_gradient(A,b,x0,N=25,tol=0.001):
    x = x0
    r = b - (A @ x)
    s = r
    iter = 0
    error = tol + 1
    while (error > tol)&(iter <= N):
        q = A @ s
        alpha = float((s.transpose() @ r)/(s.transpose() @ q))
        x = x + alpha*s
        r = b - (A @ x)
        beta = -float((r.transpose() @ q)/(s.transpose() @ q))
        #print(beta)
        s = r + beta*s
        iter = iter + 1
        error = norm(r) 
    if (iter > N):
        #print("Se ha excedido el número de iteraciones. Procedimiento FALLIDO")
        return x,iter
    else:
        return x,iter
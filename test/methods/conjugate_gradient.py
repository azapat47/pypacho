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


def conjugate_gradient(A,b,x0,N=25,tol=0.001):
    x = x0
    r = b - (A @ x)
    s = r
    iter = 0
    error = tol + 1
    while (error > tol)&(iter <= N):
        q = A @ s
        S_t = s.transpose()
        alpha = float((S_t @ r)/(S_t @ q))
        xn = x + alpha*s
        r = b - (A @ xn)
        beta = -float((r.transpose() @ q)/(S_t @ q))
        #print(beta)
        s = r + beta*s
        iter = iter + 1
        error = np.linalg.norm(xn - x) / np.linalg.norm(xn)
        x = xn
    if (iter > N):
        #print("Se ha excedido el número de iteraciones. Procedimiento FALLIDO")
        return x,iter,error
    else:
        return x,iter,error
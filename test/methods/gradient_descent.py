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



def gradient_descent(A,b,alpha,x0,N=25,tol=0.01):
    iter=0       
    error = tol+1  
    x = x0
    while (error > tol)&(iter <= N):
        grad = A.transpose() @ (A@x - b)
        xn = x - alpha*grad
        error = norm(x - xn) / norm(xn)
        x = xn
        iter = iter + 1 
    if (iter > N):
        #print("Se ha excedido el número de iteraciones. Procedimiento FALLIDO")
        return x,iter
    else:
        return x,iter

def gradient_descent2(A,b,alpha,x0,N=25,tol=0.01):
    iter=0       
    error = tol+1  
    x = x0
    A_t = A.transpose()
    grad = A_t @ (A@x - b)
    while (error > tol)&(iter <= N):
        xn = x - alpha*grad
        xn_m_x = xn -x
        grandn = A_t @ (A@xn - b)
        grandn_m_0 = grandn - grad
        alpha = float((xn_m_x.transpose() @ grandn_m_0) / (grandn_m_0.transpose() @ grandn_m_0)) 
        error = np.linalg.norm(xn - x) / np.linalg.norm(xn)
        x = xn
        grad = grandn
        iter = iter + 1 
    if (iter > N):
        print("Se ha excedido el número de iteraciones. Procedimiento FALLIDO")
        return x,iter,error
    else:
        return x,iter,error
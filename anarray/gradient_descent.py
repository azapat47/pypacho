from opencl_array import OpenCLArray

def gradient_descent(A,b,alpha,x0,np,N=25,tol=0.01):
    iter=0       
    error = tol+1  
    x = x0
    while (error > tol)&(iter <= N):
        grad = A.transpose() @ (A@x - b)
        xn = x - alpha*grad
        error = np.norm(x - xn)
        x = xn
        iter = iter + 1 
    if (iter > N):
        print("Se ha excedido el número de iteraciones. Procedimiento FALLIDO")
    else:
        return x
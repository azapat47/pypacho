from opencl_array import OpenCLArray

def conjugate_gradient(A,b,x0,np,N=25,error=0.001):
    x = x0
    r = b - (A @ x)
    s = r
    iter = 0
    while (np.norm(r) > error)&(iter <= N):
        q = A @ s
        alpha = float((s.transpose() @ r)/(s.transpose() @ q))
        x = x + alpha*s
        r = b - (A @ x)
        beta = -float((r.transpose() @ q)/(s.transpose() @ q))
        #print(beta)
        s = r + beta*s
        iter = iter + 1 
    if (iter > N):
        print("Se ha excedido el número de iteraciones. Procedimiento FALLIDO")
    else:
        return x
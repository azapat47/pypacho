from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot, sqrt

def infNorm(x, xn, absolute=None):
    if absolute is not None:
        return max(abs(xn - x))
    else:
        return max(abs(xn - x))/max(abs(xn))

def EucNorm(x, xn, absolute=None):
    if absolute is not None:
        return sqrt(sum((xn-x)**2))
    else:    
        return sqrt(sum((xn-x)**2))/sqrt(sum(xn**2))


def jacobi(A,b,N=25,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times
    print("Tabla de soluciones" + "\tNomInf\tNormEuclidiana")
    print("-"*40)   
    i=0                         
    while i < N:
       print(str(i) +" " + str(x), end="")
       xn = (b - dot(R,x)) / D
       VecError = x - xn
       #print("\t" + str(infNorm(x, xn, absolute=True)) +"\t" + str(EucNorm(x, xn, absolute=True)))
       print("\t\t" + str(infNorm(x, xn)) +"\t" + str(EucNorm(x, xn))) #with relative error
       x = xn
       i+=1
    return x

A = array([[15,4,3],[2,8,2],[4,3,9]])
b = array([4,8,9])
guess = array([0,0,0])

print("A:")
print(A)

print("b:")
print(b)

sol = jacobi(A,b,N=25,x=guess)

print("-"*40)
print("x:")
print(sol)

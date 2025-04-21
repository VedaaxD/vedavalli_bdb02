#Find the eigenvalues of the given Hessian at the given point a. [12x^2 -1; -1 2] at (3,1)
'''
Hessian Matrix:
The Hessian matrix is a square matrix of second-order partial derivatives of a scalar-valued function f.
It tells us how the function curves in different directions
Properties:
1.If all eigen values > 0 (positive definite)  the point is a local minimum
2.If all eigen values < 0 (negative definite) the point is a local maximum
3.If eigen values have mixed signs -saddle point (gradient 0)

'''
import numpy as np
import sympy as sp
def find_eigen_values(a,b,c,d): #for the given 2X2 matrix
    trace=a+d
    determinant=(a*d)-(b*c)
    discriminant=(trace**2 )- (4*determinant)
    sqrt_discriminant=sp.sqrt(discriminant)
    #when we expand the det(A-lambdaI)=0 for solving lambda, below is the eqn (lambda1 and lambda2)
    lambda1=(trace+sqrt_discriminant)/2
    lambda2=(trace-sqrt_discriminant)/2
    return lambda1,lambda2
def main():
    x=3
    a=12*x**2
    b=-1
    c=-1
    d=2
    eigen1,eigen2=find_eigen_values(a,b,c,d)
    print(F"Eigenvalues of Hessian at (3,1):{eigen1},{eigen2}")
    if eigen1>0 and eigen2>0:
        print(f"Hessian is positive definite at (3,1)")
    else:
        print(f"Hessian is negative definite at (3,1)")

if __name__=='__main__':
    main()

# #using eigvalues we can compute directly
# H = np.array([[12*3**2, -1], [-1, 2]])   # x=3
# eigvals = np.linalg.eigvals(H)
# print("Eigenvalues of Hessian at (3,1):", eigvals)
# if np.all(eigvals > 0):
#     print("Hessian is positive definite at (3,1).")
# elif np.all(eigvals < 0):
#     print("Hessian is negative definite at (3,1).")
# else:
#     print("Hessian is indefinite at (3,1).")
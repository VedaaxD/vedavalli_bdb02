import sympy as sp
from Lab27_ex2 import find_eigen_values
#Determine the concavity of
#f(x, y) = x^3 + 2y^3 -xy  at  (i) (0,0,  (ii) (3, 3),  (iii) (3, -3)
"""
About Concavity:
It tells us the shape or curvature of a function — whether it bends upward or downward.
1. If eigenvalues of Hessian are positive, then local minimum (concave up).
2. If eigenvalues of Hessian are negative, then local maximum (concave down).
3. If mixed signs, then it's a saddle point (change in curvature).
"""
import sympy as sym

def second_derivatives(f,x_val,y_val):
    x,y=sym.symbols('x y') #giving two var x and y to treat as math symbols inorder to differentiate
    f_xx=sym.diff(f,x,x) #d²f/dx²
    f_yy=sym.diff(f,y,y) # d²f/dy²
    f_xy=sym.diff(f,x,y)  # d²f/dxdy
    f_yx=sym.diff(f,y,x) # d²f/dydx
    sub={x:x_val,y:y_val}
    return (f_xx.subs(sub),f_yy.subs(sub),f_xy.subs(sub),f_yx.subs(sub))

def construct_hessian_matrix(f_xx,f_yy,f_xy,f_yx):
    return [[f_xx,f_xy],[f_yx,f_yy]]

def determine_concavity(H):
    eigen_values=find_eigen_values(H[0][0],H[0][1],H[1][0],H[1][1]) #Extracts elements from 2X2 matrix and
    #passes to the eigenval func
    e_values=[sp.N(ev) for ev in eigen_values] #converts symbolic eigenval to numerical values using sp.N
    if e_values[0]>0 and e_values[1]>0:
        return "Concave up (local minimum-positive eigenvalue)"
    elif e_values[0]<0 and e_values[1]<0:
        return "Concave down (local maximum-negative eigenvalue)"
    else:
        return "Indefinite concavity (mixed signs-saddle point)"
def check_concavity(f,points):
    for point in points:
        x_val,y_val=point
        f_xx,f_yy,f_xy,f_yx=second_derivatives(f,x_val,y_val)
        H=construct_hessian_matrix(f_xx,f_yy,f_xy,f_yx)
        concavity=determine_concavity(H)
        print(f"f At point {point}:")
        print(f"\nHessian:{H}")
        print(f"Concavity:{concavity}\n")


def main():
    x,y=sym.symbols('x y')
    f=x**3 + 2*y**3 -x*y #func in qn
    points=[(0,0),(3,3),(3,-3)]
    check_concavity(f,points)
if __name__ == "__main__":
    main()
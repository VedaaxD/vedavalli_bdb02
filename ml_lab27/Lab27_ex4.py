#f(x, y) = 4x + 2y - x^2 - 3y^2
#a)Find the gradient. Use that to find critical points, (x, y) that makes gradient 0
#b)Use the Eigenvalues of the Hessian at the point to determine whether the critical point is a minimum, maximum or neither

import sympy as sp
from Lab27_ex2 import find_eigen_values
from Lab27_ex3 import second_derivatives,construct_hessian_matrix

x,y=sp.symbols('x,y')
f= 4*x + 2*y - x**2 -3*y**2
f_x=sp.diff(f,x) #expr of partial derivative of f wrt x
f_y=sp.diff(f,y) #expr of partial derivative of f wrt y
critical_points=sp.solve([f_x,f_y],(x,y),dict=True) #solve for x and y with the expr of partial derivatives and equate to 0

if critical_points:
    for point in critical_points:
        x_critical=point[x] #extracting x and y values from each critical point dict
        y_critical=point[y]
        print(f"Critical point:({x_critical},{y_critical})")
        f_xx,f_yy,f_xy,f_yx=second_derivatives(f,x_critical,y_critical) #evaluates the 2n derivatives at their criticl point
        H=construct_hessian_matrix(f_xx,f_yy,f_xy,f_yx)
        lambda1,lambda2=find_eigen_values(H[0][0],H[0][1],H[1][0],H[1][1]) #EIGEN VALUES

        if lambda1 > 0 and lambda2 >0 :
            result= "Minimum"
        elif lambda1 < 0 and lambda2 < 0 :
            result= "Maximum"
        elif lambda1 * lambda2 < 0:
            reult= "Saddle point"
        else:
            result= "Inconclusive" #if eigen values are 0, or unclear
        print(f"Critical point:{result}")
else:
    print("No critical points found")



import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import transpose
def tr_func():
    A=np.array([[1,2,3], [4,5,6]])
    tr_A=transpose(A)
    # result=np.dot(tr_A,A)
    result=tr_A@A
    print(f"The transpose of the matrix is {result}")
def linear_func():
    x = np.linspace(-100, 100, 100)
    y = (x * 2) + 3
    plt.title('Linear equation')
    plt.plot(x, y)
    plt.show()
def quad_func():
    x=np.linspace(-10,10,100)
    y=2*(x**2)+(3*x)+4
    plt.plot(x,y)
    plt.title('Quadratic equation')
    plt.xlabel("x")
    plt.ylabel("y=2x^2+3x+4")
    plt.show()
def gaussian_pdf():
    x = np.linspace(-100, 100, 100)
    y = (1 / (15 * (2 * 3.14) ** 0.5)) * (2.71) ** ((-0.5) * ((x - 0) / 15) ** 2)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gaussian PDF')
    plt.show()
def derivative():
    x = np.linspace(-100, 100, 100)
    y = x ** 2
    y_prime = 2 * x
    plt.plot(x, y_prime)
    plt.plot(x, y)
    plt.title('Derivative')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
import random
def error_func():
    # y=2
    theta=np.array([1,-0.2,-0.35,1.1])
    random.seed(42)
    #When 1 sample was calculated
    # x=np.array([1,random.randint(1,10),random.randint(1,10),random.randint(1,10)])
    #for many samples
    samples=[np.array([1,random.randint(1,10),random.randint(1,10),random.randint(1,10)]) for _ in range(4)]
    y_val = [random.randint(1, 10) for _ in range(5)]
    for i,x in enumerate(samples):
        y = y_val[i]
        h_theta_x=np.dot(theta,x)
        E=0.5*(h_theta_x-y)**2
        print(f"Sample {i+1}:")
        print(f"Predicted value (h_theta_x): {h_theta_x}")
        print(f"Actual target value (y): {y}")
        print(f"Error: {E}")
        print(" ")

def main():
    tr_func()
    linear_func()
    quad_func()
    gaussian_pdf()
    derivative()
    error_func()
if __name__=="__main__":
    main()
# from sigmoid_function import sigmoid
import random
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
random.seed(42)
z=np.random.randint(-100,100,10)
sigmoid_derivative = z*(1-sigmoid(z))
print(f"Derivative of sigmoid function is {sigmoid_derivative}")
z_sorted=np.sort(z)
sig_der_sorted=np.sort(sigmoid_derivative)
plt.scatter(z,sigmoid_derivative,color='blue')
plt.plot(z_sorted,sig_der_sorted,color='cyan')
plt.xlabel('z')
plt.ylabel('Sigmoid derivative')
plt.title('Derivative of sigmoid function')
plt.show()



import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

np.random.seed(42)
z=np.random.randint(-100,100,size=10)

plt.scatter(z,sigmoid(z),color='red')
z_sorted = np.sort(z)
sigmoid_sorted = np.sort(sigmoid(z))
plt.plot(z_sorted,sigmoid_sorted,color='blue')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.title('sigmoid function')
plt.show()

print(z)
print(f"Sigmoid function: {sigmoid(z)}")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1.Loading the data
def load_data():
    data=pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    # print(data.head())
    #Select the features (X) and the atrget variable (y)
    X=data.iloc[:,:-2] #except last 2 cols
    y=data.iloc[:,-2]
    return X,y
#2.adding bias term
def add_bias_term(X):
    #adding a column of ones of X for the term X_0 in X vector
    ones = np.ones((X.shape[0], 1))
    print(ones.shape)
    print(X.shape)
    X_bias = np.hstack((ones, X)) # X_0 column will have 1
    return X_bias
#3.Hypothesis function (prediction)
def hypothesis(X,theta):
    #dot pdt of the features(X) and the param (theta)
    # h_theta(X)=X.theta (theta-vector of model parameters including the bias)
    return np.dot(X,theta)
#3.Cost function(MSE)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X,theta)
    cost = (1/(2 * m)) * np.sum((predictions-y)**2)
    return cost
def compute_gradient(X,y,theta):
    m=len(y)
    predictions = hypothesis(X,theta) #X.T for the multiplication
    gradient= (1/m)* np.dot(X.T,(predictions - y))
    return gradient
def gradient_descent(X, y, theta, learning_rate, no_of_iter, delta=1e-6):
    costs = []
    m = len(y)

    for i in range(no_of_iter):
        gradient = compute_gradient(X, y, theta)
        theta = theta - learning_rate * gradient  # Update theta
        cost = compute_cost(X, y, theta)
        costs.append(cost)

        # Print every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

        # **Early Stopping Condition**
        if i > 0 and abs(costs[-1] - costs[-2]) < delta:
            print(f"Converged at iteration {i}")
            print(f"Initial cost: {costs[0]}")
            print(f"Converged cost: {costs[-1]}")

            # Compute R² score (no dimension changes)
            y_train_mean = np.mean(y, axis=0)
            sst = np.sum((y - y_train_mean) ** 2)
            ssr = np.sum((hypothesis(X, theta) - y) ** 2)
            r2_sq = 1 - (ssr / sst)
            print("R² score:", r2_sq)
            break  # Stop further iterations

    return theta, costs

# def gradient_descent(X, y, theta, learning_rate, no_of_iter,delta=1e-6):
#     costs =[]
#     m = len(y)
#     for i in range(no_of_iter):
#         gradient= compute_gradient(X,y,theta)
#         # #we should update the parameters
#         # theta = theta - (learning_rate / m) * np.dot(X.T, (hypothesis(X, theta) - y))
#
#         theta=theta-learning_rate * gradient
#         #compute the cost and store
#         cost=compute_cost(X, y, theta)
#         costs.append(cost)
#         #we can print cost history
#         if i % 100 == 0:
#             print(f"Iteration{i}:Cost {costs[i]}")
#     return theta, costs

# def main():
#     X, y = load_data()
#     X=add_bias_term(X) #adding X_0 column to the feature
#     #Initializing theta value as zero array
#     theta=np.zeros(X.shape[1])
#     #setting learning rate and number of iterations
#     learning_rate=0.0001
#     no_of_iter=100
#     theta,costs=gradient_descent(X, y, theta, learning_rate, no_of_iter)
#     print(f"Optimal theta: {theta}")
#
#     #PLotting the cost function to check for the convergence
#     plt.plot(range(no_of_iter), costs)
#     plt.xlabel("No of iterations")
#     plt.ylabel("Cost")
#     plt.title("Convergence of Gradient Descent")
#     plt.show()
# main()

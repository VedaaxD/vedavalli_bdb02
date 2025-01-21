from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from functions_gd import load_data
from functions_gd import add_bias_term
from functions_gd import hypothesis
from functions_gd import compute_cost
from functions_gd import compute_gradient
from functions_gd import gradient_descent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def split_data(X,y,test_size=0.30):
    np.random.seed(7)
    num_samples=X.shape[0]
    # print(num_samples)
    indices= np.random.permutation(num_samples)
    #Determine the index at which to split the data
    split_index=int(num_samples*(1-test_size))
    # print(split_index)
    X_train,X_test=X[indices[:split_index]],X[indices[split_index:]]
    # print(X_train.shape,X_test.shape)
    Y_train,y_test=y[indices[:split_index]],y[indices[split_index:]]
    # y_train=y[:num_samples]
    return X_train,X_test,Y_train,y_test
def scale_data(X):
    X_scaled=X
    #applying z-score normalization to each column
    for col in X.columns:
        mean=X[col].mean() #mean of the col
        std=X[col].std() #std dev of the col
        #Standardize the column
        if std!=0:
            X_scaled[col]=(X[col]-mean) /std
        else:
            X_scaled[col]=X[col]
    return X_scaled
def main():
    X,y=load_data()
    #preprocessing the data (EDA)
    #scaling the features
    X_scaled=scale_data(X)
    # splitting the dataset
    X_train, X_test, y_train, y_test = split_data(X_scaled.values, y.values, test_size=0.30)
    #adding bias term (X_0=1)
    X_train=add_bias_term(X_train)
    X_test=add_bias_term(X_test)
    #initializing theta (parameters) as zero
    theta = np.zeros(X_train.shape[1])
    # print(theta)
    #setting the learning rate and no of iterations
    learning_rate=0.1
    no_of_iter=1000
    #performing gradient descent to optimize theta
    opt_theta,costs=gradient_descent(X_train,y_train,theta, learning_rate, no_of_iter, delta=1e-6)
    print(f"Optimal theta: {opt_theta}")
    #r2 score
    y_pred=hypothesis(X_test,opt_theta)
    r2=r2_score(y_test,y_pred)
    #mse score
    mse=np.mean((y_test-y_pred)**2)
    print(f"{r2} r2 close to 1 is good")
    print(f"MSE is {mse}")
    # Plot the cost function to check for convergence
    plt.plot(range(len(costs)), costs)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Convergence of Gradient Descent")
    plt.show()
    #plotting test vs ground truth value
    plt.scatter(y_pred,y_test, color='green', marker='*', label='Predicted vs Actual values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
             label='Perfect Prediction Line')
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('y_test vs y_pred')
    plt.legend()
    plt.show()

    return opt_theta
main()



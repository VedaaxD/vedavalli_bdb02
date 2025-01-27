import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from Lab3_ex2 import load_data
from sklearn.metrics import r2_score
def load_data():
    data = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    print(data.head())
    # Select the features (X) and the trget variable (y)
    X = data.iloc[:, :-2]  # except last 2 cols
    y = data.iloc[:, -1]
    return X, y
def add_bias_term(X):
    #adding a column of ones of X for the term X_0 in X vector
    ones = np.ones((X.shape[0], 1))
    print(ones.shape)
    print(X.shape)
    X_bias = np.hstack((ones, X)) # X_0 column will have 1
    return X_bias
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
def hypothesis(X,theta):
    return np.dot(X,theta)
def cost_function(theta,X,y):
    hyp=hypothesis(X,theta)
    residual=hyp-y
    cost=np.dot(residual.T,residual)
    return cost
def partial_derivative(residual,X):
    deriv=2 * (np.dot(X.T,residual))
    return deriv
def normal_equation(X,y):
    X_T_X=np.dot(X.T,X)
    inv=np.linalg.inv(X_T_X)
    theta=np.dot(inv,np.dot(X.T,y))
    return theta
def predict(X,theta):
    prediction=np.dot(X,theta)
    return prediction
def main():
    #loading the data
    X,y=load_data()
    # splitting
    X_train, X_test, y_train, y_test = split_data(X.values, y.values, test_size=0.30)
    # scaling
    X_train = scale_data(pd.DataFrame(X_train))
    X_test = scale_data(pd.DataFrame(X_test))
    # adding the bias term
    X_train = add_bias_term(X_train)
    X_test = add_bias_term(X_test)
    # Initializing theta as 0
    theta = np.zeros(X_train.shape[1])
    # setting the alpha(learning rate value) and the num of iterations
    alpha = 0.0001
    num_of_iter = 1000
    opt_theta=normal_equation(X_train,y_train)
    print(f"Optimal theta: {opt_theta}")
    # prediction on the test set
    y_pred = hypothesis(X_test, opt_theta)
    # calculate r**2 and MSE
    MSE = np.mean((y_test - y_pred) ** 2)
    print(f"MSE:{MSE}")
    r2_value = r2_score(y_test, y_pred)
    print(f"r2 value using gradient descent:{r2_value}. r2 value closer to 1 is good.")
    #plotting
    plt.scatter(y_pred, y_test, color='green', marker='*', label='Predicted vs Actual values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
             label='Perfect Prediction Line')
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('y_test vs y_pred')
    plt.legend()
    plt.show()

main()
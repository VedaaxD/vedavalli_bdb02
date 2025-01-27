import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#1.Loading the data
def load_data():
    data=pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    #Select the features (X) and the atrget variable (y)
    X=data.iloc[:,:-2] #except last 2 cols
    y=data.iloc[:,-2]
    return X,y

#2.adding bias term
def add_bias_term(X):
    #adding a column of ones of X for the term X_0 in X vector
    ones = np.ones((X.shape[0], 1))
    # print(ones.shape)
    # print(X.shape)
    X_bias = np.hstack((ones, X.values)) # X_0 column will have 1
    return X_bias

#3.Hypothesis function (prediction)
def hypothesis(X,theta):
    #dot pdt of the features(X) and the param (theta)
    # h_theta(X)=X.theta (theta-vector of model parameters including the bias)
    return np.dot(X,theta)

#3.Cost function(MSE)
def compute_cost(X,y,theta):
    m = len(y)
    predictions = hypothesis(X,theta).ravel()
    # print(f"Shape of predictions: {predictions.shape},yshape:{y.shape}")
    cost = (1/(2 * m)) * np.sum((predictions-y)**2)
    return cost

def split_data(X,y,test_size=0.30):
    np.random.seed(7)
    num_samples=X.shape[0]
    indices= np.random.permutation(num_samples)
    #Determine the index at which to split the data
    split_index=int(num_samples*(1-test_size))
    X_train,X_test=X.iloc[indices[:split_index]],X.iloc[indices[split_index:]]
    Y_train,y_test=y.iloc[indices[:split_index]],y.iloc[indices[split_index:]]
    return X_train,X_test,Y_train,y_test

def compute_gradient(X_i,y_i,theta):
    predictions = hypothesis(X_i,theta) #hypothesis for 1 row (sample)
    gradient= np.dot(X_i.T,(predictions - y_i)) #computing gradient for 1 sample
    return gradient

def stochastic_gd(X,y,theta,alpha,no_of_iter):
    m=len(y)
    costs=[]
    y=y.values.reshape(-1,1)
    for j in range(no_of_iter):
        for i in range(m):
            index=np.random.randint(m) #choosing a random datapoint
            X_i=X[index].reshape(1,-1) #making it as row vector
            # y_i = np.array(y[index]).reshape(-1, 1)
            y_i=y[index].reshape(-1,1)
            #computing the gradient for one point
            gradient=compute_gradient(X_i,y_i,theta)
            #updating the theta values
            theta=theta - alpha * gradient
            #logging the cost value
            cost=compute_cost(X, y, theta)
            costs.append(cost)
        if j % 100 == 0:
            print(f"Iteration {j}\t Cost: {cost:.4f}")
    return theta, costs

def scale_data(X):
    X_scaled=X.copy()
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
    #loading the data
    X,y=load_data()
    #splitting the data
    X_train, X_test, y_train, y_test = split_data(X,y)
    #scaling the data
    X_train=scale_data(X_train)
    X_test=scale_data(X_test)
    #adding the bias term
    X_train = add_bias_term(X_train)
    X_test=add_bias_term(X_test)
    #initialize theta as 0
    theta=np.zeros((X_train.shape[1],1))
    print(f"Initial theta:{theta}")
    #setting the alpha (hyperparameter) and the no of iterations
    alpha=0.01
    no_of_iter=1000
    #performing stochastic gradient descent to optimize theta
    opt_theta,costs=stochastic_gd(X_train, y_train, theta, alpha, no_of_iter)
    print(f"Optimal theta:{opt_theta}")
    #r2 score
    y_pred=hypothesis(X_test,opt_theta).ravel()
    print(f"y test shape:{y_test.shape}")
    print(f"y_pred shape:{y_pred.shape}")
    r2=r2_score(y_test,y_pred)
    #mse score
    mse=np.mean((y_test-y_pred)**2)
    print(f"{r2} r2 close to 1 is good")
    print(f"MSE is {mse}")
    compute_cost(X_test,y_test,opt_theta)
    #plotting
    #plotting test vs ground truth value
    plt.scatter(y_pred,y_test, color='green', marker='x', label='Predicted vs Actual values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
             label='Perfect Prediction Line')
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('y_test vs y_pred')
    plt.legend()
    plt.show()
main()






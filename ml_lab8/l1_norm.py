import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
def hypothesis(X,theta):
    return np.dot(X,theta)
def compute_cost(X,y,theta,lambda_reg):
    m=len(y)
    prediction = hypothesis(X,theta)
    cost=(1/(2*m))*np.sum((prediction-y)**2) + (lambda_reg * np.sum(np.abs(theta)))
    return cost
def lasso_regression(X,y,alpha=0.01,lambda_reg=0.1,num_iters=1000):
    m=len(y)
    m,n=X.shape
    theta=np.random.randn(n,1)
    costs=[]
    for i in range(num_iters):
        predictions=hypothesis(X,theta)
        gradient=1/m * np.dot(X.T,(predictions-y)) + (lambda_reg * np.sign(theta))
        theta = theta - alpha*gradient
        cost = compute_cost(X, y, theta, lambda_reg)
        costs.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")
    return theta, costs
def add_bias_term(X):
    ones=np.ones((X.shape[0],1))
    X_bias=np.hstack((ones,X))
    # print(X.shape)
    # print(ones.shape)
    return X_bias
def scale_data(X):
    X_scaled=X.copy()
    for col in X.columns:
        mean=X[col].mean()
        std=X[col].std()
        if std!=0:
            X_scaled[col]=(X_scaled[col]-mean)/std
        else:
            X_scaled[col]= X[col]-mean
    return X_scaled
def load_data():
    [X,y]=fetch_california_housing(return_X_y=True)
    return X,y.reshape(-1,1)
def main():
    [X,y]=load_data()
    #splitting the dataset training:70%; test:30%
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=999)
    ###scale the data
    X_train=scale_data(pd.DataFrame(X_train))
    X_test=scale_data(pd.DataFrame(X_test))
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    #adding bias term
    X_train=add_bias_term(X_train)
    X_test=add_bias_term(X_test)
    theta, costs = lasso_regression(X_train, y_train, alpha=0.01, lambda_reg=0.01, num_iters=1000)
    print(f"Optimal theta: {theta}")
    y_pred = hypothesis(X_test, theta)
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score (Gradient Descent Ridge Regression): {r2:.4f}")
if __name__ == '__main__':
    main()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

#loading the data
def load_data():
    data = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    # Select the features (X) and the trget variable (y)
    X = data[["age"]]
    y = data.iloc[:, -1]
    return X, y
def normal_equation(X,y):
    X_T_X=np.dot(X.T,X)
    inv=np.linalg.pinv(X_T_X)
    theta=np.dot(inv,np.dot(X.T,y))
    return theta
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
def add_bias_term(X):
    #adding a column of ones of X for the term X_0 in X vector
    ones = np.ones((X.shape[0], 1))
    # print(ones.shape)
    # print(X.shape)
    X_bias = np.hstack((ones, X)) # X_0 column will have 1
    return X_bias
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
def hypothesis(X,theta):
    #dot pdt of the features(X) and the param (theta)
    # h_theta(X)=X.theta (theta-vector of model parameters including the bias)
    return np.dot(X,theta)
def scikit_learn(X,X_train, y_train, X_test, y_test, theta):
    # #scale the data
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
 #training the linear regression
    model = LinearRegression()
    #training the model
    model.fit(X_train, y_train)
    #prediction on the test set
    y_pred_skl=model.predict(X_test)
    opt_theta_skl = np.hstack((model.intercept_, model.coef_))
    # opt_theta_skl = model.coef_
    # #optimal theta value
    return opt_theta_skl,y_pred_skl
    # #compute the r2 score
    # r2=r2_score(y_test, y_pred)
    # print(f"The R-squared score is {r2:.3f}.r2 value closer to 1 is Good.")
    # print('Training successful!')
def split_data(X, y, test_size=0.30):
    np.random.seed(7)
    num_samples = X.shape[0]
    # print(num_samples)
    indices = np.random.permutation(num_samples)
    # Determine the index at which to split the data
    split_index = int(num_samples * (1 - test_size))
    # print(split_index)
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    # print(X_train.shape,X_test.shape)
    Y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]
    # y_train=y[:num_samples]
    return X_train, X_test, Y_train, y_test
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
    X,y=load_data()
    #splitting the data
    X_train, X_test, y_train, y_test = split_data(X.values, y.values,test_size=0.30)
    #scaling the data
    X_train = scale_data(pd.DataFrame(X_train))
    X_test = scale_data(pd.DataFrame(X_test))
    # adding the bias term
    X_train = add_bias_term(X_train)
    X_test = add_bias_term(X_test)
    ## Initializing theta as 0
    theta = np.zeros(X_train.shape[1])
    #obtaining opt theta values
    theta_gd,_=gradient_descent(X_train, y_train, theta, learning_rate=0.001, no_of_iter=5000)
    theta_ne=normal_equation(X_train,y_train)
    theta_skl,y_pred_skl=scikit_learn(X,X_train,y_train,X_test,y_test, theta_gd)
    # Predictions
    y_pred_ne = hypothesis(X_test, theta_ne)
    y_pred_gd = hypothesis(X_test, theta_gd)
    print("-------------------------------------------------------------------------------")
    print(f"Theta from sklearn is: {theta_skl}")
    print(f"Theta from gradient descent algo is: {theta_gd}")
    print(f"Theta from normal equation is: {theta_ne}")
    print("-------------------------------------------------------------------------------")
    print(f"Predicted values from normal eqn: {y_pred_ne}")
    print(f"Predicted values from gradient descent algo: {y_pred_gd}")
    print(f"Predicted values from sklearn is: {y_pred_skl}")
    print("--------------------------------------------------------------------------------")
    #r2 scores
    r2_ne=r2_score(y_test,y_pred_ne)
    r2_skl=r2_score(y_test,y_pred_skl)
    r2_gd=r2_score(y_test,y_pred_gd)
    print(f"r2 score of sklearn is: {r2_skl:.3f}")
    print(f"r2 score of gradient descent algo is: {r2_gd:.3f}")
    print(f"r2 score of normal equation is: {r2_ne:.3f}")

    #plotting
    X_test_1d=X_test[:,1]
    plt.figure(figsize=(10,6))
    plt.scatter(X_test_1d,y_test,color='blue',label='Actual Data')
    plt.plot(X_test_1d,y_pred_gd,color='red',linestyle='solid',label='Gradient descent line',linewidth=2)
    plt.plot(X_test_1d,y_pred_skl,color='orange',linestyle='dashed',label='Sklearn line',linewidth=5)
    plt.plot(X_test_1d,y_pred_ne,color='cyan',linestyle='solid',label='Normal Equation line',linewidth=2)
    plt.xlabel('Age')
    plt.ylabel('Comparison of regression line')
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()


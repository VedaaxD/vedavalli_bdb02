import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import california_housing_sk

def load_data():
    data=pd.read_csv('/home/ibab/Downloads/housing.csv')
    # Drop columns with NaN values
    data = data.dropna(axis=1)
    # print(data.head())
    X=data.iloc[:,:-2]
    y=data.iloc[:,-2]
    return X,y
def split_data(X,y,test_size=0.30):
    np.random.seed(42)
    num_of_samples = len(X)
    indices = np.random.permutation(num_of_samples)
    split_index=int(num_of_samples*(1-test_size))
    X_train,X_test=X[indices[:split_index]],X[indices[split_index:]]
    y_train,y_test=y[indices[:split_index]],y[indices[split_index:]]
    return X_train,X_test,y_train,y_test
def add_bias_term(X):
    ones=np.ones((X.shape[0],1))
    X_bias=np.hstack((ones,X))
    print(X.shape)
    print(ones.shape)
    return X_bias

def hypothesis_function(X,theta):
    hypothesis=np.dot(X,theta)
    return hypothesis
def cost_function(theta,X,y):
    m=len(y)
    hyp=hypothesis_function(X,theta)
    cost=1/(2*m)*np.sum((hyp-y)**2)
    # print(f"number of iterations: {num_of_iter}:cost_function={cost}")
    return cost
def compute_gradient(X,y,theta):
    m=len(y)
    hyp=hypothesis_function(X,theta)
    error=hyp-y
    gradient=1/m*(np.dot(X.T,error))
    return gradient

def gradient_descent(X, y, theta, learning_rate, no_of_iter, delta=1e-6):
    costs = []
    m = len(y)

    for i in range(no_of_iter):
        gradient = compute_gradient(X, y, theta)
        theta = theta - learning_rate * gradient  # Update theta
        cost = cost_function(X, y, theta)
        costs.append(cost)

        # Print every 100 iterations
        # if i % 100 == 0:
        #     print(f"Iteration {i}: Cost {cost}")

        # **Early Stopping Condition**
        if i > 0 and abs(costs[-1] - costs[-2]) < delta:
            print(f"Converged at iteration {i}")
            print(f"Initial cost: {costs[0]}")
            print(f"Converged cost: {costs[-1]}")

            # Compute R² score (no dimension changes)
            y_train_mean = np.mean(y, axis=0)
            sst = np.sum((y - y_train_mean) ** 2)
            ssr = np.sum((hypothesis_function(X, theta) - y) ** 2)
            r2_sq = 1 - (ssr / sst)
            print("R² score:", r2_sq)
            break  # Stop further iterations
# def gradient_descent(X,y,theta,alpha,num_of_iter):
#     costs=[]
#     for i in range(num_of_iter):
#         gradient=compute_gradient(X,y,theta)
#         #update theta in every iteration to look for the optimal theta values
#         theta=theta-(alpha*gradient)
#         cost=cost_function(theta,X,y)
#         costs.append(cost)
#         if i % 100 == 0:
#             print(f"No of iteration:{i},cost value:{cost}")

    return theta,costs
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
def main():
    X,y=load_data()
    #splitting
    X_train,X_test,y_train,y_test=split_data(X.values,y.values,test_size=0.30)
    #scaling
    X_train=scale_data(pd.DataFrame(X_train))
    X_test=scale_data(pd.DataFrame(X_test))
    #adding the bias term
    X_train=add_bias_term(X_train)
    X_test=add_bias_term(X_test)
    #Initializing theta as 0
    theta=np.zeros(X_train.shape[1])
    #setting the alpha(learning rate value) and the num of iterations
    alpha=0.01
    num_of_iter=1000
    #performing gradient descent to optimize the theta values
    theta,costs=gradient_descent(X_train,y_train,theta,alpha,num_of_iter)
    #prediction on the test set
    y_pred=hypothesis_function(X_test,theta)
    #calculate r**2 and MSE
    MSE=np.mean((y_test-y_pred)**2)
    print(f"MSE:{MSE}")
    r2_value=r2_score(y_test,y_pred)
    print(f"r2 value using gradient descent:{r2_value}. r2 value closer to 1 is good.")
    #plotting the cost function to look for the converging pattern
    plt.plot(range(num_of_iter),costs,color='magenta')
    plt.title("Cost function")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()
    print(f"optimal theta values:{theta}")

    plt.scatter(y_pred, y_test, color='yellow', marker='*', label='Predicted vs Actual values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='solid',
             label='Perfect Prediction Line')
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('y_test vs y_pred')
    plt.legend()
    plt.show()

main()
print(f"Comparing values of r2 and MSE that uses sklearn:")
california_housing_sk.main()
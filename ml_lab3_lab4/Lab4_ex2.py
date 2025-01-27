# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.metrics import r2_score
# def load_data():
#     data=pd.read_csv('/home/ibab/Downloads/housing.csv')
#     print(data.head())
#     X=data.drop(columns=['median_house_value']) #select all teh rows except the target
#     y=data['median_house_value']
#     return X,y
# def ocean_proximity(X):
#     # Initialize an empty list to store the encoded values
#     encoded_values = []
#     # Loop over the rows of the 'ocean_proximity' column
#     for value in X['ocean_proximity']:
#         # Manually check conditions for each possible value
#         if value == 'NEAR BAY':
#             encoded_values.append(0)
#         elif value == 'NEAR OCEAN':
#             encoded_values.append(1)
#         elif value == 'INLAND':
#             encoded_values.append(2)
#         else:
#             # For any other categories (e.g., '<1H OCEAN')
#             encoded_values.append(3)  # Assign a different value
#
#     # Replace the 'ocean_proximity' column with the new encoded values
#     X['ocean_proximity'] = encoded_values
#     return X
# def split_data(X,y,test_size=0.30):
#     np.random.seed(42)
#     num_of_samples = len(X)
#     indices = np.random.permutation(num_of_samples)
#     split_index=int(num_of_samples*(1-test_size))
#     X_train,X_test=X[indices[:split_index]],X[indices[split_index:]]
#     y_train,y_test=y[indices[:split_index]],y[indices[split_index:]]
#     return X_train,X_test,y_train,y_test
# def add_bias_term(X):
#     ones=np.ones((X.shape[0],1))
#     X_bias=np.hstack((ones,X))
#     print(X.shape)
#     print(ones.shape)
#     return X_bias
# def hypothesis_function(X,theta):
#     hypothesis=np.dot(X,theta)
#     return hypothesis
# def cost_function(theta,X,y):
#     m=len(y)
#     hyp=hypothesis_function(X,theta)
#     cost=1/(2*m)*np.sum((hyp-y)**2)
#     # print(f"number of iterations: {num_of_iter}:cost_function={cost}")
#     return cost
# def compute_gradient(X,y,theta):
#     m=len(y)
#     hyp=hypothesis_function(X,theta)
#     error=hyp-y
#     gradient=1/m*np.dot(X.T,error)
#     return gradient
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
#     return theta,costs
# def scale_data(X):
#     X_scaled=X
#     for col in X.columns:
#         mean=X[col].mean()
#         std=X[col].std()
#         if std!=0:
#             X_scaled[col]=(X_scaled[col]-mean)/std
#         else:
#             X_scaled[col]=0
#     return X_scaled
# def main():
#     X,y=load_data()
#     #splitting
#     X_train,X_test,y_train,y_test=split_data(X.values,y.values,test_size=0.30)
#     #scaling
#     X_train=scale_data(pd.DataFrame(X_train))
#     X_test=scale_data(pd.DataFrame(X_test))
#     #adding the bias term
#     X_train=add_bias_term(X_train)
#     X_test=add_bias_term(X_test)
#     #Initializing theta as 0
#     theta=np.zeros(X_train.shape[1])
#     #setting the alpha(learning rate value) and the num of iterations
#     alpha=0.01
#     num_of_iter=1000
#     #performing gradient descent to optimize the theta values
#     theta,costs=gradient_descent(X_train,y_train,theta,alpha,num_of_iter)
#     #prediction on the test set
#     y_pred=hypothesis_function(X_test,theta)
#     #calculate r**2 and MSE
#     MSE=np.mean((y_test-y_pred)**2)
#     print(f"MSE:{MSE}")
#     r2_value=r2_score(y_test,y_pred)
#     print(f"R2 value:{r2_value}. r2 value closer to 1 is good.")
#     #plotting the cost function to look for the converging pattern
#     plt.plot(range(num_of_iter),costs,color='magenta')
#     plt.title("Cost function")
#     plt.xlabel("Number of iterations")
#     plt.ylabel("Cost")
#     plt.show()
#     print(f"optimal theta values:{theta}")
# main()
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
#
# def load_data():
#     data = pd.read_csv('/home/ibab/Downloads/housing.csv')
#     print(data.head())
#     X = data.drop(columns=['median_house_value'])  # select all the rows except the target
#     y = data['median_house_value']
#     return X, y
#
# def ocean_proximity(X):
#     # Initialize an empty list to store the encoded values
#     encoded_values = []
#     # Loop over the rows of the 'ocean_proximity' column
#     for value in X['ocean_proximity']:
#         # Manually check conditions for each possible value
#         if value == 'NEAR BAY':
#             encoded_values.append(0)
#         elif value == 'NEAR OCEAN':
#             encoded_values.append(1)
#         elif value == 'INLAND':
#             encoded_values.append(2)
#         else:
#             # For any other categories (e.g., '<1H OCEAN')
#             encoded_values.append(3)  # Assign a different value
#
#     # Replace the 'ocean_proximity' column with the new encoded values
#     X['ocean_proximity'] = encoded_values
#     return X
#
# def scale_data(X):
#     X_scaled = X.copy()
#     for col in X.columns:
#         if X[col].dtype != 'object':  # Only scale numeric columns
#             mean = X[col].mean()
#             std = X[col].std()
#             if std != 0:
#                 X_scaled[col] = (X[col] - mean) / std
#             else:
#                 X_scaled[col] = 0
#     return X_scaled
#
# def add_bias_term(X):
#     ones = np.ones((X.shape[0], 1))
#     X_bias = np.hstack((ones, X))
#     return X_bias
#
# def hypothesis_function(X, theta):
#     return np.dot(X, theta)
#
# def cost_function(theta, X, y):
#     m = len(y)
#     hyp = hypothesis_function(X, theta)
#     cost = (1 / (2 * m)) * np.sum((hyp - y) ** 2)
#     return cost
#
# def compute_gradient(X, y, theta):
#     m = len(y)
#     hyp = hypothesis_function(X, theta)
#     error = hyp - y
#     gradient = (1 / m) * np.dot(X.T, error)
#     return gradient
#
# def gradient_descent(X, y, theta, alpha, num_of_iter):
#     costs = []
#     for i in range(num_of_iter):
#         gradient = compute_gradient(X, y, theta)
#         # Update theta in every iteration
#         theta = theta - (alpha * gradient)
#         cost = cost_function(theta, X, y)
#         costs.append(cost)
#         if i % 100 == 0:
#             print(f"No of iteration: {i}, cost value: {cost}")
#     return theta, costs
#
# def main():
#     # Load data
#     X, y = load_data()
#
#     # Encoding the 'ocean_proximity' column
#     X = ocean_proximity(X)
#
#     # Splitting the data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#
#     # Scaling the features
#     X_train = scale_data(X_train)
#     X_test = scale_data(X_test)
#
#     # Adding bias term
#     X_train = add_bias_term(X_train.values)  # Convert to numpy array before adding bias term
#     X_test = add_bias_term(X_test.values)  # Convert to numpy array before adding bias term
#
#     # Initializing theta as zeros
#     theta = np.zeros(X_train.shape[1])
#
#     # Hyperparameters
#     alpha = 0.01
#     num_of_iter = 1000
#
#     # Perform gradient descent
#     theta, costs = gradient_descent(X_train, y_train, theta, alpha, num_of_iter)
#
#     # Prediction on the test set
#     y_pred = hypothesis_function(X_test, theta)
#
#     # Calculate MSE and R² score
#     MSE = np.mean((y_test - y_pred) ** 2)
#     print(f"MSE: {MSE}")
#     # r2_value = r2_score(y_test, y_pred)
#     # print(f"R2 value: {r2_value}. r2 value closer to 1 is good.")
#
#     # Plotting the cost function to observe convergence
#     plt.plot(range(num_of_iter), costs, color='magenta')
#     plt.title("Cost function")
#     plt.xlabel("Number of iterations")
#     plt.ylabel("Cost")
#     plt.show()
#
#     print(f"Optimal theta values: {theta}")
#
# if __name__ == "__main__":
#     main()
# import pandas as pd
#
# # Load your dataset
# data = pd.read_csv('/home/ibab/Downloads/housing.csv')
#
# # Select the relevant columns
# features = data[['median_income','total_rooms','housing_median_age','total_bedrooms','population','households', 'median_house_value']]
#
# # Calculate the correlation matrix
# correlation_matrix = features.corr()
#
# # Print the correlation matrix
# print(correlation_matrix)
#
# # If you want to specifically look at correlations with the target variable:
# print(f"Correlation between latitude and median_house_value: {correlation_matrix['households']['median_house_value']}")
# print(f"Correlation between longitude and median_house_value: {correlation_matrix['median_income']['median_house_value']}")
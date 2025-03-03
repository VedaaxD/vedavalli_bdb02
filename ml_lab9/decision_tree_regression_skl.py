import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

df=pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
#splitting data to train and test
X=df.iloc[:,:-2]
y=df.iloc[:,-2]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)

#fitting and training the model
regressor=DecisionTreeRegressor(criterion='squared_error',max_depth=3,random_state=42)
regressor.fit(X_train,y_train)

#prediction
y_pred=regressor.predict(X_test)

#performance metrics
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print(f"MSE: {mse}")
print(f"R2 score: {r2}")

#plotting the regression tree
plt.figure(figsize=(15,10))
plot_tree(regressor,feature_names=X.columns,filled=True)
plt.show()

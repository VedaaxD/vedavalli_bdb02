import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

data=pd.read_csv('/home/ibab/Downloads/data.csv')
# print(data.head())

#dropping the entry id which is not relevant to the cancer prediction
data.drop(columns=['id'],inplace=True)

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0}) #marking 'y' column as 0s and 1s
# Handle missing values (impute with the mean of each column)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(data.drop(columns=['diagnosis']))  # Impute missing values in X
y = data['diagnosis']


#splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=369)

#scaling the data
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#training the model
model=LogisticRegression(max_iter=10000)
model.fit(X_train,y_train)

#prediction on a test set
y_pred=model.predict(X_test_scaled)

#Evaluate the model
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy of the model: {accuracy}")


from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
import pandas as pd
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('/home/ibab/Downloads/data.csv')
data.drop(columns=['id'],inplace=True)
X=data.drop(columns=['diagnosis'])
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
y=data['diagnosis']
imputer=SimpleImputer(strategy='median')
X=imputer.fit_transform(X)

#splitting
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#training
ridge_model=RidgeClassifier(alpha=0.01,max_iter=1000)
ridge_model.fit(X_train_scaled,y_train)

#predictions
y_pred=ridge_model.predict(X_test_scaled)

#evaluation
accuracy=accuracy_score(y_test,y_pred)

print(f"accuracy score: {accuracy:.4f}")
print("Theta (Coefficient) values for RidgeClassifier:")
print(ridge_model.coef_)

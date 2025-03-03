#l1 norm for wisconsin dataset
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
import pandas as pd
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('/home/ibab/Downloads/data.csv')
X=data.drop(columns=['id','diagnosis'])
data['diagnosis']=data['diagnosis'].map({'B':0,'M':1})
y=data['diagnosis']
imputer=SimpleImputer(strategy='median')
X=imputer.fit_transform(X)
#splitting
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
#scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
#training
lasso_clf=LogisticRegression(penalty='l1',solver='liblinear',C=0.1)
lasso_clf.fit(X_train_scaled,y_train)
#predictions
y_pred=lasso_clf.predict(X_test_scaled)
#evaluation
accuracy=accuracy_score(y_test,y_pred)

print(f"accuracy score: {accuracy:.4f}")
print("Theta (Coefficient) values for LassoClassifier:")
print(lasso_clf.coef_)
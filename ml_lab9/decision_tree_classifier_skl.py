import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

data=pd.read_csv('/home/ibab/Downloads/data.csv')

#dropping the entry id which is not relevant to the cancer prediction
data.drop(columns=['id'],inplace=True)

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0}) #marking 'y' column as 0s and 1s
# Handle missing values (impute with the mean of each column)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(data.drop(columns=['diagnosis']))  # Impute missing values in X
y = data['diagnosis']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)
#fitting and training the model
classifier=DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=42)
classifier.fit(X_train,y_train)

#prediction
y_pred=classifier.predict(X_test)

#performance metrics
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy}")

#plotting
plt.figure(figsize=(10,15))
plot_tree(classifier,filled=True,rounded=True)
plt.show()
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


columns=["age","menopause","tumor-size","inv-nodes","node-caps","deg-malig","breast","breast-quad","irradiat","Class"]
df=pd.read_csv('breast_cancer.csv',header=None,names=columns)
#feature types
ordinal_feat=["age","tumor-size","inv-nodes","deg-malig"]
nominal_feat=["menopause","node-caps","breast","breast-quad","irradiat"]
target="Class"
# print(df["Class"].unique())
# print(df["Class"].isnull().sum())  # Check for NaN values


#encoding
#encoding the target variable
df[target]=df[target].map({"no-recurrence-events":0,"recurrence-events":1})
print(df["Class"].unique())
print(df["Class"].isnull().sum())  # Check for NaN values
#encoding the ordinal variables
ordinal_features={
    "age":["20-29","30-39","40-49","50-59","60-69","70-79"],
    "tumor-size":["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54"],
    "inv-nodes": ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26"],
    "deg-malig": [1, 2, 3]
}
for feat, categories in ordinal_features.items():
    df[feat]=df[feat].apply(lambda x : categories.index(x) if x in categories else -1)

#one-hot encoding for nominal variables
def ohe(df,column):
    unique_values=df[column].unique()
    for val in unique_values:
        df[f"{column}_{val}"]=(df[column] == val).astype(int)
    return df.drop(columns=[column])
for feature in nominal_feat:
    df=ohe(df,feature)


#splitting the data
X=df.drop(columns=[target])
y=df[target]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)

#training the model
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

#predictions
y_pred=model.predict(X_test)
#evaluation
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy}")
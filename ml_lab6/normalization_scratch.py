
import numpy as np
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import pandas as pd
def load_data():
    data = pd.read_csv('/home/ibab/Downloads/data.csv')
    data = data.drop(columns=['id'])
    data = data.dropna(axis=1, how='all')
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis'].values.ravel()
    return X, y
def normalize_data(X):
    X_min=X.min()
    X_max=X.max()
    X_normalized=(X-X_min)/(X_max-X_min)
    return X_normalized
def main():
    X,y=load_data()
    X=normalize_data(X)
    kf=KFold(n_splits=10,shuffle=True,random_state=369)
    accuracy=[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #initializing the model eg linear vector
        clf=LogisticRegression(max_iter=5000)
        #fitting the model
        clf.fit(X_train,y_train)
        #cross validation score
        scores=clf.score(X_test,y_test)
        accuracy.append(scores)
        print(f"Fold Accuracy : {scores:.3f}")
    accuracies=np.mean(accuracy)
    std_dev=np.std(accuracy)
    print(f"\nFold Accuracy: {accuracies:.3f}")
    print(f"Standard Deviation: {std_dev:.3f}")

main()

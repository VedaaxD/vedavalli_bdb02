import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data():
    data=pd.read_csv('/home/ibab/Downloads/sonar data.csv')
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1].map({'R':1,'M':0})
    return X,y
def main():
    X,y=load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf=KFold(n_splits=10,shuffle=True,random_state=369)
    accuracy=[]
    for train_idx,test_idx in kf.split(X_scaled):
        X_train,X_test=X.iloc[train_idx],X.iloc[test_idx]
        y_train,y_test=y.iloc[train_idx],y.iloc[test_idx]
        #initializing the model
        model=LogisticRegression(max_iter=10000)
        #fitting the model
        model.fit(X_train,y_train)
        #cross validation score
        scores=model.score(X_test,y_test)
        accuracy.append(scores)
        print(f"Fold accuracy: {scores:.3f}")
    mean_accuracy=np.mean(accuracy)
    std_accuracy=np.std(accuracy)
    print(f"\nMean of the accuracies: {mean_accuracy:.3f}")
    print(f"Standard Deviation of the accuracies:{std_accuracy:.3f}")

main()
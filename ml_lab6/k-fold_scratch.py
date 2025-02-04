import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
def load_data():
    data = pd.read_csv('/home/ibab/Downloads/data.csv')
    data = data.drop(columns=['id'])
    data = data.dropna(axis=1, how='all')
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis'].values.ravel()
    return X, y

def shuffle_data(X,y):
    indices=np.arange(len(X))
    indices=np.random.permutation(indices)
    return X.iloc[indices],y[indices]

def create_folds(X,y,k):
    X,y=shuffle_data(X,y)
    folds=np.array_split(np.arange(len(X)),k)
    return folds

def train_test_split(X,y,fold,folds):
    #here fold=some no is the fold that we choose as test set
    test_index=folds[fold]
    train_index=np.hstack([folds[i] for i in range(len(folds)) if i!= fold])
    return X.iloc[train_index],y[train_index],X.iloc[test_index],y[test_index]
# def normalize_data(X):
#     X_min=X.min()
#     X_max=X.max()
#     X_normalized=(X-X_min)/(X_max-X_min)
#     return X_normalized
def main():
    X, y = load_data()
    # X=normalize_data(X)
    k=10
    folds=create_folds(X,y,k)
    model=LogisticRegression(max_iter=10000)
    accuracies=[]

    for i in range(k):
        X_train, y_train, X_test, y_test = train_test_split(X,y,i,folds)
        mean_train=np.mean(X_train)
        print(f"The mean value of training data is: {mean_train}")
        mean_test=np.mean(X_test)
        print(f"The mean value of test data is: {mean_test}")
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        accuracies.append(accuracy)
        print(f"Fold Accuracy : {accuracy:.3f}")
        print(f"_________________________________________________________")
    mean_accuracy=np.mean(accuracies)
    std=np.std(accuracies)
    print(f"\nThe accuracy is {mean_accuracy:.3f}")
    print(f"The std deviation is {std:.3f}")
main()







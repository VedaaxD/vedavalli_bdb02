import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data():
    data = pd.read_csv('/home/ibab/Downloads/sonar data.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].map({'R': 1, 'M': 0})
    return X, y
def normalization(X_train,X_test):
    X_train_min=X_train.min()
    X_train_max=X_train.max()
    X_train_normalized=(X_train-X_train_min)/(X_train_max-X_train_min)
    X_test_normalized=(X_test-X_train_min)/(X_train_max-X_train_min)
    return X_train_normalized, X_test_normalized
def normalization_skl(X_train,X_test):
    min_max_scaler=MinMaxScaler()
    X_train_normalized_skl=min_max_scaler.fit_transform(X_train)
    X_test_normalized_skl=min_max_scaler.transform(X_test)
    return X_train_normalized_skl,X_test_normalized_skl
def main():
    X,y=load_data()
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model=LogisticRegression(max_iter=10000)
    #manually normalizing
    X_train_normalized,X_test_normalized=normalization(X_train,X_test)
    model.fit(X_train_normalized,y_train)
    #predict
    y_pred=model.predict(X_test_normalized)
    accuracy=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    print(f"Accuracy score for manually normalized data(preprocessed) : {accuracy}")
    print(f"Confusion matrix:\n{cm}")
    print(f"------------------------------------------------------------")
    print(f"Sklearn normalization:")
    #sklearn
    X_train_normalized_skl, X_test_normalized_skl=normalization_skl(X_train,X_test)
    model.fit(X_train_normalized_skl, y_train)
    y_pred_skl = model.predict(X_test_normalized_skl)
    accuracy_skl = accuracy_score(y_test, y_pred_skl)
    cm_skl = confusion_matrix(y_test, y_pred_skl)
    print(f"Accuracy score for sklearn normalized data: {accuracy_skl}")
    print(f"Confusion matrix for sklearn normalized data: {cm_skl}")
main()
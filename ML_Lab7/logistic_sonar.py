import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_data():
    data = pd.read_csv('/home/ibab/Downloads/sonar data.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].map({'R': 1, 'M': 0})
    return X, y

def main():
    X,y=load_data()
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model=LogisticRegression(max_iter=10000)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f"Accuracy score w/o preprocessing: {accuracy}")
main()
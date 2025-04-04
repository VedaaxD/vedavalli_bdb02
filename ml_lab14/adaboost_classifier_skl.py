#Implement Adaboost classifier using scikit-learn. Use the Iris dataset.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
def load_data():
    iris = load_iris()
    X,y=iris.data,iris.target
    return X,y
def main():
    X,y=load_data()
    #splitting
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)
    #scaling
    scaler = StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    #fitting a model
    model=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5),n_estimators=50,learning_rate=0.1,random_state=42)
    model.fit(X_train_scaled,y_train)
    y_pred=model.predict(X_test_scaled)
    #evaluation
    acc=accuracy_score(y_test,y_pred)
    print(f"Accuracy of the model using Adaboost classifier: {acc}")
if __name__ == '__main__':
    main()

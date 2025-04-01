#Random Forest classifier implementation using iris dataset
from sklearn.model_selection import train_test_split,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np
def load_data():
    iris=load_iris()
    X_c,y_c=iris.data,iris.target
    return X_c,y_c
def main():
    X_c,y_c=load_data()
    # KFold splitting
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracy_scores = []
    for fold, (train_index, test_index) in enumerate(kf.split(X_c)):
        # Splitting
        X_c_train, X_c_test = X_c[train_index], X_c[test_index]
        y_c_train, y_c_test = y_c[train_index], y_c[test_index]
        #fitting a model
        model=RandomForestClassifier(n_estimators=10,random_state=42)
        model.fit(X_c_train,y_c_train)
        #prediction
        y_pred=model.predict(X_c_test)
        #evaluation
        accuracy=accuracy_score(y_c_test,y_pred)
        accuracy_scores.append(accuracy)
        print(f"Fold {fold+1} Accuracy using Random Forest Classifier: {accuracy:.4f}")
    #final results
    print("\nOverall Performance across all folds:")
    print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
if __name__=="__main__":
    main()
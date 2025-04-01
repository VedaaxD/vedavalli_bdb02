# Bagging classifier implementation using iris dataset
from sklearn.model_selection import train_test_split,KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np
def load_data():
    diabetes=load_iris()
    X_r,y_r=diabetes.data,diabetes.target
    return X_r,y_r
def main():
    X_c, y_c = load_data()
    # KFold splitting
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracy_scores = []
    for fold, (train_index, test_index) in enumerate(kf.split(X_c)):
        print(f"\nFold {fold + 1}/{kf.get_n_splits()}")
        # Splitting K-Fold train and test
        X_c_train, X_c_test = X_c[train_index], X_c[test_index]
        y_c_train, y_c_test = y_c[train_index], y_c[test_index]
        # Scaling
        scaler = StandardScaler()
        X_c_train_scaled = scaler.fit_transform(X_c_train)
        X_c_test_scaled = scaler.transform(X_c_test)  # Use test set, not validation
        # Base model
        model = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)
        # Bagging Classifier
        bag_model = BaggingClassifier(estimator=model, n_estimators=50, random_state=42)
        bag_model.fit(X_c_train_scaled, y_c_train)
        # Predictions on test fold
        y_pred = bag_model.predict(X_c_test_scaled)
        # Evaluation
        accuracy = accuracy_score(y_c_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f"Accuracy for Fold {fold + 1}: {accuracy:.4f}")

    # Final results
    print("\nOverall Performance across all folds:")
    print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
if __name__ == '__main__':
    main()
#
# #HYPERPARAMTER TUNING
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.ensemble import BaggingClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.datasets import load_iris
# import numpy as np
#
# def load_data():
#     iris = load_iris()
#     X_c, y_c = iris.data, iris.target
#     return X_c, y_c
# #separate function for tuning hyperparamters
# def tune_hyperparameters(X_train, y_train, X_val, y_val):
#     best_score = 0
#     best_n_estimators = 10  # Default value
#
#     for n_estimators in [10, 20, 50, 100]:  #  different values
#         model = BaggingClassifier(
#             estimator=DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42),
#             n_estimators=n_estimators,
#             random_state=42
#         )
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_val)
#         val_accuracy = accuracy_score(y_val, y_pred)
#         print(f"Tuned n_estimators={n_estimators}, Validation Accuracy={val_accuracy:.4f}")
#         if val_accuracy > best_score:
#             best_score = val_accuracy
#             best_n_estimators = n_estimators
#     print(f"Best n_estimators selected: {best_n_estimators} (Validation Accuracy={best_score:.4f})\n")
#     return best_n_estimators
# def main():
#     X_c, y_c = load_data()
#     # K-Fold splitting
#     kf = KFold(n_splits=10, shuffle=True, random_state=42)
#     accuracy_scores = []
#
#     for fold, (train_index, test_index) in enumerate(kf.split(X_c)):
#         print(f"\nFold {fold + 1}/{kf.get_n_splits()}")
#
#         # Splitting into train and test folds
#         X_c_train_full, X_c_test = X_c[train_index], X_c[test_index]
#         y_c_train_full, y_c_test = y_c[train_index], y_c[test_index]
#
#         # Further split into train and validation (80% train, 20% validation)
#         X_c_train, X_c_val, y_c_train, y_c_val = train_test_split(
#             X_c_train_full, y_c_train_full, test_size=0.2, random_state=42)
#         # Scaling
#         scaler = StandardScaler()
#         X_c_train_scaled = scaler.fit_transform(X_c_train)
#         X_c_val_scaled = scaler.transform(X_c_val)
#         X_c_test_scaled = scaler.transform(X_c_test)
#         # **Step 1: Tune hyperparameters using validation set**
#         best_n_estimators = tune_hyperparameters(X_c_train_scaled, y_c_train, X_c_val_scaled, y_c_val)
#
#         # **Step 2: Train final model on full training data (train_full) with best hyperparameters**
#         scaler = StandardScaler()
#         X_c_train_full_scaled = scaler.fit_transform(X_c_train_full)
#         X_c_test_scaled = scaler.transform(X_c_test)
#
#         final_model = BaggingClassifier(
#             estimator=DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42),
#             n_estimators=best_n_estimators,
#             random_state=42
#         )
#         final_model.fit(X_c_train_full_scaled, y_c_train_full)
#         # **Step 3: Evaluate on the K-Fold test set**
#         y_pred = final_model.predict(X_c_test_scaled)
#         accuracy = accuracy_score(y_c_test, y_pred)
#         accuracy_scores.append(accuracy)
#         print(f"  Final Model Accuracy on Test Set (Fold {fold + 1}): {accuracy:.4f}")
#     # Final results
#     print("\nOverall Performance across all folds:")
#     print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
# if __name__ == "__main__":
#     main()
# #in this the n-estimator=20 was the best across, if we were to choose the best model, we can use 10 as the no of estimators
# #for the bagging classifier
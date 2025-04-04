#Implementing adaboost classifier from scratch w/o scikit learn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
# for constructing weak learners
# class DecisionStump:
#     def __init__(self):
#         self.polarity=1
#         self.feature_index=None
#         self.threshold=None
#         self.alpha=None
#     def predict(self,X):
#         n_samples=X.shape[0]
#         X_column=X[:,self.feature_index]
#         #all the rows with only the selected feature
#
#         predictions=np.ones(n_samples)
#         if self.polarity==1:
#             predictions[X_column < self.threshold] = -1
#         else:
#             predictions[X_column > self.threshold] = 1
#         return predictions
# class AdaBoostClassifier:
#     def __init__(self,num_clf=5):
#         self.num_clf = num_clf
#         self.clfs=[] #list to store weak classifiers
#     def fit(self,X,y):
#         n_samples,n_features=X.shape
#         #if the labels are not 1,-1 convert them
#         y=np.where(y==0,-1,y)
#         #initializing the weights(w)
#         w=np.full(n_samples,(1/n_samples))
#         for _ in range(self.num_clf):
#             stump=DecisionStump()
#             min_error=float('inf') #initializing it to a higher value
#             #find the best feature and threshold
#             for feature_index in range(n_features):
#                 X_column=X[:,feature_index]
#             thresholds=np.unique(X_column)
#             for threshold in thresholds:
#                 for polarity in [-1,1]:
#                     predictions=np.ones(n_samples)
#                     if polarity==1:
#                         predictions[X_column < threshold]=-1
#                     else:
#                         predictions[X_column > threshold]=1
#                     #Compute the weighted error
#                     error=sum(w[y != predictions])
#
#                     if error < min_error:
#                         min_error=error
#                         stump.polarity=polarity
#                         stump.threshold=threshold
#                         stump.feature_index=feature_index
#             #compute the alpha - classifier weight
#             epsilon=1e-10
#             stump.alpha=np.log((1-min_error)/(min_error+epsilon))
#             # Update sample weights
#             predictions = stump.predict(X)
#             w *= np.exp(-stump.alpha * y * predictions)
#             w /= np.sum(w)  # Normalize
#
#             # Store the weak classifier
#             self.clfs.append(stump)
#
#     def predict(self, X):
#         clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
#         y_pred = np.sign(np.sum(clf_preds, axis=0))
#         return y_pred
# def main():
#     # Load dataset
#     iris = load_iris()
#     X, y = iris.data, iris.target
#     # Using only two classes (binary classification)
#     X = X[y != 2]
#     y = y[y != 2]
#     # Train AdaBoost
#     ada = AdaBoostClassifier(num_clf=10)
#     ada.fit(X, y)
#     predictions = ada.predict(X)
#     # Evaluate
#     accuracy = np.mean(predictions == y)
#     print("Training Accuracy:", accuracy)

from sklearn.datasets import load_iris

# Decision stump (weak learner)
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = 1
        return predictions

# AdaBoost Classifier
class AdaBoostClassifier:
    def __init__(self, num_clf=5):
        self.num_clf = num_clf
        self.clfs = []  # List to store weak classifiers

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y == 0, -1, y)  # Convert labels to {-1,1} if not already converted

        w = np.full(n_samples, (1 / n_samples))  # Initialize weights

        for _ in range(self.num_clf):
            stump = DecisionStump()
            min_error = float('inf')

            # Find the best feature and threshold
            for feature_index in range(n_features):
                X_column = X[:, feature_index]
                thresholds = np.unique(X_column)  # Move this inside the loop!
                for threshold in thresholds:
                    for polarity in [-1, 1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[X_column < threshold] = -1
                        else:
                            predictions[X_column > threshold] = 1
                        # Compute weighted error
                        error = np.sum(w[y != predictions])
                        if error < min_error:
                            min_error = error
                            stump.polarity = polarity
                            stump.threshold = threshold
                            stump.feature_index = feature_index  # Fixing feature selection
            # Compute the alpha (classifier weight)
            epsilon = 1e-10  # Avoid division by zero
            stump.alpha = 0.5 * np.log((1 - min_error) / (min_error + epsilon))
            # Update sample weights
            predictions = stump.predict(X)
            w *= np.exp(-stump.alpha * y * predictions)
            w /= np.sum(w)  # Normalize
            # Store the weak classifier
            self.clfs.append(stump)
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sign(np.sum(clf_preds, axis=0))
        return y_pred

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    # Use only two classes (binary classification)
    X = X[y != 2]
    y = y[y != 2]
    # Train AdaBoost
    ada = AdaBoostClassifier(num_clf=10)
    ada.fit(X, y)
    predictions = ada.predict(X)
    # Evaluate
    accuracy = np.mean(predictions == y)
    print("Training Accuracy:", accuracy)

if __name__ == "__main__":
    main()
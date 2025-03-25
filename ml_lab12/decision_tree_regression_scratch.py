import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,KFold
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error
#Loading the dataset
diabetes=load_diabetes()
X=diabetes.data
y=diabetes.target

#Splitting the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=72)

#function for MSE
def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

#function for the best split
def best_split(X,y):
    #initializing the split values
    best_feature,best_threshold,best_mse=None,None,float('inf')
    n_features=X.shape[1]
    for feature_index in range(n_features):
        thresholds=np.unique(X[:,feature_index])
        for threshold in thresholds: #all the threshold points will be tested
            left_indices=X[:,feature_index]<threshold
            right_indices=~left_indices
            #extraction of labels for the left and right indices
            left_y,right_y=y[left_indices],y[right_indices]
            if len(left_y)>0 and len(right_y)>0:
                weighted_mse = (len(left_y) * mse(left_y, np.mean(left_y)) + len(right_y) * mse(right_y, np.mean(
                    right_y))) / len(y)
                if weighted_mse < best_mse:
                    best_mse=weighted_mse
                    best_feature,best_threshold=feature_index,threshold
    return best_feature,best_threshold
#function to build a decision tree
def build_tree(X,y,max_depth=None,depth=3,min_samples_split=5):
    if max_depth is not None and depth == max_depth:
        return np.mean(y)
    if len(y) < min_samples_split:
        return np.mean(y)
    best_feature,best_threshold=best_split(X,y)

    if best_feature is None:
        return np.mean(y)
    left_X,left_y,right_X,right_y=[],[],[],[]
    for i in range(len(X)):
        if X[i][best_feature]<best_threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])

    #Convert lists back to numpy arrays
    left_X,left_y=np.array(left_X),np.array(left_y)
    right_X,right_y=np.array(right_X),np.array(right_y)

    left_subtree = build_tree(left_X, left_y, max_depth, depth + 1, min_samples_split)
    right_subtree = build_tree(right_X, right_y, max_depth, depth + 1, min_samples_split)

    return {
        'feature_index': best_feature,
        'threshold': best_threshold,
        'left': left_subtree,
        'right': right_subtree
    }
# Function to make predictions
def predict(tree, x):
    if not isinstance(tree, dict):
        return tree
    if x[tree['feature_index']] < tree['threshold']:
        return predict(tree['left'], x)
    else:
        return predict(tree['right'], x)


# Function to predict all values
def predict_all(tree, X):
    return np.array([predict(tree, x) for x in X])
#K-Fold Cross validation
kf=KFold(n_splits=5,shuffle=True,random_state=72)
mse_scores=[]
r2_scores=[]
for train_index,test_index in kf.split(X):
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]

    # Build tree
    tree = build_tree(X_train, y_train, max_depth=None)

    # Predict
    y_pred = predict_all(tree, X_test)

    # Compute Mean Squared Error
    mse_value = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse_value)
    r2_scores.append(r2)
# Print Mean MSE and R² Score
print(f"Mean MSE (Custom Tree): {np.mean(mse_scores)}")
print(f"Mean R² Score (Custom Tree): {np.mean(r2_scores)}")
#comparing it with sklearn
dtr_skl=DecisionTreeRegressor(criterion='squared_error',splitter='best',max_depth=5,random_state=72)
#fitting the model
dtr_skl.fit(X_train,y_train)
#prediction
y_pred=dtr_skl.predict(X_test)
#r2 score and MSE
MSE_skl=mean_squared_error(y_test,y_pred)
r2_skl=r2_score(y_test,y_pred)
print(f"The MSE for skl decision tree regressor:{MSE_skl}")
print(f"The r2 score for skl decision tree regressor:{r2_skl}")
# Visualizing actual vs predicted
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Regression Tree Predictions vs Actual")
plt.show()




#decision tree classifier from scratch
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
iris=load_iris()
X=iris.data
y=iris.target
#loading the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)
def count_classes(y):
    counts={}
    for label in y:
        if label in counts:
            counts[label]+=1
        else:
            counts[label]=1
    return counts
def entropy(y):
    counts = count_classes(y)
    n = sum(counts.values())
    probabilities=[ count/n for count in counts.values() ]
    entropy= -np.sum([p * np.log2(p) for p in probabilities if p>0])
    return entropy
def weighted_entropy(left_y,right_y):
    n_total=len(left_y)+len(right_y)
    left_total=len(left_y)
    right_total=len(right_y)
    H_left=entropy(left_y)
    H_right=entropy(right_y)
    return (H_left*(left_total)/n_total) + (H_right*(right_total)/n_total)

def ig(parent_y,left_y,right_y):
    H_parent=entropy(parent_y)
    H_weighted=weighted_entropy(left_y,right_y)
    IG=H_parent-H_weighted
    return IG

#best split decision based on max IG
def best_split(X,y):
    #initializing
    best_feature,best_threshold,best_ig=None,None,-1
    for feature_index in range(X.shape[1]): #iterating over all the features(columns)
#finding  unique values of the current feature
        thresholds=np.unique(X[:,feature_index])
        for threshold in thresholds: #all threshold points will be tested
            left_indices=X[:,feature_index]<threshold
            right_indices=~left_indices
            #extraction of labels for the left and right indices
            left_y,right_y=y[left_indices],y[right_indices]
            if len(left_y)>0 and len(right_y)>0:
                current_IG=ig(y,left_y,right_y) #calculating IG for the split
                if current_IG>best_ig:
                    best_ig=current_IG
                    best_feature,best_threshold=feature_index,threshold
        return best_feature,best_threshold

def build_tree(X,y,max_depth=None,depth=0):
    if max_depth is not None and depth==max_depth:
        return Counter(y).most_common(1)[0][0] #from the tuple of the most common, it exracts the label which occurs the most
    if len(set(y))==1: #as set removes the duplicates (gives only the uniq)
        return y[0]
    feature_index,threshold=best_split(X,y)
    if feature_index is None:
       return Counter(y).most_common(1)[0][0]

    #left and right splits
    left_X,right_X =[],[]
    left_y,right_y=[],[]
    for i in range(len(X)):
        if X[i][feature_index]<threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
    left_X,right_X=np.array(left_X),np.array(right_X)
    left_y,right_y=np.array(left_y),np.array(right_y)

    #building the trees recursively
    left_subtree=build_tree(left_X,left_y,max_depth,depth+1)
    right_subtree=build_tree(right_X,right_y,max_depth,depth+1)

    return{
        'feature_index':feature_index,
        'threshold':threshold,
        'left':left_subtree,
        'right':right_subtree,
    }
def predict(tree,inputs):
    if not isinstance(tree,dict):
        return tree
    if inputs[tree['feature_index']] < tree['threshold']:
        return predict(tree['left'],inputs)
    else:
        return predict(tree['right'],inputs)
def predict_all(tree,X):
    return np.array([predict(tree, inputs) for inputs in X])

#splitting the datasets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)

tree=build_tree(X_train,y_train,max_depth=5)
y_pred=predict_all(tree,X_test)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Accuracy (modeled from scratch) : {accuracy}")

#comparing it with Decision Tree Classifier in sklearn
dtc_skl=DecisionTreeClassifier(max_depth=5,random_state=42)
#fitting the model
dtc_skl.fit(X_train,y_train)
#prediction
y_pred_skl=dtc_skl.predict(X_test)
#accuracy score
accuracy=accuracy_score(y_test,y_pred_skl)
print(f"Accuracy score (from sklearn) :{accuracy}")





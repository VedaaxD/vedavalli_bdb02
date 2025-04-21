import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from collections import defaultdict
def load_data():
    iris=load_iris(as_frame=True)
    df=iris.frame
    print(df.head())
    features=['sepal length (cm)','sepal width (cm)']
    y=df['target']
    #adding random noise
    df[features] += np.random.normal(0,0.2,size=df[features].shape)
    #discretizing the features to 5 bins
    discretizer=KBinsDiscretizer(n_bins=5,encode='ordinal',strategy='uniform')
    df[features]=discretizer.fit_transform(df[features])
    X=df[features]
    y=df['target']
    return X,y
# load_data()
def decision_tree_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)
    clf=DecisionTreeClassifier(max_depth=3,random_state=42)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)

    acc=accuracy_score(y_test,y_pred)
    cm=confusion_matrix(y_test,y_pred)
    return acc
#using joint probabilities
def generative_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.3,random_state=42)
    #estimating the joint probabilities
    joint_prob=defaultdict(lambda : defaultdict(int)) #nested defaultdict
    class_counts=defaultdict(int)

    #for every training point, we convert the feature row to tuple like (feature combn)(2,1) (3,2)
    # which is used as key
    for i in range(len(X_train)):
        row=tuple(X_train.iloc[i])
        label=y_train.iloc[i]
        joint_prob[label][row]+=1 #counting the frq of the feature combination
        class_counts[label]+=1 #we count how many samples exist for each class

    #Converting them into probabilities
    for label in joint_prob: #outer key of the nested dict
        for row in joint_prob[label]: #inner key
            joint_prob[label][row] /= class_counts[label]

    #prior probability calculation
    priors={label:class_counts[label]/len(y_train)for label in class_counts}

    #prediction
    preds=[]

    for _,row in X_test.iterrows(): #loops through each row (feature combn) in the test data
        row=tuple(row)
        probs={} #to store P(class)* P(features|class) for each class

        for label in priors: #loops through each class label in priors
            likelihood=joint_prob[label].get(row,1e-6)
            #computing the joint probability
            probs[label]=likelihood * priors[label]
        preds.append(max(probs,key=probs.get))

    acc=accuracy_score(y_test,preds)
    cm=confusion_matrix(y_test,preds)

    print("Generative Model Accuracy: ",acc)
    print("Generative Model Confusion Matrix: ",cm)

    return acc

def main():
    X,y=load_data()
    print("\n Decision Tree Model:")
    acc_tree=decision_tree_model(X,y)
    print("\n Generative Model:")
    acc_gen=generative_model(X,y)

    print("f\n Accuracy comparison:")
    print(f"Decision Tree Accuracy: {acc_tree:.2f}")
    print(f"Generative Model Accuracy: {acc_gen:.2f}")

if __name__ == "__main__":
    main()
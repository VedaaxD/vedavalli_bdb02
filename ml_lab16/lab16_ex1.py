#Write a Python program to aggregate  predictions from multiple trees to output a final prediction for a regression problem.

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

def load_data():
    diabetes=load_diabetes()
    X,y=diabetes.data,diabetes.target
    return X,y
def main():
    X,y=load_data()
    #splitting
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)
    #bagging - aggregating predictions from multiple decision trees
    n_trees=10
    predictions=[]
    for i in range(n_trees):
        #bootstrap sample
        indices=np.random.choice(len(X_train),size=len(X_train),replace=True) #randomly selecting samples with replacement
        X_sample,y_sample=X_train[indices],y_train[indices]
        #train the decision tree
        tree=DecisionTreeRegressor(random_state=42)
        tree.fit(X_sample,y_sample)
        #predict the test set
        y_pred=tree.predict(X_test)
        predictions.append(y_pred)
        print(f"Predictions:{predictions}")
    #avg predictions
    final_pred=np.mean(predictions,axis=0)
    #evaluation
    r2=r2_score(y_test,final_pred)
    mse=mean_squared_error(y_test,final_pred)
    print(f"Bagging r2 score: {r2}")
    print(f"MSE: {mse}")
if __name__=="__main__":
    main()

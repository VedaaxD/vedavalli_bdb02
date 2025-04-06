#For the heart.csv dataset, build a logistic regression classifier to predict the risk of heart disease.  Vary the threshold to generate multiple confusion matrices.  Implement a python code to calculate the following metrics
#Accuracy
#Precision
#Sensitivity
#Specificity
#F1-score
#Plot the ROC curve
#AUC
#THIS IS USING SKLEARN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score,roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    df=pd.read_csv('heart.csv')
    X=df.drop('output',axis=1)
    y=df['output']
    return X,y
def main():
    X,y=load_data()
    #splitting the dataset
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)
    #standardizing the features
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    #Train the logistic regression model
    model=LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    #predicted probabilities
    y_prob=model.predict_proba(X_test)[:,1] #filters only col1 which is the prob of the sample being class1
    thresholds=np.linspace(0,1,10) #creates an array of 10 threshold values lying btw 0 and1
    results=[]
    for threshold in thresholds:
        y_pred=(y_prob>=threshold).astype(int)
        #calculating metrics
        tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()
        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        sensitivity=recall_score(y_test,y_pred)
        specificity=tn/(tn+fp)
        f1=f1_score(y_test,y_pred)
        results.append({
            'Threshold':threshold,
            'True Positive Rate':tp,
            'False Positive Rate':fp,
            'True Negative Rate':tn,
            'False Negative Rate':fn,
            'Accuracy':accuracy,
            'Precision':precision,
            'Sensitivity':sensitivity,
            'Specificity':specificity,
            'F1 score':f1,
        })
    results=pd.DataFrame(results)
    print(f"\nPerformance metrics at different thresholds:")
    #printing the dataframe w/o index
    print(results.to_string(index=False)) #table format
    #plotting the ROC curve
    fpr,tpr,thresholds=roc_curve(y_test,y_prob) #roc
    #auc calc
    auc_score=roc_auc_score(y_test,y_prob)
    print(f"\nArea Under the Curve (AUC): {auc_score:.4f}")

    plt.figure(figsize=(8,8))
    #plot ROC
    plt.plot(fpr,tpr,color='blue',lw=2,label=f'ROC curve (AUC={auc_score:.4F})')
    #plot Random chance classifier
    plt.plot([0,1],[0,1],color='red',lw=2,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
if __name__ == '__main__':
    main()

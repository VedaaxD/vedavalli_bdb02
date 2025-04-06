#Implementing evaluation metrics from scratch
from evaluation_metrics_functions import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_data():
    df=pd.read_csv('heart.csv')
    X=df.drop(columns=['output'],axis=1)
    y=df['output']
    return X,y
def main():
    X,y=load_data()
    #splitting and scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    #training
    model = LogisticRegression()
    model.fit(X_train_scaled,y_train)
    #prediction
    predicted_probs=model.predict_proba(X_test_scaled)[:,1] #getting only the probs of class 1
    thresholds=[0.1,0.5,0.7,0.9]
    confusion_matrices=confusion_matrix(y_test,predicted_probs,thresholds)
    for threshold,(TP,FP,FN,TN) in confusion_matrices.items():
        print(f"Threshold: {threshold}")
        print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        print(f"Accuracy:{calculate_accuracy(TP,TN,FP,FN)}")
        print(f"Precision:{calculate_precision(TP,FP)}")
        print(f"Sensitivity:{calculate_sensitivity(TP,FN)}")
        print(f"Specificity:{calculate_specificity(TN,FP)}")
        print(f"F1-score:{calculate_f1_score(calculate_precision(TP,FP),calculate_sensitivity(TP,FN))}")
        print("-" * 50)

    fpr,tpr=roc_curve(y_test,predicted_probs,thresholds)
    plot_roc_curve(fpr,tpr)
    auc_val=auc(fpr,tpr)
    print(f"AUC: {auc_val}")
if __name__ == "__main__":
    main()
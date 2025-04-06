#function for evaluation metrics
import matplotlib.pyplot as plt

def confusion_matrix(actual_labels,predicted_probs,thresholds):
    results={}
    for threshold in thresholds:
        #init all the positives and negatives to 0
        TP=0
        FP=0
        TN=0
        FN=0
        #we want each label with it's respective predicted labels
        for actual,pred_prob in zip(actual_labels,predicted_probs):
            pred_label=1 if pred_prob >= threshold else 0
            if actual ==1 and pred_label == 1:
                TP+=1
            elif actual ==0 and pred_label==1:
                FP+=1
            elif actual ==0 and pred_label==0:
                TP+=1
            elif actual == 1 and pred_label == 0:
                FN+=1
        results[threshold]=(TP,FP,TN,FN)
    return results
def calculate_accuracy(TP,TN,FP,FN):
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    return accuracy
def calculate_precision(TP,FP):
    precision= TP/(TP+FP) if (TP+FP) > 0 else 0
    return precision
def calculate_sensitivity(TP,FN):
    sensitivity= TP/(TP+FN) if (TP+FN) > 0 else 0
    return sensitivity
def calculate_specificity(TN,FP):
    sensitivity=TN/(TN+FP) if (TN+FP) > 0 else 0
    return sensitivity
def calculate_f1_score(precision,TPR):
    f1_score= 2* (precision*TPR)/(precision+TPR) if (precision+TPR) > 0 else 0
    return f1_score
def roc_curve(actual_labels,predicted_probs,thresholds):
    fpr=[] #1-specificity values
    tpr=[] #sensitivity values
    for threshold in thresholds:
        TP,FP,TN,FN=confusion_matrix(actual_labels,predicted_probs,[threshold])[threshold] #passing a single threshold value as a list format
        TPR=calculate_sensitivity(TP,FN)
        FPR=1-calculate_specificity(TN,FP)
        tpr.append(TPR)
        fpr.append(FPR)
    return fpr,tpr
def auc(fpr,tpr):
    auc=0
    for i in range(1,len(fpr)):
        auc=auc+( (fpr[i]-fpr[i-1]) *(tpr[i]+tpr[i-1]) )/2
    return auc
def plot_roc_curve(fpr,tpr):
    plt.figure(figsize=(10,10))
    plt.plot(fpr,tpr,color='r',linestyle='dashed',label='ROC curve')
    plt.plot([0,1],[0,1],color='b',linestyle='dotted',label='Random guesses')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()


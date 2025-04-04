#Implementing gradient boosting (classification) using sklearn
from ISLP import load_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score

def load():
    weekly=load_data('Weekly')
    weekly['Direction']=weekly['Direction'].map({'Up':1,'Down':0})
    X=weekly.drop(columns=['Direction','Year','Today'])
    y=weekly['Direction']
    return X,y
def main():
    #load data
    X,y=load()
    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    #training the model
    gb_clf=GradientBoostingClassifier(n_estimators=50,learning_rate=0.1,random_state=42)
    gb_clf.fit(X_train,y_train)
    #predictions
    y_pred=gb_clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f"Accuracy:{accuracy:.4f}")
    # Perform 10-Fold Cross-Validation
    cv_scores = cross_val_score(gb_clf, X, y, cv=10, scoring='accuracy')
    print(f"10-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
if __name__=='__main__':
    main()

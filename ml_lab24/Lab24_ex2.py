#BAYESIAN LEARNING
#Implement Naive Bayes classifier for spam detection using scikit-learn library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

def load_data():
    df=pd.read_csv('spam_sms.csv',encoding='latin-1')
    print(df.head())
    #naming the cols
    df=df[['v1','v2']]
    df.columns=['label','message']
    #mapping the labels to binary values
    df['label']=df['label'].map({'ham':0,'spam':1})
    X=df['message']
    y=df['label']
    return X,y
def main():
    X,y=load_data()
    #splitting the dataset
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    #Vectorizing the data
    #we've to transform the text msgs into numerical feature vectors
    #Initializing the CountVectorizer
    vectorizer=CountVectorizer()
    #fit and transform the training data
    X_train_vec=vectorizer.fit_transform(X_train)
    #transform the test data
    X_test_vec=vectorizer.transform(X_test)

    #training the naive bayes clf
    #init the clf
    nb_clf=MultinomialNB()
    #train
    nb_clf.fit(X_train_vec,y_train)

    #prediction
    y_pred=nb_clf.predict(X_test_vec)

    print(f"Confusion Matrix: {confusion_matrix(y_test,y_pred)}")
    print(f"\nClassification Report: {classification_report(y_test,y_pred)}")
    print(f"\n Accuracy score: {accuracy_score(y_test,y_pred)}")
if __name__=='__main__':
    main()


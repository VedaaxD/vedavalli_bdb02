#Bagging regressor implementation using diabetes dataset
from sklearn.model_selection import train_test_split,KFold
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import load_diabetes
import numpy as np
def load_data():
    diabetes=load_diabetes()
    X_r,y_r=diabetes.data,diabetes.target
    print(diabetes.target)
    return X_r,y_r
def main():
    X_r,y_r=load_data()
    #kfold splits
    kf=KFold(n_splits=10,shuffle=True,random_state=42)
    mse_scores=[]
    r2_scores=[]
    for fold, (train_index, test_index) in enumerate(kf.split(X_r)):
        #splitting
        X_r_train_full, X_r_test_full = X_r[train_index], X_r[test_index]
        y_r_train_full, y_r_test_full = y_r[train_index], y_r[test_index]
        #further splitting to train and val set
        X_r_train, X_r_val,y_r_train, y_r_val = train_test_split(X_r_train_full, y_r_train_full,test_size=0.2,random_state=42)
        #scaling
        scaler=StandardScaler()
        X_r_train_scaled=scaler.fit_transform(X_r_train)
        X_r_val_scaled=scaler.transform(X_r_val)
        #base model
        model=DecisionTreeRegressor(max_depth=3)
        #model for bagging
        bag_model=BaggingRegressor(estimator=model,n_estimators=50,random_state=42)
        bag_model.fit(X_r_train_scaled,y_r_train)
        #prediction
        y_r_pred=bag_model.predict(X_r_val_scaled)
        #evaluation
        mse=mean_squared_error(y_r_val,y_r_pred)
        print(f"Bagging Regressor MSE:{mse:.4f}")
        r2=r2_score(y_r_val,y_r_pred)
        print(f"R2 score for a Bagging regressor:{r2:.4f}")
        mse_scores.append(mse)
        r2_scores.append(r2)
    #final results:
    print(f"Overall performance of all folds:\n")
    print(f"Mean MSE: {np.mean(mse_scores):.4f}")
    print(f"Mean R2: {np.mean(r2_scores):.4f}")

if __name__ == '__main__':
    main()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import pandas as pd
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBRegressor

import matplotlib.pyplot as plt
#TEMPLATE CODE
# def eda(X_df,X_df_scaled):

def load_data():
    [X,y]=fetch_california_housing(return_X_y=True)
    return (X,y)
def main():
    [X,y]=load_data()
    #splitting the dataset training:70%; test:30%
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=999)
    ###scale the data
    scaler=StandardScaler()
    scaler=scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    #train a model
    print('---TRAINING---')
    print("N=%d " % (len(X)))
    #training a Linear Regression
    model=LinearRegression()
    #train the model
    model.fit(X_train,y_train)
    #prediction on a test set
    y_pred = model.predict(X_test)
    #compute the r2 score
    r2=r2_score(y_test,y_pred)
    print("r2 score is %0.2f (closer to 1 is good)" % r2)
    #compute MSE
    MSE=mean_squared_error(y_test,y_pred)
    print(f"MSE value is {MSE:.3f} ")
    print('Done!')
    #optimal theta values
    feature_names = fetch_california_housing().feature_names
    # Convert to numpy array to pd DataFrame
    opt_theta = pd.DataFrame(model.coef_, index=feature_names, columns=['Coefficient'])
    print(f"Optimal theta:\n{opt_theta}")
    #plotting
    plt.scatter(y_pred, y_test, color='green', marker='*', label='Predicted vs Actual values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
             label='Perfect Prediction Line')
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('y_test vs y_pred')
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
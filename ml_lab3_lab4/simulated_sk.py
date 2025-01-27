import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
# from Lab4_ex2 import
def load_data():
    data=pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    print(data.head())
    X=data.iloc[:,:-2]
    y=data.iloc[:,-2]
    return X,y
def main():
    [X,y]=load_data()
    #splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=25)
    #scale the data
    scaler=StandardScaler()
    scaler=scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    print('----TRAINING THE MODEL----')
    print(f"Length of the training set,{len(X_train)}")
    print(f"Length of the test set,{len(X_test)}")
    #training the linear regression
    model = LinearRegression()
    #training the model
    model.fit(X_train_scaled, y_train)
    #prediction on the test set
    y_pred=model.predict(X_test_scaled)
    #optimal theta value
    opt_theta=model.coef_
    print(f"Optimal theta values are {opt_theta}")
    #compute the r2 score
    r2=r2_score(y_test, y_pred)
    print(f"The R-squared score is {r2:.3f}.r2 value closer to 1 is Good.")
    print('Training successful!')
    #plotting test vs ground truth value
    plt.scatter(y_test,y_pred,color='green',marker='*',label='Predicted vs Actual values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
             label='Perfect Prediction Line')
    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('y_test vs y_pred')
    plt.legend()
    plt.show()
main()
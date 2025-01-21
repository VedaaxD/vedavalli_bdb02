from seaborn import histplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBRegressor

import matplotlib.pyplot as plt
def load_data():
    data=pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    # X=data[['age','BMI','BP','blood_sugar','Gender']]
    # y=data['disease_score']
    # y=data['disease_score_fluct']
    X= data.iloc[:,:-2]
    # y=data.iloc[:,-1]
    y=data.iloc[:,-2]
    return (X,y)
def EDA(data):
    color = {"whiskers": "black", "medians": "black", "caps": "black"}
    data.plot.box(vert=False, color=color,patch_artist=True)
    plt.axvline(x=0, ymin=-1, ymax=1, color="black", linestyle="--")
    _ = plt.title("Original Data")
    plt.tight_layout()
    plt.show()

    # Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns)

    # Plotting scaled data boxplot
    color = {"whiskers": "black", "medians": "black", "caps": "black"}
    scaled_data_df.plot.box(vert=False, color=color, patch_artist=True)
    plt.axvline(x=0, ymin=-1, ymax=1, color="black", linestyle="--")
    _ = plt.title("Data after Scaling")
    plt.tight_layout()
    plt.show()
def plot(data):
    #visualizing the feature distributions
    data.hist(bins=30,figsize=(12,10),edgecolor='black')
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    plt.title('Feature data')
    plt.show()
def main():
    data = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    [X,y]=load_data()
    #splitting the dataset training:70%; test:30%
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=999)
    print(X_train.shape, y_train.shape)

    # #test-train split visualization
    plt.scatter(X_train.iloc[:,0],y_train,label="Training Data",color='red',alpha=0.7)  #training set
    plt.scatter(X_test.iloc[:,0],y_test,label="Test Data",color='green',alpha=0.7) #test set
    plt.legend()
    plt.title('Training Data vs Test Data')
    plt.show()

    ###scale the data
    #The training data is scaled(standardized) to have zero mean and unit variance using StandardScaler()
    scaler=StandardScaler()
    scaler=scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    #train a model
    print('---TRAINING---')
    print(f"No of samples:{X.shape[0]}")
    #training a Linear Regression
    model=LinearRegression()
    #train the model
    model.fit(X_train_scaled,y_train)
    #theta values
    theta_values=model.coef_
    #intercept value
    intercept_value=model.intercept_
    print("Theta values (coefficients):", theta_values,"the highest theta value impacts the disease score.")
    print("Intercept (theta_0):", intercept_value)

    #prediction on a test set
    y_pred = model.predict(X_test_scaled)
    predicted_df = pd.DataFrame(y_pred, columns=['Predicted Disease Score'])
    print(f"{predicted_df}")
    #compute the r2 score
    r2=r2_score(y_test,y_pred)
    print("r2 score is %0.2f (closer to 1 is good)" % r2)
    print('Successfully predicted!')
    #mse value
    mse=mean_squared_error(y_test,y_pred)
    print("This is the mse value",mse)
    plt.scatter(y_test,y_pred)
    plt.plot(y_test,y_pred,color='red')
    plt.xlabel('Actual disease score')
    plt.ylabel('Predicted disease score')
    plt.title('Actual vs Predicted Disease score')
    plt.legend()
    plot(data=X)

    EDA(data)
if __name__=="__main__":
    main()

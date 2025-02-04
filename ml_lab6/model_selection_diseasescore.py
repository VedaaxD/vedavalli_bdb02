import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# from Lab4_ex2 import
def load_data():
    data=pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
    print(data.head())
    X=data.iloc[:,:-2]
    y=data.iloc[:,-2]
    return X,y
def main():
    [X, y] = load_data()
    # 70% training set and 30% tempoarary set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=369)
    # now splitting that 70% into final training and val set
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=369)
    # standaradizing
    scaler = StandardScaler()
    scaler = scaler.fit(X_train_final)
    X_train_scaled = scaler.transform(X_train_final)
    # additionally here including the scaling for the val set too
    X_val_scaled = scaler.transform(X_val)
    # training the model
    print("\n----------------------POLYNOMIAL REGRESSION----------------------")
    r2_max=0
    d=0
    # polynomial features of degree 2
    degrees = [1,2,3]
    for degree in degrees:
        poly_model = PolynomialFeatures(degree=degree)
        # transforming the X_train and the val set for the polynomial feature
        X_train_poly = poly_model.fit_transform(X_train_scaled)
        X_val_poly = poly_model.transform(X_val_scaled)
        # fitting in the model
        model = LinearRegression()
        model.fit(X_train_poly, y_train_final)
        poly_y_pred = model.predict(X_val_poly)
        # r2 score
        r2_val = r2_score(y_val, poly_y_pred)
        print(f"r2 score for degree {degree} in Polynomial regression is {r2_val}")
        #plotting on the val set
        # MSE
        # MSE = mean_squared_error(y_val, poly_y_pred)
        # print(f"MSE for degree {degree} in Polynomial regression is {MSE}")
        if r2_val > r2_max:
            r2_max=r2_val
            d=degree
            print(f"r2 score for degree {d} is better.")
        # training the model with the combined training and validation data
    print(f"\n---------Training the final model with degree {d} on the combined dataset.-----------------")
    poly_model = PolynomialFeatures(degree=d)
    # transform the combined data
    X_combined_train = np.vstack((X_train_scaled, X_val_scaled))
    y_combined_train = np.hstack((y_train_final, y_val))

    X_combined_poly = poly_model.fit_transform(X_combined_train)
    # re- train
    final_model = LinearRegression()
    final_model.fit(X_combined_poly, y_combined_train)

    # Evaluate the model on the test set
    X_test_scaled = scaler.transform(X_test)  # Scale X_test using the same scaler
    X_test_poly = poly_model.transform(X_test_scaled)

    y_test_pred = final_model.predict(X_test_poly)
    # Compute R2 and MSE for the test set
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"r2 for degree {d} in Polynomial regression is {r2_test}")
    print(f"MSE for degree {d} in Polynomial regression is {mse_test}")


    # plotting
        # print(f"\ny_pred{degree}= {poly_y_pred[:30] }")

        # plt.scatter(y_val,poly_y_pred,color='green',marker="*",label='Val set prediction vs Actual values')
        # plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--',label='Perfect Prediction Line')
        # plt.xlabel('Actual values (y_val)')
        # plt.ylabel('Predicted values (poly_y_pred)')
        # plt.title('val set vs predicted values')
        # plt.legend()
        # plt.show()
main()
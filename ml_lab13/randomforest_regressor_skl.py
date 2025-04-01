#Random Forest Regressor using diabetes dataset
from sklearn.model_selection import train_test_split,KFold
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import load_diabetes
import numpy as np
def load_data():
    diabetes=load_diabetes()
    X_r,y_r=diabetes.data,diabetes.target
    return X_r,y_r

def main():
    X_r, y_r = load_data()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    best_fold = None
    best_r2 = -np.inf  # Initialize with the lowest possible R2
    best_train_index, best_test_index = None, None
    for fold, (train_index, test_index) in enumerate(kf.split(X_r)):
        X_r_train, X_r_test = X_r[train_index], X_r[test_index]
        y_r_train, y_r_test = y_r[train_index], y_r[test_index]
        # Scaling
        scaler = StandardScaler()
        X_r_train_scaled = scaler.fit_transform(X_r_train)
        X_r_test_scaled = scaler.transform(X_r_test)
        # Fit model
        model = RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42)
        model.fit(X_r_train_scaled, y_r_train)
        y_r_pred = model.predict(X_r_test_scaled)
        # Evaluate
        r2 = r2_score(y_r_test, y_r_pred)
        print(f"Fold {fold + 1}: R2 Score = {r2:.4f}")
        # Track best fold
        if r2 > best_r2:
            best_r2 = r2
            best_fold = fold + 1
            best_train_index, best_test_index = train_index, test_index
    print(f"\nBest fold selected: {best_fold} with R2 Score = {best_r2:.4f}")
    # Train final model using best fold's train set
    X_train_final, X_test_final = X_r[best_train_index], X_r[best_test_index]
    y_train_final, y_test_final = y_r[best_train_index], y_r[best_test_index]
    #scaling for the final model
    scaler_final = StandardScaler()
    X_train_final_scaled = scaler_final.fit_transform(X_train_final)
    X_test_final_scaled = scaler_final.transform(X_test_final)
    #final model fitting
    final_model = RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42)
    final_model.fit(X_train_final_scaled, y_train_final)
    y_final_pred = final_model.predict(X_test_final_scaled)
    # Final evaluation
    final_mse = mean_squared_error(y_test_final, y_final_pred)
    final_r2 = r2_score(y_test_final, y_final_pred)

    print(f"\nFinal Model Evaluation on Best Fold's Test Set:")
    print(f"Final MSE: {final_mse:.4f}")
    print(f"Final R2 Score: {final_r2:.4f}")

if __name__ == "__main__":
    main()

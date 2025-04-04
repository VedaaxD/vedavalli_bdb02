#Implementing XGBoost regressor using sklearn
from ISLP import load_data
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold

def load():
    boston = load_data('Boston')
    X = boston.drop(columns=['medv'])
    y = boston['medv']
    return X, y
def k_fold_cross_validation(X, y, depth_values, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_mse_scores = {depth: [] for depth in depth_values}

    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

        # Further split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=42)
        mse_scores = {}

        # Hyperparameter tuning - max depth
        for depth in depth_values:
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=depth, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mse_scores[depth] = mse
            all_mse_scores[depth].append(mse)

        best_depth = min(mse_scores, key=mse_scores.get)
        print(f"Fold {fold}: Best max depth = {best_depth}, MSE = {mse_scores[best_depth]:.4f}\n")
    return all_mse_scores
def main():
    X, y = load()
    depth_values = [2, 3, 4, 5, 6]  # Example depths for tuning
    results = k_fold_cross_validation(X, y, depth_values)
    for depth, scores in results.items():
        print(f"Max Depth {depth}: Mean MSE = {sum(scores) / len(scores):.4f}")
if __name__ == '__main__':
    main()

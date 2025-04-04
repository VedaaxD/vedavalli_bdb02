#IMplementing XGBoost classifier using sklearn
from ISLP import load_data
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def load():
    weekly = load_data('Weekly')
    weekly['Direction'] = weekly['Direction'].map({'Up': 1, 'Down': 0})
    X = weekly.drop(columns=['Direction', 'Year', 'Today'])
    y = weekly['Direction']
    return X, y

def k_fold_cross_validation(X, y, depth_values, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_accuracy_scores = {depth: [] for depth in depth_values}
    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

        # Further split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=42)
        accuracy_scores = {}

        # Hyperparameter tuning (max depth)
        for depth in depth_values:
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=depth, eval_metric='logloss', random_state=42)
            model.fit(X_train, y_train)
            #predictions
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            accuracy_scores[depth] = accuracy
            all_accuracy_scores[depth].append(accuracy)

        best_depth = max(accuracy_scores, key=accuracy_scores.get)
        print(f"Fold {fold}: Best max depth = {best_depth}, Accuracy = {accuracy_scores[best_depth]:.4f}\n")

    return all_accuracy_scores

def main():
    X, y = load()
    depth_values = [2, 3, 4, 5, 6]
    results = k_fold_cross_validation(X, y, depth_values)
    for depth, scores in results.items():
        print(f"Max Depth {depth}: Mean Accuracy = {sum(scores)/len(scores):.4f}")

if __name__ == '__main__':
    main()

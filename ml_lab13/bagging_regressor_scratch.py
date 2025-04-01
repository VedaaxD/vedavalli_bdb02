#Implement bagging regressor without using scikit-learn
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Custom Bagging Regressor
class CustomBaggingRegressor:
    def __init__(self, base_model, n_estimators=10, max_samples=1.0, random_state=None):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.models = []  # To store base models

    def bootstrap_sample(self, X, y):
        #sample w replacement
        n_samples = X.shape[0]
        sample_size = int(self.max_samples * n_samples) if isinstance(self.max_samples, float) else self.max_samples
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_samples, size=sample_size, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.models = []  # Reset models
        for i in range(self.n_estimators):
            # Create a new instance of the base model
            model = self.base_model.__class__(**self.base_model.get_params())
            # model = self.base_model
            # model.fit(X_sample1, y_sample1)  # Train on the first bootstrap sample
            # model.fit(X_sample2, y_sample2)  # Overwrites training on the second bootstrap sample
            # Both self.base_model and model refer to the same object.
            # The model’s training is overwritten with each iteration.You only end up with one trained model.
            # Generate a bootstrap sample
            X_sample, y_sample = self.bootstrap_sample(X, y)
            # Fit the model
            model.fit(X_sample, y_sample)
            # Store the trained model
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators)) #matrix of zeros with shape of the no of samples as rows and col as n (n_estimators=n)
        for i, model in enumerate(self.models): #iterating through multiple models, and calling predict for each one(not recursive)
            predictions[:, i] = model.predict(X)
        return np.mean(predictions, axis=1)
#Each base model handles the predict call separately and independently.
# Initialize Bagging Regressor
bagging_model = CustomBaggingRegressor(
    base_model=DecisionTreeRegressor(max_depth=5, random_state=42),
    n_estimators=10,
    max_samples=0.8,
    random_state=42
)

# Train the Bagging Regressor
bagging_model.fit(X_train, y_train)

# Predict on test data
y_pred = bagging_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (Bagging): {mse}")
print(f"R² Score (Bagging): {r2}")

# Visualizing Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Bagging Regressor Predictions vs Actual")
plt.show()

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

class RegressionModel:
    """Base class for regression models."""

    def __init__(self, model_name="linear", random_state=42):
        """
        Initialize the model.
        model_name: "linear", "decision_tree", "random_forest"
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = self._init_model()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def _init_model(self):
        """Initialize the selected model."""
        if self.model_name == "linear":
            return LinearRegression()
        elif self.model_name == "decision_tree":
            return DecisionTreeRegressor(random_state=self.random_state)
        elif self.model_name == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

    def train_test_split(self, X: pd.DataFrame, y: pd.Series, test_size=0.2):
        """Split the data into train/test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

    def train(self):
        """Train the model on the training set."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before training.")
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X=None):
        """Predict on new data or test set if X is None."""
        if X is None:
            X = self.X_test
        self.y_pred = self.model.predict(X)
        return self.y_pred

    def evaluate(self):
        """Evaluate the model on the test set."""
        if self.y_test is None or self.y_pred is None:
            raise ValueError("Make predictions before evaluating.")
        r2 = r2_score(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        return {"R2": r2, "MSE": mse}

    def summary(self):
        """Print a summary of model performance."""
        metrics = self.evaluate()
        print(f"Model: {self.model_name}")
        print(f"R2: {metrics['R2']:.4f}")
        print(f"MSE: {metrics['MSE']:.4f}")

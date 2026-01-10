import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


class RegressionModel:
    """
    Wrapper class for training and evaluating regression models.
    Supports linear regression, decision tree, and random forest.
    """

    def __init__(self, model_name: str = "linear", random_state: int = 42):
        """
        Initialize the regression model.

        Parameters:
        -----------
        model_name : str, optional
            Type of regression model to use
            ("linear", "decision_tree", "random_forest")
        random_state : int, optional
            Random seed for reproducibility

        Raises:
        -------
        ValueError
            If model_name is not supported
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = self._init_model()
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.y_pred: np.ndarray | None = None

    def _init_model(self):
        """
        Initialize the selected regression model.

        Returns:
        --------
        object
            Instantiated scikit-learn regression model

        Raises:
        -------
        ValueError
            If model_name is not recognized
        """
        if self.model_name == "linear":
            return LinearRegression()
        elif self.model_name == "decision_tree":
            return DecisionTreeRegressor(random_state=self.random_state)
        elif self.model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=100, random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> None:
        """
        Split data into training and testing sets.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        test_size : float, optional
            Proportion of the dataset to include in the test split

        Returns:
        --------
        None
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

    def train(self) -> None:
        """
        Train the regression model using the training data.

        Returns:
        --------
        None

        Raises:
        -------
        ValueError
            If training data has not been split
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() before training.")
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X: pd.DataFrame | None = None) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Parameters:
        -----------
        X : pd.DataFrame, optional
            Input features for prediction.
            If None, the test set is used.

        Returns:
        --------
        np.ndarray
            Predicted target values
        """
        if X is None:
            X = self.X_test
        self.y_pred = self.model.predict(X)
        return self.y_pred

    def evaluate(self) -> dict:
        """
        Evaluate model performance on the test set.

        Returns:
        --------
        dict
            Dictionary containing R2 score and Mean Squared Error

        Raises:
        -------
        ValueError
            If predictions have not been generated
        """
        if self.y_test is None or self.y_pred is None:
            raise ValueError("Make predictions before evaluating.")
        r2 = r2_score(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        return {"R2": r2, "MSE": mse}

    def summary(self) -> None:
        """
        Print a summary of the model performance.

        Returns:
        --------
        None
        """
        metrics = self.evaluate()
        print(f"Model: {self.model_name}")
        print(f"R2: {metrics['R2']:.4f}")
        print(f"MSE: {metrics['MSE']:.4f}")

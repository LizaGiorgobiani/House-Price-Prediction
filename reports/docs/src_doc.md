# SRC Package Documentation

## Overview
The `src` package contains utility modules for a complete machine learning pipeline, including data preprocessing, visualization, and model implementation. It is designed for structured workflows in house price prediction or similar regression tasks.

---

## Module: `data_processing.py`

### Description
Provides functions for loading, inspecting, and manipulating datasets. This module focuses on general-purpose data preprocessing tasks.

### Functions

- **`load_csv(path: str) -> pd.DataFrame`**  
  Loads a CSV file into a Pandas DataFrame.  
  **Parameters:**  
  - `path`: Path to the CSV file  
  **Returns:**  
  - `pd.DataFrame`

- **`data_info(df: pd.DataFrame)`**  
  Prints the general info of a DataFrame, including column names, types, and non-null counts.  
  **Parameters:**  
  - `df`: Pandas DataFrame  

- **`describe_numeric(df: pd.DataFrame) -> pd.DataFrame`**  
  Returns descriptive statistics for numeric columns only.  
  **Parameters:**  
  - `df`: Pandas DataFrame  
  **Returns:**  
  - Pandas DataFrame with statistics (mean, std, min, max, etc.)

- **`check_missing(df: pd.DataFrame) -> pd.Series`**  
  Returns a count of missing values per column.  

- **`has_duplicates(df: pd.DataFrame) -> bool`**  
  Checks if the DataFrame contains duplicate rows.  

- **`get_numeric_features(df: pd.DataFrame) -> pd.DataFrame`**  
  Returns a DataFrame containing only numeric columns.  

- **`get_categorical_features(df: pd.DataFrame) -> pd.DataFrame`**  
  Returns a DataFrame containing only categorical columns.  

- **`save_csv(df: pd.DataFrame, path: str)`**  
  Saves a DataFrame to CSV at the specified path.  

---

## Module: `visualization.py`

### Description
Provides functions for plotting and saving figures, including histograms, boxplots, scatter plots, heatmaps, and advanced visualizations for numeric and categorical data.

### Functions

- **`save_figure(fig, filepath: str)`**  
  Saves a Matplotlib figure to a file, creating directories if necessary. Closes the figure after saving.

- **`plot_distribution(df, column, bins=30, save_path=None)`**  
  Plots a histogram with KDE for a numeric column.  

- **`plot_box(df, column, save_path=None)`**  
  Plots a boxplot for a numeric column to detect outliers.  

- **`plot_scatter(df, x_col, y_col, regression=False, save_path=None)`**  
  Plots a scatter plot. Optionally adds a regression line.  

- **`plot_heatmap(df, save_path=None)`**  
  Plots a correlation heatmap using numeric columns only.  

- **`plot_bar(df, x_col, y_col, agg_func="mean", save_path=None)`**  
  Creates a bar chart of aggregated values.  

- **`plot_count(df, column, save_path=None)`**  
  Plots a count plot for a categorical variable.  

- **`plot_violin(df, x_col, y_col, save_path=None)`**  
  Creates a violin plot for a categorical x and numeric y.  

- **`plot_strip(df, x_col, y_col, sample_size=None, save_path=None)`**  
  Plots a strip/ swarm plot with optional sampling.  

- **`plot_pair(df, columns, save_path=None)`**  
  Creates a pair plot for selected numeric columns.  

---

## Module: `models.py`

### Description
Contains implementations for regression models used for house price prediction. Supports Linear Regression, Decision Tree Regressor, and Random Forest Regressor. Wraps training, prediction, and evaluation in a single class.

### Class: `RegressionModel`

#### Initialization
```python
```
RegressionModel(model_name="linear", random_state=42)

**Parameters:**

- `model_name (str)`: Type of regression model. Options: `"linear"`, `"decision_tree"`, `"random_forest"`  
- `random_state (int)`: Seed for reproducibility  

---

## Methods

- **`train_test_split(X, y, test_size=0.2)`**  
  Splits dataset into training and testing sets.  

- **`train()`**  
  Trains the model.  

- **`predict(X=None)`**  
  Generates predictions on the test set or on custom input data.  

- **`evaluate() -> dict`**  
  Returns a dictionary with evaluation metrics: R² and MSE.  

- **`summary()`**  
  Prints a summary of the model type and evaluation metrics.  

---

## Example Usage

```python
import pandas as pd
from src import models as ml

# Load dataset
data = pd.read_csv("../data/housing_data_cleaned.csv")
X = data[["Gr Liv Area", "Total Bsmt SF", "Garage Area", "Overall Qual"]]
y = data["SalePrice"]

# Initialize model
reg_model = ml.RegressionModel(model_name="random_forest")

# Split data
reg_model.train_test_split(X, y, test_size=0.2)

# Train the model
reg_model.train()

# Make predictions
predictions = reg_model.predict()

# Evaluate performance
metrics = reg_model.evaluate()
print(metrics)

# Print model summary
reg_model.summary()
```
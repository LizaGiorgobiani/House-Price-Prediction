# ML documentation

## Overview
This notebook implements machine learning models to predict house prices using the cleaned housing dataset (`housing_data_cleaned.csv`). It covers:

1. Feature selection
2. Train/test splitting
3. Training multiple regression models
4. Evaluation and comparison of model performance
5. Identification of the best-performing model

The goal is to select a model that predicts **SalePrice** accurately while balancing interpretability and performance.

---

## Packages and Modules
The notebook imports standard data science libraries:

- `pandas`, `numpy` — data handling and numerical operations  
- `matplotlib.pyplot`, `seaborn` — visualization  
- `os`, `sys` — file and path management  

Custom modules from the `src` package:

- `src.data_processing as dp` — handles loading datasets and feature selection  
- `src.models as mdl` — contains the `RegressionModel` class to train linear, decision tree, and random forest regressors  

---

## Workflow

### 1. Load Data
Load the cleaned dataset:

```python
house_df = dp.load_csv('../data/housing_data_cleaned.csv')
```

Preview the dataset to ensure it loaded correctly.

---

### 2. Feature Selection
- Target variable: `SalePrice`  
- Numerical features selected using `dp.get_numeric_features()`  
- Selected features are used as predictors for the models.

---

### 3. Model Initialization
Three regression models are initialized using the `RegressionModel` class:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**  

Example:

```python
linear_model = mdl.RegressionModel(model_name="linear")
tree_model = mdl.RegressionModel(model_name="decision_tree")
rf_model = mdl.RegressionModel(model_name="random_forest")
```

---

### 4. Train/Test Split
Dataset is split into training and testing sets:

- `X_train`, `X_test` — predictor features  
- `y_train`, `y_test` — target variable  

Test set size: 20% of the total data.

---

### 5. Model Training
Each model is trained on the training data:

```python
model.train()
```

The `RegressionModel` class handles fitting internally.

---

### 6. Prediction
Predictions are generated on the test set:

```python
y_pred = model.predict()
```

---

### 7. Model Evaluation
Models are evaluated using:

1. **R² (Coefficient of Determination)** — higher is better  
2. **RMSE (Root Mean Squared Error)** — lower is better  

```python
metrics = model.evaluate()
```

**Example results:**

| Model           | R²       | RMSE      |
|-----------------|----------|-----------|
| Random Forest   | 0.9152   | 0.1252    |
| Linear          | 0.9037   | 0.1335    |
| Decision Tree   | 0.8376   | 0.1733    |

> Random Forest performs the best overall.

The `summary()` method prints a concise report for each model.

---

### 8. Model Comparison
Performance metrics are compiled into a DataFrame for easy comparison. Visual or tabular comparison confirms that **Random Forest** achieves the best balance of low error and high variance explained.

---

### 9. Insights
- **Random Forest** is the best-performing model.  
- **Linear Regression** performs slightly worse but is interpretable.  
- **Decision Tree** has the lowest performance and may overfit.  

---

### 10. Optional Next Steps
- Hyperparameter tuning for Random Forest  
- Additional feature engineering  
- Cross-validation to ensure robustness  
- Residual analysis to check model assumptions  

---

### 11. File Outputs
No files are explicitly saved in this notebook, but predicted values and performance metrics can be exported for reporting or further analysis.


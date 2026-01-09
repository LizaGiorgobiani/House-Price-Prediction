# 01 Data Exploration – Raw Housing Dataset

## Purpose

The purpose of this notebook is to perform an **initial exploration of the raw housing dataset** before any cleaning, transformation, or feature engineering is applied.  
This step ensures a solid understanding of the dataset’s structure, data quality, variable types, and potential issues that must be addressed during preprocessing.

All analysis in this stage is **read-only** and does not modify the dataset.

---

## Tools and Libraries Used

The following Python libraries were used throughout the data exploration process:

- **Pandas**: Data loading, inspection, and summary statistics  
- **NumPy**: Numerical operations and type identification  
- **Matplotlib & Seaborn**: Visualization libraries (plots generated but not documented here)

---

## Dataset Loading

The raw dataset was loaded directly from the `data/raw/` directory using Pandas:

```python
pd.read_csv('data/raw/housing_data_raw.csv')
```

This ensures a clear separation between **raw data** and **processed data**, following best practices for reproducible data science workflows.

---

## Dataset Structure Overview

An initial preview of the dataset was performed using `.head()` to inspect:

- Column names
- Feature formatting
- Presence of categorical vs numerical variables

The `.info()` method was used to examine:

- Total number of rows and columns
- Data types of each feature
- Non-null value counts
- Early indications of missing data

This step confirmed that the dataset contains a mix of **numerical**, **ordinal**, and **categorical** features.

---

## Descriptive Statistics

Descriptive statistics for all numerical features were generated using:

```python
house_df.describe()
```

This provided insight into:

- Mean, median, and standard deviation
- Minimum and maximum values
- Quartile ranges
- Feature scale differences

Several variables showed wide ranges and skewed distributions, indicating potential outliers and the need for robust preprocessing techniques.

---

## Missing Value Inspection

Missing values were identified using:

```python
house_df.isna().sum()
```

Only columns with at least one missing value were selected for inspection.  
The missing values were primarily concentrated in features related to **optional house characteristics**, such as:

- Alley access
- Basement details
- Garage attributes
- Pool quality
- Fence and miscellaneous features

This suggested that missing values often represent **feature absence rather than data errors**, an important observation for later imputation decisions.

---

## Duplicate Record Check

Duplicate rows were checked using:

```python
house_df.duplicated().any()
```

The result confirmed that **no duplicate records** were present in the dataset.  
Therefore, no rows were removed at this stage.

---

## Numerical Feature Summary

Numerical features were isolated using data type selection and analyzed separately to:

- Identify scale differences
- Detect extreme values
- Observe potential skewness
- Prepare for later normalization or transformation steps

Several numerical variables showed signs of right-skewed distributions and large variance, particularly those related to size and price.

---

## Categorical Feature Summary

Categorical features were examined using value counts for each category.  
This helped identify:

- Dominant categories
- Rare or sparse categories
- Imbalanced distributions
- Potential grouping or encoding challenges

Features such as neighborhood, house style, and exterior materials showed uneven category distributions, which may influence model behavior.

---

## Initial Observations

Based on the exploration performed in this notebook:

- Several columns contain missing values, primarily related to optional property features
- No duplicate rows were found in the dataset
- The target variable (SalePrice) is right-skewed with high-value outliers
- Size-related features exhibit wide ranges and potential outliers
- Many categorical features are imbalanced across categories

---

## Conclusion

This notebook establishes a clear understanding of the **raw dataset** and highlights key data quality issues that must be addressed in subsequent stages.

The findings from this exploration directly inform:
- Missing value handling strategies
- Feature engineering decisions
- Outlier handling approaches
- Model selection and evaluation considerations

The next step in the workflow is **Data Preprocessing**, where missing values, data types, and derived features will be systematically addressed.

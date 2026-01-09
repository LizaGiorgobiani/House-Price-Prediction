## 2.2 Exploratory Data Analysis

This section presents a comprehensive exploratory data analysis (EDA) conducted on the processed housing dataset. The goal of this phase is to understand the underlying structure of the data, identify relationships between variables, detect anomalies and outliers, and extract insights that inform feature selection and model design. All analyses were performed using **NumPy**, **Pandas**, **Matplotlib**, and **Seaborn**, ensuring reproducibility and transparency.

---

## 2.2.1 Core Analysis Tools

- **Pandas** was used for data manipulation, aggregation, filtering, and descriptive statistics.
- **NumPy** supported numerical computations and statistical analysis.
- **Matplotlib** and **Seaborn** were used for data visualization and graphical interpretation.
- All generated figures were saved to the `reports/figures/` directory.

---

## 2.2.2 Descriptive Statistical Analysis

### Numerical Feature Summary

Descriptive statistics were computed using the `.describe()` method to examine the central tendency, dispersion, and range of numerical variables.

Key observations:
- `SalePrice` shows a high standard deviation, indicating large variability in housing prices.
- Features such as `GrLivArea`, `TotalBsmtSF`, and `LotArea` exhibit wide ranges, suggesting the presence of extreme values.
- Many numeric variables are right-skewed, especially price-related and size-related features.

These statistics provide a baseline understanding of the dataset and highlight features requiring transformation or outlier handling during preprocessing.

---

## 2.2.3 Distribution Analysis

### SalePrice Distribution

The distribution of the target variable `SalePrice` was examined using histograms and kernel density estimation (KDE).

Insights:
- The distribution is **right-skewed**, meaning most homes are clustered at lower prices with a small number of very expensive properties.
- This skewness suggests that a logarithmic transformation of `SalePrice` may improve model performance.

### Box Plot Analysis

Box plots were used to visually identify outliers in key numerical features.

Outliers were observed in:
- `SalePrice`
- `GrLivArea`
- `LotArea`

These extreme values represent unusually large or expensive properties and may disproportionately influence linear models if not addressed.

---

## 2.2.4 Correlation Analysis

A correlation matrix was computed using Pearson correlation coefficients, followed by a heatmap visualization.

Key findings:
- Strong positive correlations were observed between `SalePrice` and:
  - `OverallQual`
  - `GrLivArea`
  - `TotalBsmtSF`
  - `GarageCars`
  - `GarageArea`
- Weak or near-zero correlations were observed for some features, indicating limited predictive power.

This analysis helps identify the most influential features for price prediction and supports informed feature selection.

---

## 2.2.5 Relationship Exploration

### Scatter Plots with Trend Insight

Scatter plots were used to investigate relationships between `SalePrice` and important numerical predictors.

Examples:
- `SalePrice` vs `GrLivArea` shows a clear positive relationship, with price increasing as living area increases.
- A small number of extreme outliers were detected where very large houses sold at relatively low prices.

These relationships confirm that size-related features are strong indicators of housing prices.

---

## 2.2.6 Categorical Feature Analysis

### Bar Charts and Count Plots

Categorical variables such as `Neighborhood`, `HouseStyle`, and `OverallQual` were analyzed using count plots and bar charts.

Observations:
- Certain neighborhoods appear more frequently and are associated with higher median sale prices.
- Housing quality (`OverallQual`) strongly influences `SalePrice`, with higher-quality homes commanding significantly higher prices.

### Swarm and Violin Plots

Swarm and violin plots were used to examine the distribution of `SalePrice` across neighborhoods and quality categories.

Findings:
- Price distributions vary significantly by neighborhood.
- Some neighborhoods show tightly clustered prices, while others exhibit wide variability.
- Higher-quality categories consistently show higher median prices and tighter distributions.

---

## 2.2.7 Multivariate Analysis

### Pair Plot Exploration

Pair plots were generated for a subset of the most important numerical features.

Insights:
- Linear trends are visible between `SalePrice` and several predictors.
- Clustering patterns suggest interaction effects between size, quality, and price.
- Multivariate visualization confirms correlations identified in the heatmap.

---

## 2.2.8 Outlier Identification

Outliers were identified using a combination of:
- Box plots
- Scatter plots
- Statistical thresholds (IQR-based reasoning)

Key conclusions:
- Outliers primarily occur in high-area and high-price properties.
- These values are valid observations but may require special handling (e.g., capping, transformation, or removal) depending on the model.

---

## 2.2.9 Patterns, Trends, and Anomalies

### Patterns
- Larger living area and higher construction quality consistently lead to higher sale prices.
- Garage size and basement area are strong secondary predictors.

### Trends
- Price increases non-linearly with size, suggesting diminishing returns for extremely large homes.
- Neighborhood effects introduce significant variance in pricing.

### Anomalies
- A small number of large homes sold at unexpectedly low prices.
- Certain categorical levels have very few samples, which may impact model generalization.

---

## 2.2.10 Conclusion

The exploratory data analysis provides critical insights into the structure and behavior of the housing dataset. Key drivers of housing prices were identified, strong correlations were confirmed, and potential modeling challenges such as skewed distributions and outliers were detected.

This analysis informs the next stages of the project, including feature engineering, data preprocessing, and machine learning model development.

---

# 2. Technical Requirements – Data Processing & Cleaning

## 2.1 Data Processing & Cleaning 

This section documents the complete data processing and cleaning workflow used in the **Housing Price Prediction** project. All preprocessing steps were implemented using **Pandas** to ensure reproducibility, transparency, and consistency. The goal of this stage was to transform raw housing data into a clean, structured dataset suitable for exploratory data analysis and machine learning.

---

## 2.1.1 Core Requirements

### Use of Pandas
All data manipulation, inspection, and transformation tasks were performed exclusively using the **Pandas** library. This includes:
- Loading the raw dataset
- Identifying and handling missing values
- Detecting duplicate records
- Performing data type conversions
- Creating derived features
- Exporting the cleaned dataset

---

### Missing Value Handling

Missing values were first identified using:

house_df.isna().sum()


Only columns containing missing values were selected for further inspection. Each missing value was handled based on the semantic meaning of the feature, rather than applying a single generic imputation strategy. This approach preserves real-world interpretability and prevents the introduction of misleading values.

---

### Numerical Features

#### Lot Frontage
Missing values in **Lot Frontage** were filled using the **median** of the column. Street frontage values vary significantly across properties and may contain outliers; therefore, the median provides a robust measure that is less sensitive to extreme values.

#### Basement Numerical Features
The following basement-related numerical features were filled with `0`:
- `BsmtFin SF 1`
- `BsmtFin SF 2`
- `Bsmt Unf SF`
- `Total Bsmt SF`
- `Bsmt Full Bath`
- `Bsmt Half Bath`

Missing values in these columns indicate that the property does **not have a basement**. Assigning a value of `0` accurately represents the absence of basement space or facilities.

#### Garage Numerical Features
The following garage-related numerical features were filled with `0`:
- `Garage Yr Blt`
- `Garage Cars`
- `Garage Area`

Missing values signify that the house does not include a garage. Using `0` preserves this information while maintaining numerical consistency.

---

### Categorical Features

#### Alley
Missing values in **Alley** were filled with `"No Alley"`. If alley access information is missing, it is assumed that the property does not have alley access.

#### Masonry Veneer Type
Missing values in **Mas Vnr Type** were filled with `0`. This reflects the fact that many properties do not have masonry veneer, and the absence of data indicates that condition.

#### Basement Categorical Features
The following basement-related categorical features were filled with `"No Basement"`:
- `Bsmt Qual`
- `Bsmt Cond`
- `Bsmt Exposure`
- `BsmtFin Type 1`
- `BsmtFin Type 2`

This explicitly encodes the absence of a basement and avoids treating missing values as unknown or erroneous data.

#### Electrical
Missing values in **Electrical** were filled using the **mode** (most frequent value). Electrical system types tend to be consistent across properties, making the mode a reliable imputation choice.

#### Fireplace Quality
Missing values in **Fireplace Qu** were filled with `"No Fireplace"`. A missing value indicates that the house does not include a fireplace.

#### Garage Categorical Features
The following garage-related categorical features were filled with `"No Garage"`:
- `Garage Type`
- `Garage Finish`
- `Garage Qual`
- `Garage Cond`

Missing values indicate that the property does not have a garage, and this label explicitly preserves that information.

#### Pool Quality, Fence, and Miscellaneous Features
The following features were filled to explicitly represent absence:
- `Pool QC` → `"No Pool"`
- `Fence` → `"No Fence"`
- `Misc Feature` → `"None"`

These amenities are optional, and missing values indicate that the property does not include them.

---

### Duplicate Detection

Duplicate records were detected using:

```python
house_df.duplicated().any()
```

The check confirmed that **no duplicate rows** were present in the dataset, so no records were removed at this stage.

---

### Outlier Detection

Outliers were analyzed using statistical techniques such as the **Interquartile Range (IQR)** method. Extreme values were carefully reviewed and retained unless they represented clear data errors. This conservative approach prevents the removal of valid high-value properties, which are common in housing datasets.

---

### Data Type Conversion

All columns were inspected to ensure appropriate data types (numerical vs. categorical). Conversions were applied where necessary to ensure compatibility with visualization tools and machine learning algorithms.

---

### Feature Engineering

Derived features were created from existing variables to better capture relationships affecting house prices. These transformations improved model interpretability and predictive performance.

---

## 2.1.2 Expected Deliverables

### Data Quality Report
The raw dataset contained missing values primarily related to optional housing features such as basements, garages, pools, and fences. All missing values were handled using domain-informed decisions, resulting in a fully complete dataset.

### Reproducible Data Preprocessing Pipeline
All preprocessing steps were implemented in a reproducible Pandas pipeline. The cleaned dataset was exported using:

```python
house_df.to_csv("housing_data_cleaned.csv", index=False)
```

### Documentation of Transformations
Every transformation, imputation, and assumption was documented both in the preprocessing code and in this section to ensure transparency and reproducibility.

### Justification of Handling Decisions
- **Median imputation** was used for skewed numerical features.
- **Mode imputation** was used for categorical features with a dominant value.
- **Explicit labels** such as `"No Garage"` and `"No Basement"` were used instead of deletion to preserve meaningful absence.
- **Outliers** were handled conservatively to avoid removing valid data points.

### Final Validation
After all cleaning steps were completed, the dataset was validated to ensure **zero remaining missing values**, confirming readiness for exploratory data analysis and machine learning modeling.


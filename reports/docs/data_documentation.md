# Data Documentation

## Dataset Overview
This project uses the **Ames Housing Dataset**, which contains detailed information about residential properties in Ames, Iowa.  
The dataset includes approximately **2,900 observations** and **80+ features**, describing physical characteristics, quality, location, and amenities of houses.

- **Dataset Source:** Kaggle – Ames Housing Dataset  
- **Link:** https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset  
- **Target Variable:** `SalePrice` (final sale price of each house)

---

## Data Dictionary (Key Features)

### Target Variable
- **SalePrice**  
  The final sale price of the property in USD.

### Numerical Features
- **Gr Liv Area** – Above-ground living area in square feet  
- **Total Bsmt SF** – Total basement area in square feet  
- **Garage Area** – Garage size in square feet  
- **Overall Qual** – Overall material and finish quality (1–10)  
- **Lot Frontage** – Linear feet of street connected to property  

### Categorical Features
- **Neighborhood** – Physical location within Ames city limits  
- **House Style** – Type of dwelling  
- **Garage Type** – Type of garage  
- **Bsmt Qual** – Basement quality rating  
- **Fireplace Qu** – Fireplace quality  

### Engineered Features
- **Total House Area** – `Gr Liv Area + Total Bsmt SF`  
- **Has Garage** – Binary indicator  
- **Has Basement** – Binary indicator  
- **Has Fireplace** – Binary indicator  
- **SalePrice_log** – Log-transformed sale price  

---

## Instructions to Acquire the Dataset

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset
2. Place the raw file in:
   `data/housing_data_raw.csv`
3. Run preprocessing to generate:
   `data/housing_data_cleaned.csv`

---

## Data Preprocessing Documentation

### Missing Values
- Numerical: median or zero where appropriate  
- Categorical: filled with meaningful labels (e.g., No Garage)

### Outliers
- Detected using IQR method  
- Retained to preserve real-world variability

### Feature Engineering
- Combined size-related features  
- Added binary indicators

### Target Transformation
- Log transformation applied to reduce skewness

---

## Output
- Final cleaned dataset:
  `data/housing_data_cleaned.csv`

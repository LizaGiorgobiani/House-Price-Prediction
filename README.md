# House-Price-Prediction

## Project Overview
This project is a **data-driven approach to predicting house prices** using historical housing data. The goal is to clean and preprocess the dataset, perform exploratory data analysis (EDA), engineer useful features, and train machine learning models to predict house sale prices.  

The project includes the following steps:  
1. **01_data_exploration.ipynb** – Initial exploration of raw data, inspecting columns, data types, and distributions.  
2. **02_data_preprocessing.ipynb** – Cleaning missing values, handling outliers, feature engineering, and target transformation.  
3. **03_eda_visualisations.ipynb** – Visual exploratory data analysis to understand relationships and patterns.  
4. **04_machine_learning.ipynb** – Implementing multiple regression models and comparing their performance.

---

## Project Structure
```
project_root/
│
├── data/
│   ├── housing_data_raw.csv        # Original raw dataset
│   └── housing_data_cleaned.csv    # Cleaned dataset after preprocessing
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda_visualisations.ipynb
│   └── 04_machine_learning.ipynb
│
├── reports/
│   ├── figures/                    # Saved visualizations
│   └── docs/                       # Documentation for each notebook
│
├── src/
│   ├── data_processing.py          # Functions for loading, cleaning, and saving data
│   ├── visualization.py            # Functions for plotting
│   └── models.py                   # Regression model classes
│
└── README.md
```

---

## Installation
1. Clone the repository:
```bash
git clone <your-repo-url>
cd project_root
```

2. Install required packages (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```
**Required packages include:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

---

## 01_data_exploration.ipynb
- Initial inspection of raw dataset.  
- Explore data types, missing values, and feature distributions.  
- Generate summary tables to understand dataset structure and plan preprocessing steps.

---

## 02_data_preprocessing.ipynb
- Handle missing values for categorical and numerical features.  
- Fill basement, garage, fireplace, pool, fence, and miscellaneous features intelligently.  
- Detect outliers using the **IQR method**.  
- Feature engineering:
  - `Total House Area` = `Total Bsmt SF` + `Gr Liv Area`
  - Binary indicators: `Has Garage`, `Has Basement`, `Has Fireplace`  
- Target variable transformation: `SalePrice` is log-transformed to reduce skewness.  
- Save the cleaned dataset as `housing_data_cleaned.csv`.

---

## 03_eda_visualisations.ipynb
- Distribution and boxplot of `SalePrice`.  
- Correlation heatmap for numerical features.  
- Scatter plots of `SalePrice` vs `Gr Liv Area` and `Total Bsmt SF`.  
- Average `SalePrice` by `Neighborhood` (bar chart).  
- Analysis of `Overall Qual` vs `SalePrice` (violin and strip plots).  
- Pair plots for key features: `SalePrice`, `Gr Liv Area`, `Total Bsmt SF`, `Garage Area`, `Overall Qual`.  

All visualizations are saved in `reports/figures/`, and supporting explanations are stored in `reports/docs/`.

---

## 04_machine_learning.ipynb
The notebook implements and compares three regression models using the cleaned dataset:

| Model           | R²       | RMSE    |
|-----------------|----------|---------|
| Random Forest   | 0.9152   | 0.1252  |
| Linear          | 0.9037   | 0.1335  |
| Decision Tree   | 0.8376   | 0.1733  |

**Key points:**
- Random Forest performed best in terms of R² and RMSE.  
- Linear Regression also performed well, indicating a relatively linear relationship between key features and `SalePrice`.  
- Decision Tree underperformed slightly, likely due to overfitting.

The regression models are implemented using the `RegressionModel` class in `src/models.py`.

---

## How to Run
1. **Data Exploration**:
```python
notebooks/01_data_exploration.ipynb
```
2. **Data Preprocessing**:
```python
notebooks/02_data_preprocessing.ipynb
```
3. **EDA & Visualizations**:
```python
notebooks/03_eda_visualisations.ipynb
```
4. **Machine Learning**:
```python
notebooks/04_machine_learning.ipynb
```

---

## Dependencies
- Python >= 3.10  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

---

## Project Highlights
- **End-to-end workflow:** Raw data → Preprocessing → EDA → Feature engineering → Machine learning → Model evaluation.  
- **Reusable code:** Preprocessing, visualization, and ML models are modular in `src/`.  
- **Insightful analysis:** Key features driving house prices include `Gr Liv Area`, `Total Bsmt SF`, and `Overall Qual`.  
- **Strong predictive performance:** Random Forest achieved R² of 0.915.  
- **Comprehensive documentation:** Each notebook has supporting explanations in `reports/docs/`.

---

## Future Improvements
- Add more features (e.g., proximity to amenities, school ratings).  
- Experiment with gradient boosting models (XGBoost, LightGBM).  
- Perform hyperparameter tuning with GridSearchCV or RandomizedSearchCV.  
- Evaluate models using cross-validation for robustness.

---

## Author
**Elizabet Giorgobiani** 


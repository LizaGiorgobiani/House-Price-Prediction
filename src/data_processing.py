import pandas as pd

#load CSV
def load_csv(path):
    return pd.read_csv(path)

#print general DataFrame info
def data_info(df):
    return df.info()

#describe numeric features
def describe_numeric(df):
    numeric_df = df.select_dtypes(include=['number'])
    return numeric_df.describe()

#check missing values
def check_missing(df):
    return df.isna().sum()

#check duplicates
def has_duplicates(df):
    return df.duplicated().any()

#get numeric features
def get_numeric_features(df):
    return df.select_dtypes(include=['number'])

#get categorical features
def get_categorical_features(df):
    return df.select_dtypes(include=['object'])

#save CSV
def save_csv(df, path):
    df.to_csv(path, index=False)

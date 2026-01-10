import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    -----------
    path : str
        File path to the CSV file

    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame containing the CSV data

    Raises:
    -------
    FileNotFoundError
        If the file path does not exist
    """
    return pd.read_csv(path)


def data_info(df: pd.DataFrame) -> None:
    """
    Print general information about the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    None
        Prints DataFrame information to stdout
    """
    return df.info()


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate descriptive statistics for numeric features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        Descriptive statistics of numeric columns
    """
    numeric_df = df.select_dtypes(include=['number'])
    return numeric_df.describe()


def check_missing(df: pd.DataFrame) -> pd.Series:
    """
    Count missing values in each column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.Series
        Number of missing values per column
    """
    return df.isna().sum()


def has_duplicates(df: pd.DataFrame) -> bool:
    """
    Check whether the DataFrame contains duplicate rows.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    bool
        True if duplicates exist, False otherwise
    """
    return df.duplicated().any()


def get_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric features from the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame containing only numeric columns
    """
    return df.select_dtypes(include=['number'])


def get_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract categorical features from the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame containing only categorical columns
    """
    return df.select_dtypes(include=['object'])


def save_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to a CSV file.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to be saved
    path : str
        Output file path

    Returns:
    --------
    None
        Saves the DataFrame as a CSV file
    """
    df.to_csv(path, index=False)

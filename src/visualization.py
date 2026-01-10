# src/visualization.py
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Global settings
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def save_figure(fig: plt.Figure, filepath: str) -> None:
    """
    Save a Matplotlib figure to disk, creating directories if necessary.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object to be saved
    filepath : str
        Output file path

    Returns:
    --------
    None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight")


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot the distribution of a numerical column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column to visualize
    bins : int, optional
        Number of histogram bins
    save_path : str, optional
        File path to save the plot
    show : bool, optional
        Whether to display the plot

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, bins=bins, ax=ax)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_box(
    df: pd.DataFrame,
    column: str,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a boxplot for a numerical column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column to visualize
    save_path : str, optional
        File path to save the plot
    show : bool, optional
        Whether to display the plot

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Boxplot of {column}")

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    regression: bool = False,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a scatter plot between two variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis
    regression : bool, optional
        Whether to include a regression line
    save_path : str, optional
        File path to save the plot
    show : bool, optional
        Whether to display the plot

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots()

    if regression:
        sns.regplot(
            x=x_col,
            y=y_col,
            data=df,
            scatter_kws={"alpha": 0.5},
            ax=ax
        )
    else:
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)

    ax.set_title(f"{y_col} vs {x_col}")

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a correlation heatmap for numeric features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    save_path : str, optional
        File path to save the plot
    show : bool, optional
        Whether to display the plot

    Returns:
    --------
    None
    """
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Correlation Heatmap")

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    agg_func: str = "mean",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot an aggregated bar chart.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    x_col : str
        Categorical column
    y_col : str
        Numerical column
    agg_func : str, optional
        Aggregation function ("mean", "sum", etc.)
    save_path : str, optional
        File path to save the plot
    show : bool, optional
        Whether to display the plot

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    grouped = df.groupby(x_col)[y_col].agg(agg_func).sort_values(ascending=False)
    grouped.plot(kind="bar", ax=ax)

    ax.set_title(f"{agg_func.title()} {y_col} by {x_col}")
    ax.set_ylabel(f"{agg_func.title()} {y_col}")
    plt.xticks(rotation=90)

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_count(
    df: pd.DataFrame,
    column: str,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a count plot for a categorical column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column to visualize
    save_path : str, optional
        File path to save the plot
    show : bool, optional
        Whether to display the plot

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots()
    sns.countplot(x=column, data=df, ax=ax)
    ax.set_title(f"Distribution of {column}")

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_violin(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a violin plot for grouped distributions.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    x_col : str
        Categorical column
    y_col : str
        Numerical column
    save_path : str, optional
        File path to save the plot
    show : bool, optional
        Whether to display the plot

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x=x_col, y=y_col, data=df, ax=ax)
    ax.set_title(f"{y_col} Distribution by {x_col}")

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_strip(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sample_size: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a strip plot, optionally sampling the data.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    x_col : str
        Categorical column
    y_col : str
        Numerical column
    sample_size : int, optional
        Number of samples to plot
    save_path : str, optional
        File path to save the plot
    show : bool, optional
        Whether to display the plot

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = df.sample(sample_size, random_state=42) if sample_size else df
    sns.stripplot(x=x_col, y=y_col, data=plot_df, ax=ax)
    ax.set_title(f"{y_col} vs {x_col} (Strip Plot)")

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_pair(
    df: pd.DataFrame,
    columns: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot pairwise relationships between selected columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list[str]
        Columns to include in the pair plot
    save_path : str, optional
        File path to save the plot
    show : bool, optional
        Whether to display the plot

    Returns:
    --------
    None
    """
    g = sns.pairplot(df[columns])

    if save_path:
        g.savefig(save_path)
    if show:
        plt.show()

# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

#global settings
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def save_figure(fig, filepath: str):
    """Save a Matplotlib figure to file, creating directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, bbox_inches='tight')

def plot_distribution(df, column, bins=30, save_path=None, show=True):
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

def plot_box(df, column, save_path=None, show=True):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Boxplot of {column}")
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_scatter(df, x_col, y_col, regression=False, save_path=None, show=True):
    fig, ax = plt.subplots()
    if regression:
        sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={"alpha":0.5}, ax=ax)
    else:
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
    ax.set_title(f"{y_col} vs {x_col}")
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_heatmap(df, save_path=None, show=True):
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

def plot_bar(df, x_col, y_col, agg_func="mean", save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(12,6))
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

def plot_count(df, column, save_path=None, show=True):
    fig, ax = plt.subplots()
    sns.countplot(x=column, data=df, ax=ax)
    ax.set_title(f"Distribution of {column}")
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_violin(df, x_col, y_col, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.violinplot(x=x_col, y=y_col, data=df, ax=ax)
    ax.set_title(f"{y_col} Distribution by {x_col}")
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_strip(df, x_col, y_col, sample_size=None, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(12,6))
    plot_df = df.sample(sample_size, random_state=42) if sample_size else df
    sns.stripplot(x=x_col, y=y_col, data=plot_df, ax=ax)
    ax.set_title(f"{y_col} vs {x_col} (Strip Plot)")
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_pair(df, columns, save_path=None, show=True):
    g = sns.pairplot(df[columns])
    if save_path:
        g.savefig(save_path)
    if show:
        plt.show()

"""
EDA Visualization Module

Creates comprehensive exploratory data analysis plots for the Wine Quality dataset.
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoids Tkinter issues

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_eda_plots(df: pd.DataFrame, 
                    save_dir: str = "reports",
                    target_column: str = 'quality') -> Dict[str, str]:
    """
    Create comprehensive EDA plots for the wine quality dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        save_dir (str): Directory to save plots
        target_column (str): Name of target column
        
    Returns:
        Dict[str, str]: Dictionary mapping plot names to file paths
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("Creating comprehensive EDA plots...")
    
    plot_paths = {}
    
    # 1. Data overview
    plot_paths['data_overview'] = plot_data_overview(df, save_dir)
    
    # 2. Target distribution
    plot_paths['target_distribution'] = plot_target_distribution(df, target_column, save_dir)
    
    # 3. Feature distributions
    plot_paths['feature_distributions'] = plot_feature_distributions(df, target_column, save_dir)
    
    # 4. Correlation heatmap
    plot_paths['correlation_heatmap'] = plot_correlation_heatmap(df, save_dir)
    
    # 5. Box plots
    plot_paths['box_plots'] = plot_box_plots(df, target_column, save_dir)
    
    # 6. Pair plots
    plot_paths['pair_plots'] = plot_pair_plots(df, target_column, save_dir)
    
    # 7. Missing values
    plot_paths['missing_values'] = plot_missing_values(df, save_dir)
    
    # 8. Outlier analysis
    plot_paths['outlier_analysis'] = plot_outlier_analysis(df, save_dir)
    
    logger.info(f"EDA plots created and saved to {save_dir}")
    return plot_paths

def plot_data_overview(df: pd.DataFrame, save_dir: str) -> str:
    """Create data overview plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Dataset shape
    axes[0, 0].text(0.5, 0.5, f'Dataset Shape:\n{df.shape[0]} rows × {df.shape[1]} columns', 
                    ha='center', va='center', fontsize=14, transform=axes[0, 0].transAxes)
    axes[0, 0].set_title('Dataset Overview', fontsize=16)
    axes[0, 0].axis('off')
    
    # Data types
    dtype_counts = df.dtypes.value_counts()
    axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Data Types Distribution', fontsize=16)
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    axes[1, 0].bar(['Memory Usage'], [memory_usage])
    axes[1, 0].set_title(f'Memory Usage: {memory_usage:.2f} MB', fontsize=16)
    axes[1, 0].set_ylabel('MB')
    
    # Missing values
    missing_count = df.isnull().sum().sum()
    missing_pct = (missing_count / (df.shape[0] * df.shape[1])) * 100
    axes[1, 1].bar(['Missing Values'], [missing_pct])
    axes[1, 1].set_title(f'Missing Values: {missing_pct:.2f}%', fontsize=16)
    axes[1, 1].set_ylabel('Percentage')
    
    plt.tight_layout()
    
    filepath = f"{save_dir}/01_data_overview.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_target_distribution(df: pd.DataFrame, target_column: str, save_dir: str) -> str:
    """Create target variable distribution plots."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    value_counts = df[target_column].value_counts().sort_index()
    axes[0].bar(value_counts.index, value_counts.values)
    axes[0].set_title(f'{target_column.title()} Distribution (Count)', fontsize=14)
    axes[0].set_xlabel(target_column.title())
    axes[0].set_ylabel('Count')
    
    # Add value labels on bars
    for i, v in enumerate(value_counts.values):
        axes[0].text(value_counts.index[i], v + 0.1, str(v), ha='center', va='bottom')
    
    # Percentage plot
    value_pct = df[target_column].value_counts(normalize=True).sort_index() * 100
    axes[1].bar(value_pct.index, value_pct.values)
    axes[1].set_title(f'{target_column.title()} Distribution (Percentage)', fontsize=14)
    axes[1].set_xlabel(target_column.title())
    axes[1].set_ylabel('Percentage')
    
    # Add percentage labels on bars
    for i, v in enumerate(value_pct.values):
        axes[1].text(value_pct.index[i], v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    filepath = f"{save_dir}/02_target_distribution.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_feature_distributions(df: pd.DataFrame, target_column: str, save_dir: str) -> str:
    """Create feature distribution plots."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            # Histogram
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col.title()} Distribution', fontsize=12)
            axes[i].set_xlabel(col.title())
            axes[i].set_ylabel('Frequency')
            
            # Add statistics
            mean_val = df[col].mean()
            std_val = df[col].std()
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_val:.2f}')
            axes[i].axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
            axes[i].legend()
    
    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    filepath = f"{save_dir}/03_feature_distributions.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str) -> str:
    """Create correlation heatmap."""
    # Calculate correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8})
    
    plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
    plt.tight_layout()
    
    filepath = f"{save_dir}/04_correlation_heatmap.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_box_plots(df: pd.DataFrame, target_column: str, save_dir: str) -> str:
    """Create box plots for features by target variable."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            # Create box plot
            df.boxplot(column=col, by=target_column, ax=axes[i])
            axes[i].set_title(f'{col.title()} by {target_column.title()}', fontsize=12)
            axes[i].set_xlabel(target_column.title())
            axes[i].set_ylabel(col.title())
    
    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions by Target Variable', fontsize=16, y=1.02)
    plt.tight_layout()
    
    filepath = f"{save_dir}/05_box_plots.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_pair_plots(df: pd.DataFrame, target_column: str, save_dir: str) -> str:
    """Create pair plots for selected features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if target_column in df.columns:
        correlations = df[numeric_cols + [target_column]].corr()[target_column].abs().sort_values(ascending=False)
        top_features = correlations.head(6).index.tolist()
        if target_column in top_features:
            top_features.remove(target_column)
    else:
        top_features = numeric_cols[:6]

    pair_plot_data = df[top_features + [target_column]]

    g = sns.pairplot(pair_plot_data, hue=target_column, diag_kind="hist")
    g.fig.suptitle("Pair Plot of Top Features", y=1.02, fontsize=16)

    filepath = f"{save_dir}/06_pair_plots.png"
    g.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    return filepath


def plot_missing_values(df: pd.DataFrame, save_dir: str) -> str:
    """Create missing values analysis plots."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Missing values count
    missing_count = df.isnull().sum()
    missing_count = missing_count[missing_count > 0].sort_values(ascending=False)
    
    if len(missing_count) > 0:
        axes[0].bar(range(len(missing_count)), missing_count.values)
        axes[0].set_xticks(range(len(missing_count)))
        axes[0].set_xticklabels(missing_count.index, rotation=45, ha='right')
        axes[0].set_title('Missing Values Count by Column', fontsize=14)
        axes[0].set_ylabel('Count')
    else:
        axes[0].text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', fontsize=14)
        axes[0].set_title('Missing Values Analysis', fontsize=14)
    
    # Missing values percentage
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if len(missing_pct) > 0:
        axes[1].bar(range(len(missing_pct)), missing_pct.values)
        axes[1].set_xticks(range(len(missing_pct)))
        axes[1].set_xticklabels(missing_pct.index, rotation=45, ha='right')
        axes[1].set_title('Missing Values Percentage by Column', fontsize=14)
        axes[1].set_ylabel('Percentage')
    else:
        axes[1].text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', fontsize=14)
        axes[1].set_title('Missing Values Analysis', fontsize=14)
    
    plt.tight_layout()
    
    filepath = f"{save_dir}/07_missing_values.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_outlier_analysis(df: pd.DataFrame, save_dir: str) -> str:
    """Create outlier analysis plots."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            # Box plot for outlier detection
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'{col.title()} - Outlier Analysis', fontsize=12)
            axes[i].set_ylabel(col.title())
            
            # Calculate and display outlier count
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            axes[i].text(0.02, 0.98, f'Outliers: {outlier_count}', 
                        transform=axes[i].transAxes, va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Outlier Analysis by Feature', fontsize=16, y=1.02)
    plt.tight_layout()
    
    filepath = f"{save_dir}/08_outlier_analysis.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_data_distribution(df: pd.DataFrame, 
                          columns: List[str] = None,
                          save_path: str = None) -> None:
    """
    Plot distribution of specified columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        columns (List[str], optional): Columns to plot
        save_path (str, optional): Path to save the plot
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col.title()} Distribution')
            axes[i].set_xlabel(col.title())
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution plot saved to {save_path}")
    
    plt.close()
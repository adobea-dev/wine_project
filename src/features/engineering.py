"""
Feature Engineering Module

Creates new features and transformations for the Wine Quality dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def create_interaction_features(df: pd.DataFrame, 
                              feature_pairs: List[Tuple[str, str]] = None,
                              include_self_interactions: bool = True,
                              target_cols: List[str] = ['quality', 'quality_category']) -> pd.DataFrame:
    """
    Create interaction features between pairs of variables safely 
    (excluding target variables to prevent data leakage).
    """
    logger.info("Creating interaction features...")
    df_interactions = df.copy()

    # Exclude target columns from numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in target_cols]

    if feature_pairs is None:
        feature_pairs = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i <= j:
                    feature_pairs.append((col1, col2))

    interaction_count = 0
    for col1, col2 in feature_pairs:
        if col1 in df.columns and col2 in df.columns:
            if col1 == col2 and include_self_interactions:
                interaction_name = f"{col1}_squared"
                df_interactions[interaction_name] = df[col1] ** 2
                interaction_count += 1
            elif col1 != col2:
                interaction_name = f"{col1}_x_{col2}"
                df_interactions[interaction_name] = df[col1] * df[col2]
                interaction_count += 1

    logger.info(f"Created {interaction_count} safe interaction features (excluding targets)")
    return df_interactions

def create_polynomial_features(df: pd.DataFrame, 
                             degree: int = 2,
                             include_bias: bool = False,
                             feature_names: List[str] = None) -> pd.DataFrame:
    """
    Create polynomial features using sklearn's PolynomialFeatures.
    
    Args:
        df (pd.DataFrame): Input dataset
        degree (int): Degree of polynomial features
        include_bias (bool): Whether to include bias term
        feature_names (List[str], optional): Specific features to use
        
    Returns:
        pd.DataFrame: Dataset with polynomial features
    """
    logger.info(f"Creating polynomial features (degree={degree})...")
    
    if feature_names is None:
        # Use all numeric columns
        feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Select features for polynomial transformation
    X_poly = df[feature_names].values
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_poly_transformed = poly.fit_transform(X_poly)
    
    # Create feature names
    poly_feature_names = poly.get_feature_names_out(feature_names)
    
    # Create DataFrame with polynomial features
    df_poly = pd.DataFrame(X_poly_transformed, columns=poly_feature_names, index=df.index)
    
    # Add original non-polynomial columns
    non_poly_cols = [col for col in df.columns if col not in feature_names]
    for col in non_poly_cols:
        df_poly[col] = df[col]
    
    logger.info(f"Created {len(poly_feature_names)} polynomial features")
    return df_poly 

def create_ratio_features(df: pd.DataFrame, 
                         numerator_cols: List[str],
                         denominator_cols: List[str]) -> pd.DataFrame:
    """
    Create ratio features between numerator and denominator columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        numerator_cols (List[str]): Columns to use as numerators
        denominator_cols (List[str]): Columns to use as denominators
        
    Returns:
        pd.DataFrame: Dataset with ratio features
    """
    logger.info("Creating ratio features...")
    df_ratios = df.copy()
    
    ratio_count = 0
    for num_col in numerator_cols:
        for den_col in denominator_cols:
            if num_col in df.columns and den_col in df.columns and num_col != den_col:
                # Avoid division by zero
                ratio_name = f"{num_col}_over_{den_col}"
                df_ratios[ratio_name] = np.where(
                    df[den_col] != 0, 
                    df[num_col] / df[den_col], 
                    0
                )
                ratio_count += 1
    
    logger.info(f"Created {ratio_count} ratio features")
    return df_ratios

def create_log_features(df: pd.DataFrame, 
                       columns: List[str] = None,
                       add_constant: float = 1.0) -> pd.DataFrame:
    """
    Create logarithmic features for specified columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        columns (List[str], optional): Columns to transform. Defaults to all numeric columns
        add_constant (float): Constant to add before taking log to handle zeros
        
    Returns:
        pd.DataFrame: Dataset with log features
    """
    logger.info("Creating logarithmic features...")
    df_log = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    log_count = 0
    for col in columns:
        if col in df.columns:
            # Add constant to handle zeros and negative values
            log_col = f"log_{col}"
            df_log[log_col] = np.log(df[col] + add_constant)
            log_count += 1
    
    logger.info(f"Created {log_count} logarithmic features")
    return df_log

def create_binning_features(df: pd.DataFrame, 
                           columns: List[str] = None,
                           bins: int = 5,
                           method: str = 'quantile') -> pd.DataFrame:
    """
    Create binned features for continuous variables.
    
    Args:
        df (pd.DataFrame): Input dataset
        columns (List[str], optional): Columns to bin. Defaults to all numeric columns
        bins (int): Number of bins
        method (str): Binning method ('quantile', 'uniform', 'kmeans')
        
    Returns:
        pd.DataFrame: Dataset with binned features
    """
    logger.info(f"Creating binned features (bins={bins}, method={method})...")
    df_binned = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    bin_count = 0
    for col in columns:
        if col in df.columns and col != 'quality_category':  # Don't bin target variable
            bin_col = f"{col}_binned"
            
            if method == 'quantile':
                df_binned[bin_col] = pd.qcut(df[col], q=bins, duplicates='drop', labels=False)
            elif method == 'uniform':
                df_binned[bin_col] = pd.cut(df[col], bins=bins, labels=False)
            elif method == 'kmeans':
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=bins, random_state=42)
                df_binned[bin_col] = kmeans.fit_predict(df[[col]])
            
            bin_count += 1
    
    logger.info(f"Created {bin_count} binned features")
    return df_binned

def create_statistical_features(df: pd.DataFrame, 
                              window_size: int = 10,
                              columns: List[str] = None) -> pd.DataFrame:
    """
    Create statistical features using rolling windows.
    
    Args:
        df (pd.DataFrame): Input dataset
        window_size (int): Size of rolling window
        columns (List[str], optional): Columns to create statistics for
        
    Returns:
        pd.DataFrame: Dataset with statistical features
    """
    logger.info(f"Creating statistical features (window_size={window_size})...")
    df_stats = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stat_count = 0
    for col in columns:
        if col in df.columns and col != 'quality_category':  # Don't create stats for target
            # Rolling mean
            df_stats[f"{col}_rolling_mean"] = df[col].rolling(window=window_size).mean()
            
            # Rolling std
            df_stats[f"{col}_rolling_std"] = df[col].rolling(window=window_size).std()
            
            # Rolling min/max
            df_stats[f"{col}_rolling_min"] = df[col].rolling(window=window_size).min()
            df_stats[f"{col}_rolling_max"] = df[col].rolling(window=window_size).max()
            
            stat_count += 4
    
    # Fill NaN values created by rolling window
    df_stats = df_stats.fillna(method='bfill').fillna(method='ffill')
    
    logger.info(f"Created {stat_count} statistical features")
    return df_stats
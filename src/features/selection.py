"""
Feature Selection Module

Handles feature selection methods for the Wine Quality dataset.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def select_features_correlation(df: pd.DataFrame, 
                               target_column: str = 'quality_category',
                               threshold: float = 0.1) -> List[str]:
    """
    Select features based on correlation with target variable.
    
    Args:
        df (pd.DataFrame): Dataset with features and target
        target_column (str): Name of target column
        threshold (float): Minimum correlation threshold
        
    Returns:
        List[str]: List of selected feature names
    """
    logger.info(f"Selecting features based on correlation (threshold={threshold})")
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target column and other quality-related columns
    exclude_cols = [target_column, 'quality', 'quality_label', 'quality_category']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlations with target
    correlations = df[numeric_cols + [target_column]].corr()[target_column].abs()
    
    # Select features above threshold
    selected_features = correlations[correlations >= threshold].index.tolist()
    
    # Remove target column from results
    if target_column in selected_features:
        selected_features.remove(target_column)
    
    logger.info(f"Selected {len(selected_features)} features out of {len(numeric_cols)}")
    logger.info(f"Selected features: {selected_features}")
    
    return selected_features

def select_features_mutual_info(df: pd.DataFrame,
                               target_column: str = 'quality_category',
                               k: int = 10) -> List[str]:
    """
    Select features using mutual information.
    
    Args:
        df (pd.DataFrame): Dataset with features and target
        target_column (str): Name of target column
        k (int): Number of top features to select
        
    Returns:
        List[str]: List of selected feature names
    """
    from sklearn.feature_selection import mutual_info_classif
    
    logger.info(f"Selecting top {k} features using mutual information")
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target column if present
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    # Calculate mutual information
    X = df[numeric_cols]
    y = df[target_column]
    
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Get top k features
    feature_scores = list(zip(numeric_cols, mi_scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    selected_features = [feature for feature, score in feature_scores[:k]]
    
    logger.info(f"Selected {len(selected_features)} features using mutual information")
    logger.info(f"Selected features: {selected_features}")
    
    return selected_features

def select_features_rfe(df: pd.DataFrame,
                       target_column: str = 'quality_category',
                       n_features: int = 10,
                       estimator=None) -> List[str]:
    """
    Select features using Recursive Feature Elimination.
    
    Args:
        df (pd.DataFrame): Dataset with features and target
        target_column (str): Name of target column
        n_features (int): Number of features to select
        estimator: Base estimator for RFE
        
    Returns:
        List[str]: List of selected feature names
    """
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    
    logger.info(f"Selecting {n_features} features using RFE")
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target column if present
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    # Use RandomForest as default estimator
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Prepare data
    X = df[numeric_cols]
    y = df[target_column]
    
    # Apply RFE
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    
    # Get selected features
    selected_features = [col for col, selected in zip(numeric_cols, rfe.support_) if selected]
    
    logger.info(f"Selected {len(selected_features)} features using RFE")
    logger.info(f"Selected features: {selected_features}")
    
    return selected_features
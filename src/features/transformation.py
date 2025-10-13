"""
Feature Transformation Module

Handles scaling, normalization, and other transformations for features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class StandardScalerWrapper:
    """
    Wrapper for StandardScaler with additional functionality.
    """
    
    def __init__(self, columns: Optional[List[str]] = None):
        self.scaler = StandardScaler()
        self.columns = columns
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'StandardScalerWrapper':
        """Fit the scaler on the data."""
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.scaler.fit(X[self.columns])
        self.is_fitted = True
        logger.info(f"StandardScaler fitted on {len(self.columns)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(X).transform(X)

class MinMaxScalerWrapper:
    """
    Wrapper for MinMaxScaler with additional functionality.
    """
    
    def __init__(self, columns: Optional[List[str]] = None, feature_range: tuple = (0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.columns = columns
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'MinMaxScalerWrapper':
        """Fit the scaler on the data."""
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.scaler.fit(X[self.columns])
        self.is_fitted = True
        logger.info(f"MinMaxScaler fitted on {len(self.columns)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(X).transform(X)

class RobustScalerWrapper:
    """
    Wrapper for RobustScaler with additional functionality.
    """
    
    def __init__(self, columns: Optional[List[str]] = None, 
                 with_centering: bool = True, with_scaling: bool = True):
        self.scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling)
        self.columns = columns
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'RobustScalerWrapper':
        """Fit the scaler on the data."""
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.scaler.fit(X[self.columns])
        self.is_fitted = True
        logger.info(f"RobustScaler fitted on {len(self.columns)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(X).transform(X)

def apply_power_transformation(df: pd.DataFrame, 
                             columns: Optional[List[str]] = None,
                             method: str = 'yeo-johnson') -> pd.DataFrame:
    """
    Apply power transformation to make data more Gaussian-like.
    
    Args:
        df (pd.DataFrame): Input dataset
        columns (Optional[List[str]]): Columns to transform
        method (str): Transformation method ('yeo-johnson', 'box-cox')
        
    Returns:
        pd.DataFrame: Transformed dataset
    """
    logger.info(f"Applying power transformation (method={method})...")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_transformed = df.copy()
    transformer = PowerTransformer(method=method)
    
    for col in columns:
        if col in df.columns and col != 'quality_category':  # Don't transform target
            # Reshape for sklearn
            col_data = df[col].values.reshape(-1, 1)
            
            # Fit and transform
            transformed_data = transformer.fit_transform(col_data)
            df_transformed[col] = transformed_data.flatten()
    
    logger.info(f"Applied power transformation to {len(columns)} columns")
    return df_transformed

def apply_quantile_transformation(df: pd.DataFrame, 
                                columns: Optional[List[str]] = None,
                                output_distribution: str = 'uniform') -> pd.DataFrame:
    """
    Apply quantile transformation to map data to a uniform or normal distribution.
    
    Args:
        df (pd.DataFrame): Input dataset
        columns (Optional[List[str]]): Columns to transform
        output_distribution (str): Target distribution ('uniform', 'normal')
        
    Returns:
        pd.DataFrame: Transformed dataset
    """
    logger.info(f"Applying quantile transformation (output={output_distribution})...")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_transformed = df.copy()
    transformer = QuantileTransformer(output_distribution=output_distribution, random_state=42)
    
    for col in columns:
        if col in df.columns and col != 'quality_category':  # Don't transform target
            # Reshape for sklearn
            col_data = df[col].values.reshape(-1, 1)
            
            # Fit and transform
            transformed_data = transformer.fit_transform(col_data)
            df_transformed[col] = transformed_data.flatten()
    
    logger.info(f"Applied quantile transformation to {len(columns)} columns")
    return df_transformed

def create_polynomial_features(df: pd.DataFrame, 
                             degree: int = 2,
                             include_bias: bool = False) -> pd.DataFrame:
    """
    Create polynomial features for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        degree (int): Degree of polynomial features
        include_bias (bool): Whether to include bias term
        
    Returns:
        pd.DataFrame: Dataset with polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    logger.info(f"Creating polynomial features (degree={degree})...")
    
    # Get numeric columns (excluding target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'quality_category' in numeric_cols:
        numeric_cols.remove('quality_category')
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_poly = poly.fit_transform(df[numeric_cols])
    
    # Create feature names
    poly_feature_names = poly.get_feature_names_out(numeric_cols)
    
    # Create new DataFrame
    df_poly = pd.DataFrame(X_poly, columns=poly_feature_names, index=df.index)
    
    # Add non-numeric columns back
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    for col in non_numeric_cols:
        df_poly[col] = df[col]
    
    logger.info(f"Created {len(poly_feature_names)} polynomial features")
    return df_poly

def handle_skewness(df: pd.DataFrame, 
                   columns: Optional[List[str]] = None,
                   threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle skewed features by applying appropriate transformations.
    
    Args:
        df (pd.DataFrame): Input dataset
        columns (Optional[List[str]]): Columns to check for skewness
        threshold (float): Skewness threshold for transformation
        
    Returns:
        pd.DataFrame: Dataset with skewness handled
    """
    from scipy.stats import skew
    
    logger.info(f"Handling skewness (threshold={threshold})...")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_skewed = df.copy()
    transformed_count = 0
    
    for col in columns:
        if col in df.columns and col != 'quality_category':  # Don't transform target
            col_skewness = abs(skew(df[col].dropna()))
            
            if col_skewness > threshold:
                # Apply log transformation for positive skewness
                if skew(df[col].dropna()) > 0:
                    df_skewed[col] = np.log1p(df[col])  # log1p handles zeros
                    logger.info(f"Applied log transformation to {col} (skewness: {col_skewness:.3f})")
                else:
                    # Apply square root transformation for negative skewness
                    df_skewed[col] = np.sqrt(np.abs(df[col])) * np.sign(df[col])
                    logger.info(f"Applied sqrt transformation to {col} (skewness: {col_skewness:.3f})")
                
                transformed_count += 1
    
    logger.info(f"Transformed {transformed_count} skewed features")
    return df_skewed
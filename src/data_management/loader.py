"""
Data Loader Module

Handles loading and basic validation of the Wine Quality dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Expected schema for Wine Quality dataset
EXPECTED_COLUMNS = [
    #'type', 
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'
] 

# Data types for validation
EXPECTED_DTYPES = {
    #'type': 'object',
    'fixed acidity': 'float64',
    'volatile acidity': 'float64',
    'citric acid': 'float64',
    'residual sugar': 'float64',
    'chlorides': 'float64',
    'free sulfur dioxide': 'float64',
    'total sulfur dioxide': 'float64',
    'density': 'float64',
    'pH': 'float64',
    'sulphates': 'float64',
    'alcohol': 'float64',
    'quality': 'int64'
}

def load_wine_data(file_path: Path) -> pd.DataFrame:
    """
    Load the Wine Quality dataset from a CSV file.
    Automatically detects separator and handles malformed rows.

    Args:
        file_path (Path): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be parsed or is empty
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    logger.info(f"Loading dataset from: {file_path}")

    df = None
    try:
        # Try comma separator
        try:
            df = pd.read_csv(file_path, sep=",")
        except Exception as e1:
            logger.warning(f"Failed to read with comma separator: {e1}")
            # Try semicolon separator
            try:
                df = pd.read_csv(file_path, sep=";")
            except Exception as e2:
                logger.warning(f"Failed to read with semicolon separator: {e2}")
                raise ValueError(f"Could not read file {file_path} with ',' or ';' separators.")

        # Handle the case where all data is in one column (common with misread CSVs)
        if df.shape[1] == 1 and ";" in str(df.iloc[0, 0]):
            logger.info("Detected single concatenated column, splitting data...")
            first_row = df.columns[0]
            column_names = [col.strip().strip('"') for col in first_row.split(";")]

            data_rows = []
            for _, row in df.iterrows():
                values = str(row.iloc[0]).split(";")
                data_rows.append([val.strip().strip('"') for val in values])

            df = pd.DataFrame(data_rows, columns=column_names)

            # Convert numeric columns where possible
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass  # Ignore if conversion fails (likely categorical)

        # Sanity check
        if df is None or df.empty:
            raise ValueError(f"Loaded dataframe is empty or invalid from file: {file_path}")

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise ValueError(f"Failed to load dataset: {e}")
    
def validate_data_schema(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate dataset schema and return validation results.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, validation_info)
    """
    validation_info = {
        "shape": df.shape,
        "missing_columns": [],
        "extra_columns": [],
        "type_mismatches": [],
        "missing_values": {},
        "duplicate_rows": 0,
        "is_valid": True
    }
    
    # Check for missing columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    validation_info["missing_columns"] = list(missing_cols)
    
    # Check for extra columns
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    validation_info["extra_columns"] = list(extra_cols)
    
    # Check data types
    for col in EXPECTED_COLUMNS:
        if col in df.columns:
            expected_type = EXPECTED_DTYPES[col]
            actual_type = str(df[col].dtype)
            
            if expected_type != actual_type:
                validation_info["type_mismatches"].append({
                    "column": col,
                    "expected": expected_type,
                    "actual": actual_type
                })
    
    # Check for missing values
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            validation_info["missing_values"][col] = int(missing_count)
    
    # Check for duplicate rows
    validation_info["duplicate_rows"] = int(df.duplicated().sum())
    
    # Overall validation
    validation_info["is_valid"] = (
        len(validation_info["missing_columns"]) == 0 and
        len(validation_info["type_mismatches"]) == 0
    )
    
    if validation_info["is_valid"]:
        logger.info("Dataset validation passed")
    else:
        logger.warning(f"Dataset validation issues: {validation_info}")
    
    return validation_info["is_valid"], validation_info 

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data summary.
    
    Args:
        df (pd.DataFrame): Dataset to summarize
        
    Returns:
        Dict[str, Any]: Summary statistics and information
    """
    summary = {
        "basic_info": {
            "shape": df.shape,
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "columns": list(df.columns)
        },
        "data_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        "categorical_summary": {}
    }
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        summary["categorical_summary"][col] = {
            "unique_values": int(df[col].nunique()),
            "value_counts": df[col].value_counts().to_dict()
        }
    
    # Target variable analysis
    if 'quality' in df.columns:
        summary["target_analysis"] = {
            "unique_values": sorted(df['quality'].unique()),
            "value_counts": df['quality'].value_counts().sort_index().to_dict(),
            "class_distribution": (df['quality'].value_counts(normalize=True) * 100).round(2).to_dict()
        }
    
    return summary

def detect_outliers_iqr(df: pd.DataFrame, columns: list = None) -> Dict[str, list]:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        columns (list, optional): Columns to check. Defaults to numeric columns
        
    Returns:
        Dict[str, list]: Dictionary with outlier indices for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = df[outlier_mask].index.tolist()
    
    return outliers
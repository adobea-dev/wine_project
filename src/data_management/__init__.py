"""
Data Management Module

This module provides functions for downloading, loading, and preprocessing
the Wine Quality dataset.
"""

# Import from downloader module
from .downloader import (
    download_wine_dataset,
    get_data_info,
    check_kaggle_credentials,
    download_via_kaggle,
    download_via_fallback
)

# Import from loader module
from .loader import (
    load_wine_data,
    validate_data_schema,
    get_data_summary,
    detect_outliers_iqr
)

# Import from processor module
from .processor import (
    clean_data,
    create_quality_categories,
    split_data,
    get_feature_importance_correlation
)

__all__ = [
    # Downloader functions
    'download_wine_dataset',
    'get_data_info',
    'check_kaggle_credentials',
    'download_via_kaggle',
    'download_via_fallback',
    
    # Loader functions
    'load_wine_data',
    'validate_data_schema',
    'get_data_summary',
    'detect_outliers_iqr',
    
    # Processor functions
    'clean_data',
    'create_quality_categories',
    'split_data',
    'get_feature_importance_correlation'
]
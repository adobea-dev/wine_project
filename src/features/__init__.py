"""
Features Module

This module provides functions for feature engineering, transformation, and selection.
"""

# Import from engineering module
from .engineering import (
    create_interaction_features,
    create_polynomial_features,
    create_ratio_features,
    create_log_features,
    create_binning_features,
    create_statistical_features
)

# Import from transformation module
from .transformation import (
    StandardScalerWrapper,
    MinMaxScalerWrapper,
    RobustScalerWrapper,
    apply_power_transformation,
    apply_quantile_transformation,
    create_polynomial_features as create_poly_features,
    handle_skewness
)

# Import from selection module
from .selection import (
    select_features_correlation,
    select_features_mutual_info,
    select_features_rfe
)

__all__ = [
    # Engineering functions
    'create_interaction_features',
    'create_polynomial_features',
    'create_ratio_features',
    'create_log_features',
    'create_binning_features',
    'create_statistical_features',
    
    # Transformation classes and functions
    'StandardScalerWrapper',
    'MinMaxScalerWrapper',
    'RobustScalerWrapper',
    'apply_power_transformation',
    'apply_quantile_transformation',
    'create_poly_features',
    'handle_skewness',
    
    # Selection functions
    'select_features_correlation',
    'select_features_mutual_info',
    'select_features_rfe'
]
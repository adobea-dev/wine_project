"""
Data Preprocessor Module
Handles data cleaning, preprocessing, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


# =====================================================
#  1. CLEANING
# =====================================================
def clean_data(df: pd.DataFrame, 
               handle_missing: str = 'median',
               remove_outliers: bool = False,
               outlier_method: str = 'iqr') -> pd.DataFrame:
    """Clean the Wine Quality dataset."""
    logger.info("Starting data cleaning process...")
    df_clean = df.copy()

    # Handle missing values
    missing_before = df_clean.isnull().sum().sum()
    logger.info(f"Missing values before cleaning: {missing_before}")

    if missing_before > 0:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if handle_missing == 'drop':
            df_clean = df_clean.dropna()
        elif handle_missing == 'median':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif handle_missing == 'mean':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        logger.info(f"Missing values handled using method: {handle_missing}")

    # Handle categorical missing values
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_value)
            logger.info(f"Filled missing values in {col} with mode: {mode_value}")

    # Remove outliers if requested
    if remove_outliers:
        outliers_removed = 0
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if outlier_method == 'iqr':
            for col in numeric_cols:
                if col != 'quality':
                    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    before = len(df_clean)
                    df_clean = df_clean[
                        (df_clean[col] >= Q1 - 1.5 * IQR) & (df_clean[col] <= Q3 + 1.5 * IQR)
                    ]
                    outliers_removed += before - len(df_clean)
            logger.info(f"Removed {outliers_removed} outlier rows using IQR method")

    # Encode 'type' column if present
    if 'type' in df_clean.columns:
        le = LabelEncoder()
        df_clean['type_encoded'] = le.fit_transform(df_clean['type'])
        logger.info(f"Encoded 'type' column: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    logger.info(f"Final dataset shape after cleaning: {df_clean.shape}")
    return df_clean


# =====================================================
#  2. LABEL CREATION
# =====================================================
def create_quality_categories(df: pd.DataFrame, 
                              method: str = 'multi', 
                              threshold: int = 6) -> pd.DataFrame:
    """Create quality categories from numeric quality scores."""
    df_cat = df.copy()

    if method == 'binary':
        df_cat['quality_category'] = (df_cat['quality'] >= threshold).astype(int)
        df_cat['quality_label'] = df_cat['quality_category'].map({0: 'Bad', 1: 'Good'})
        logger.info(f"Created binary categories with threshold {threshold}")

    elif method == 'multi':
        def categorize_quality(score):
            if score <= 4: return 0
            elif score <= 6: return 1
            else: return 2
        df_cat['quality_category'] = df_cat['quality'].apply(categorize_quality)
        df_cat['quality_label'] = df_cat['quality_category'].map({0: 'Low', 1: 'Medium', 2: 'High'})
        logger.info("Created multi-class categories: Low / Medium / High")

    elif method == 'custom':
        def categorize_quality_custom(score):
            if score == 3: return 0
            elif score == 4: return 1
            elif score in [5, 6]: return 2
            elif score in [7, 8]: return 3
            else: return 4
        df_cat['quality_category'] = df_cat['quality'].apply(categorize_quality_custom)
        df_cat['quality_label'] = df_cat['quality_category'].map({
            0: 'Very Low', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'
        })
        logger.info("Created custom 5-class categories")

    return df_cat


# =====================================================
#  3. SPLITTING
# =====================================================
def split_data(df: pd.DataFrame,
               target_column: str = 'quality_category',
               test_size: float = 0.2,
               val_size: float = 0.2,
               random_state: int = 42,
               stratify: bool = True
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into train, validation, and test sets."""
    logger.info("Splitting data into train/validation/test sets...")

    # Separate features & target (avoid label leakage)
    exclude_cols = [target_column, 'quality', 'quality_label']
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    y = df[target_column]

    stratify_param = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    val_size_adjusted = val_size / (1 - test_size)
    stratify_param = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted,
        random_state=random_state, stratify=stratify_param
    )

    logger.info(f"Data split completed:")
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Validation: {X_val.shape[0]} samples")
    logger.info(f"  Test: {X_test.shape[0]} samples")

    # ✅ Safe overlap check
    try:
        overlap = set(X_train.index).intersection(set(X_test.index))
        if overlap:
            logger.warning(f"⚠️ Overlap detected between train/test indices: {len(overlap)} samples")
        else:
            logger.info("✅ No overlap between train/test splits.")
    except Exception as e:
        logger.warning(f"Could not compute overlap check: {e}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =====================================================
# 📈 4. CORRELATION
# =====================================================
def get_feature_importance_correlation(df: pd.DataFrame,
                                       target_column: str = 'quality_category') -> pd.DataFrame:
    """Compute absolute correlation between numeric features and target."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    corr = df[numeric_cols + [target_column]].corr()[target_column].abs().sort_values(ascending=False)
    return corr.drop(target_column)

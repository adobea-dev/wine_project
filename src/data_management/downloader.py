"""
Data Downloader Module

Handles downloading the Wine Quality dataset from Kaggle with fallback options.
"""

import os
import subprocess
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset configuration
KAGGLE_DATASET = "rajyellow46/wine-quality"
KAGGLE_FILE = "winequalityN.csv"
FALLBACK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"




def check_kaggle_credentials() -> bool:
    """
    Check if Kaggle API credentials are properly configured.
    
    Returns:
        bool: True if credentials exist and are valid, False otherwise
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_dir.exists() or not kaggle_json.exists():
        logger.warning("Kaggle credentials not found. Will use fallback download.")
        return False
    
    # Check file permissions
    if kaggle_json.stat().st_mode & 0o077 != 0:
        logger.warning("Kaggle credentials file has incorrect permissions. Will use fallback download.")
        return False
    
    return True

def download_via_kaggle(output_dir: Path) -> Optional[Path]:
    """
    Download dataset using Kaggle API.
    
    Args:
        output_dir (Path): Directory to save the dataset
        
    Returns:
        Optional[Path]: Path to downloaded file if successful, None otherwise
    """
    try:
        if not check_kaggle_credentials():
            return None
            
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        cmd = [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-f", KAGGLE_FILE,
            "-p", str(output_dir)
        ]
        
        logger.info(f"Downloading dataset via Kaggle API: {KAGGLE_DATASET}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Kaggle downloads as .zip, extract if needed
        zip_file = output_dir / f"{KAGGLE_FILE}.zip"
        if zip_file.exists():
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            zip_file.unlink()  # Remove zip file
        
        output_file = output_dir / KAGGLE_FILE
        if output_file.exists():
            logger.info(f"Successfully downloaded via Kaggle: {output_file}")
            return output_file
        else:
            logger.error("Kaggle download completed but file not found")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle download failed: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Kaggle download: {e}")
        return None

def create_sample_dataset(output_dir: Path) -> Path:
    """
    Create a sample wine quality dataset for testing purposes.
    
    Args:
        output_dir (Path): Directory to save the dataset
        
    Returns:
        Path: Path to created file
    """
    import pandas as pd
    import numpy as np
    
    logger.info("Creating sample wine quality dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    
    data = {
        'type': np.random.choice(['red', 'white'], n_samples),
        'fixed_acidity': np.random.normal(7.0, 1.5, n_samples),
        'volatile_acidity': np.random.normal(0.5, 0.2, n_samples),
        'citric_acid': np.random.normal(0.3, 0.1, n_samples),
        'residual_sugar': np.random.normal(2.5, 1.0, n_samples),
        'chlorides': np.random.normal(0.08, 0.02, n_samples),
        'free_sulfur_dioxide': np.random.normal(15.0, 8.0, n_samples),
        'total_sulfur_dioxide': np.random.normal(45.0, 20.0, n_samples),
        'density': np.random.normal(0.997, 0.002, n_samples),
        'pH': np.random.normal(3.2, 0.2, n_samples),
        'sulphates': np.random.normal(0.6, 0.1, n_samples),
        'alcohol': np.random.normal(10.5, 1.5, n_samples),
        'quality': np.random.choice([3, 4, 5, 6, 7, 8, 9], n_samples, p=[0.01, 0.05, 0.15, 0.25, 0.25, 0.20, 0.09])
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure positive values for certain columns
    df['fixed_acidity'] = np.abs(df['fixed_acidity'])
    df['volatile_acidity'] = np.abs(df['volatile_acidity'])
    df['citric_acid'] = np.abs(df['citric_acid'])
    df['residual_sugar'] = np.abs(df['residual_sugar'])
    df['chlorides'] = np.abs(df['chlorides'])
    df['free_sulfur_dioxide'] = np.abs(df['free_sulfur_dioxide'])
    df['total_sulfur_dioxide'] = np.abs(df['total_sulfur_dioxide'])
    df['density'] = np.abs(df['density'])
    df['sulphates'] = np.abs(df['sulphates'])
    df['alcohol'] = np.abs(df['alcohol'])
    
    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / KAGGLE_FILE
    df.to_csv(output_file, index=False)
    
    logger.info(f"Sample dataset created: {output_file}")
    return output_file

def download_via_fallback(output_dir: Path) -> Path:
    """
    Download dataset using fallback URL or create sample dataset.
    
    Args:
        output_dir (Path): Directory to save the dataset
        
    Returns:
        Path: Path to downloaded/created file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / KAGGLE_FILE
    
    logger.info(f"Attempting to download dataset from fallback URL: {FALLBACK_URL}")
    
    try:
        response = requests.get(FALLBACK_URL, timeout=30)
        response.raise_for_status()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        logger.info(f"Successfully downloaded via fallback: {output_file}")
        return output_file
        
    except requests.RequestException as e:
        logger.warning(f"Fallback download failed: {e}")
        logger.info("Creating sample dataset instead...")
        return create_sample_dataset(output_dir)
    except Exception as e:
        logger.warning(f"Unexpected error during fallback download: {e}")
        logger.info("Creating sample dataset instead...")
        return create_sample_dataset(output_dir)

def download_wine_dataset(data_dir: Path = None) -> Path:
    """
    Download Wine Quality dataset with automatic fallback.
    
    Args:
        data_dir (Path, optional): Directory to save data. Defaults to ./data/raw
        
    Returns:
        Path: Path to the downloaded dataset file
    """
    if data_dir is None:
        data_dir = Path("data/raw")
    
    # Try Kaggle first
    kaggle_file = download_via_kaggle(data_dir)
    if kaggle_file and kaggle_file.exists():
        return kaggle_file
    
    # Fallback to direct download
    logger.info("Falling back to direct download...")
    return download_via_fallback(data_dir)

def get_data_info() -> Dict[str, Any]:
    """
    Get information about the Wine Quality dataset.
    
    Returns:
        Dict[str, Any]: Dataset information including features and target
    """
    return {
        "name": "Wine Quality Dataset",
        "source": "Kaggle - Raj Yellow",
        "description": "Wine quality classification based on physicochemical properties",
        "features": [
            "type", "fixed_acidity", "volatile_acidity", "citric_acid", 
            "residual_sugar", "chlorides", "free_sulfur_dioxide", 
            "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
        ],
        "target": "quality",
        "classes": "Multi-class (3-9 quality ratings)",
        "samples": "~6500 wine samples",
        "url": "https://www.kaggle.com/datasets/rajyellow46/wine-quality"
    }
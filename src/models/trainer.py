"""
Model Training Module

Handles training of various machine learning models for wine quality classification.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A comprehensive model trainer for wine quality classification.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.training_results = {}
        
    def get_default_models(self) -> Dict[str, Any]:
        """
        Get a dictionary of default models for wine quality multi-classification.
        
        Returns:
            Dict[str, Any]: Dictionary of model names and instances
        """
        return {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                #solver='liblinear',
                multi_class='multinomial',  #for our mult-classification
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=200
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced', 
                kernel='rbf'
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'ada_boost': AdaBoostClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'ridge': RidgeClassifier(
                random_state=self.random_state
            )
        }
    
    def train_single_model(self, 
                          model_name: str,
                          model: Any,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: pd.DataFrame = None,
                          y_val: pd.Series = None,
                          use_scaling: bool = True) -> Dict[str, Any]:
        """
        Train a single model and return results.
        
        Args:
            model_name (str): Name of the model
            model: Model instance to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation target
            use_scaling (bool): Whether to use feature scaling
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info(f"Training {model_name}...")
        
        # Create pipeline with scaling if needed
        if use_scaling and not isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, 
                                                DecisionTreeClassifier, GaussianNB)):
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
        else:
            pipeline = Pipeline([
                ('classifier', model)
            ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Store the trained model
        self.models[model_name] = pipeline
        
        # Calculate training score
        train_score = pipeline.score(X_train, y_train)
        
        # Calculate validation score if validation data provided
        val_score = None
        if X_val is not None and y_val is not None:
            val_score = pipeline.score(X_val, y_val)
        
        # Store results
        results = {
            'model_name': model_name,
            'model': pipeline,
            'train_score': train_score,
            'val_score': val_score,
            'training_completed': True
        }
        
        self.training_results[model_name] = results
        
        val_score_str = f"{val_score:.4f}" if val_score is not None else "N/A"
        logger.info(f"{model_name} training completed. Train score: {train_score:.4f}, Val score: {val_score_str}")
        
        return results
    
    def train_all_models(self, 
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame = None,
                        y_val: pd.Series = None,
                        models: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
        """
        Train all models and return results.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation target
            models (Dict[str, Any], optional): Custom models dictionary
            
        Returns:
            Dict[str, Dict[str, Any]]: Results for all models
        """
        if models is None:
            models = self.get_default_models()
        
        logger.info(f"Training {len(models)} models...")
        
        all_results = {}
        for model_name, model in models.items():
            try:
                results = self.train_single_model(
                    model_name, model, X_train, y_train, X_val, y_val
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                all_results[model_name] = {
                    'model_name': model_name,
                    'error': str(e),
                    'training_completed': False
                }
        
        return all_results
    
    def hyperparameter_tuning(self, 
                             model_name: str,
                             model: Any,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             param_grid: Dict[str, List],
                             cv: int = 5,
                             search_type: str = 'grid',
                             n_iter: int = 100) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a model.
        
        Args:
            model_name (str): Name of the model
            model: Model instance
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            param_grid (Dict[str, List]): Parameter grid for tuning
            cv (int): Number of cross-validation folds
            search_type (str): Type of search ('grid' or 'random')
            n_iter (int): Number of iterations for random search
            
        Returns:
            Dict[str, Any]: Tuning results
        """
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Update parameter grid for pipeline
        param_grid_pipeline = {}
        for param, values in param_grid.items():
            param_grid_pipeline[f'classifier__{param}'] = values
        
        # Choose search method
        if search_type == 'grid':
            search = GridSearchCV(
                pipeline, param_grid_pipeline, cv=cv, 
                scoring='accuracy', n_jobs=-1, random_state=self.random_state
            )
        else:
            search = RandomizedSearchCV(
                pipeline, param_grid_pipeline, cv=cv, 
                scoring='accuracy', n_jobs=-1, random_state=self.random_state,
                n_iter=n_iter
            )
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Store best model
        self.best_models[model_name] = search.best_estimator_
        
        # Return results
        results = {
            'model_name': model_name,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_model': search.best_estimator_,
            'cv_results': search.cv_results_
        }
        
        logger.info(f"Hyperparameter tuning completed for {model_name}. Best score: {search.best_score_:.4f}")
        
        return results
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
        elif model_name in self.best_models:
            joblib.dump(self.best_models[model_name], filepath)
            logger.info(f"Best model {model_name} saved to {filepath}")
        else:
            logger.error(f"Model {model_name} not found for saving")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load a model from disk.
        
        Args:
            model_name (str): Name to assign to the loaded model
            filepath (str): Path to the model file
        """
        try:
            model = joblib.load(filepath)
            self.models[model_name] = model
            logger.info(f"Model {model_name} loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")

def train_multiple_models(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_val: pd.DataFrame = None,
                         y_val: pd.Series = None,
                         models: Dict[str, Any] = None,
                         random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to train multiple models.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame, optional): Validation features
        y_val (pd.Series, optional): Validation target
        models (Dict[str, Any], optional): Custom models dictionary
        random_state (int): Random seed
        
    Returns:
        Dict[str, Dict[str, Any]]: Results for all models
    """
    trainer = ModelTrainer(random_state=random_state)
    return trainer.train_all_models(X_train, y_train, X_val, y_val, models)

def get_hyperparameter_grids() -> Dict[str, Dict[str, List]]:
    """
    Get default hyperparameter grids for common models.
    
    Returns:
        Dict[str, Dict[str, List]]: Parameter grids for different models
    """
    return {
        'logistic_regression': {
            'C': [0.1, 1, 3, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'random_forest': {
            'n_estimators': [200, 400, 600],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'svm': {
            'C': [0.5, 1, 3, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    }
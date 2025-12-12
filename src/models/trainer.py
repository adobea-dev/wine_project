"""
Model Training Module

Handles training of various machine learning models for wine quality classification.

"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A comprehensive model trainer for wine quality classification.

    Model registry:
    - self.models[model_name] stores the trained sklearn Pipeline for that model.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.best_models: Dict[str, Any] = {}
        self.training_results: Dict[str, Dict[str, Any]] = {}

    def get_default_models(self) -> Dict[str, Any]:
        """Default models for our 3-class classification."""
        return {
            "logistic_regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=5000,  # helps convergence
                solver="lbfgs",
                multi_class="multinomial",
                class_weight="balanced",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "gradient_boosting": GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=200,
            ),
            "svm": SVC(
                random_state=self.random_state,
                probability=True,
                class_weight="balanced",
                kernel="rbf",
            ),
            "knn": KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1,
            ),
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(
                random_state=self.random_state
            ),
            "ada_boost": AdaBoostClassifier(
                random_state=self.random_state,
                n_estimators=100,
            ),
            "ridge": RidgeClassifier(
                random_state=self.random_state
            ),
        }

    def list_registry(self) -> List[str]:
        """Convenience: what models are currently stored in the registry."""
        return sorted(list(self.models.keys()))

    def train_single_model(
        self,
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train a single model and return results.

        IMPORTANT:
        - No scaling happens here. X_train / X_val must already be scaled in main.py.
        """
        logger.info(f"Training {model_name}...")

        pipeline = Pipeline([
            ("classifier", model),
        ])

        pipeline.fit(X_train, y_train)

        # Registry: store trained pipeline
        self.models[model_name] = pipeline

        train_score = pipeline.score(X_train, y_train)
        val_score = pipeline.score(X_val, y_val) if (X_val is not None and y_val is not None) else None

        results = {
            "model_name": model_name,
            "model": pipeline,
            "train_score": train_score,
            "val_score": val_score,
            "training_completed": True,
        }

        self.training_results[model_name] = results

        val_score_str = f"{val_score:.4f}" if val_score is not None else "N/A"
        logger.info(f"{model_name} training completed. Train score: {train_score:.4f}, Val score: {val_score_str}")

        return results

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        models: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Train all models and return results."""
        if models is None:
            models = self.get_default_models()

        logger.info(f"Training {len(models)} models...")

        all_results: Dict[str, Dict[str, Any]] = {}
        for model_name, model in models.items():
            try:
                res = self.train_single_model(model_name, model, X_train, y_train, X_val, y_val)
                all_results[model_name] = res
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                all_results[model_name] = {
                    "model_name": model_name,
                    "error": str(e),
                    "training_completed": False,
                }

        return all_results

    def hyperparameter_tuning(
        self,
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, List],
        cv: int = 5,
        search_type: str = "grid",
        n_iter: int = 100,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Hyperparameter tuning.

        Recommendation A aligned:
        - We do NOT include a scaler here because main.py already scaled X_train.
        - If you later decide to tune on raw features, then add scaling here intentionally.
        """
        logger.info(f"Performing hyperparameter tuning for {model_name}...")

        pipeline = Pipeline([
            ("classifier", model),
        ])

        param_grid_pipeline = {f"classifier__{k}": v for k, v in param_grid.items()}

        if search_type == "grid":
            search = GridSearchCV(
                pipeline,
                param_grid_pipeline,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
        else:
            search = RandomizedSearchCV(
                pipeline,
                param_grid_pipeline,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                n_iter=n_iter,
                random_state=self.random_state,
            )

        search.fit(X_train, y_train)
        self.best_models[model_name] = search.best_estimator_

        results = {
            "model_name": model_name,
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "best_model": search.best_estimator_,
            "cv_results": search.cv_results_,
        }

        logger.info(f"Hyperparameter tuning completed for {model_name}. Best score: {search.best_score_:.4f}")
        return results

    def save_model(self, model_name: str, filepath: str) -> None:
        """Save a trained model to disk."""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
        elif model_name in self.best_models:
            joblib.dump(self.best_models[model_name], filepath)
            logger.info(f"Best model {model_name} saved to {filepath}")
        else:
            logger.error(f"Model {model_name} not found for saving")

    def load_model(self, model_name: str, filepath: str) -> None:
        """Load a model from disk into the registry."""
        try:
            model = joblib.load(filepath)
            self.models[model_name] = model
            logger.info(f"Model {model_name} loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")


def train_multiple_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    models: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Convenience function to train multiple models."""
    trainer = ModelTrainer(random_state=random_state)
    return trainer.train_all_models(X_train, y_train, X_val, y_val, models)


def get_hyperparameter_grids() -> Dict[str, Dict[str, List]]:
    """Default hyperparameter grids for common models."""
    return {
        "logistic_regression": {
            "C": [0.1, 1, 3, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        },
        "random_forest": {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "gradient_boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 0.9, 1.0],
        },
        "svm": {
            "C": [0.5, 1, 3, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        },
    }

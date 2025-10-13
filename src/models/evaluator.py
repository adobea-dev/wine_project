"""
Model Evaluation Module

Handles comprehensive model evaluation and performance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A comprehensive model evaluator for wine quality classification.
    """
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_classification_performance(self, 
                                          y_true: pd.Series,
                                          y_pred: pd.Series,
                                          y_pred_proba: np.ndarray = None,
                                          model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate classification performance with comprehensive metrics.
        
        Args:
            y_true (pd.Series): True labels
            y_pred (pd.Series): Predicted labels
            y_pred_proba (np.ndarray, optional): Predicted probabilities
            model_name (str): Name of the model for identification
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        logger.info(f"Evaluating {model_name} performance...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Macro averages
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            try:
                # For multi-class, use one-vs-rest approach
                if len(np.unique(y_true)) > 2:
                    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    avg_precision = average_precision_score(y_true, y_pred_proba, average='weighted')
                else:
                    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
                
                results.update({
                    'roc_auc': roc_auc,
                    'average_precision': avg_precision
                })
            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {e}")
        
        # Store results
        self.evaluation_results[model_name] = results
        
        logger.info(f"{model_name} evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def cross_validate_model(self, 
                           model: Any,
                           X: pd.DataFrame,
                           y: pd.Series,
                           cv: int = 5,
                           scoring: List[str] = None) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to evaluate
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of cross-validation folds
            scoring (List[str], optional): Scoring metrics
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        if scoring is None:
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        cv_results = {}
        for metric in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max())
            }
        
        logger.info("Cross-validation completed")
        return cv_results
    
    def plot_confusion_matrix(self, 
                            y_true: pd.Series,
                            y_pred: pd.Series,
                            model_name: str = "Model",
                            save_path: str = None) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            y_true (pd.Series): True labels
            y_pred (pd.Series): Predicted labels
            model_name (str): Name of the model
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class labels
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, 
                      y_true: pd.Series,
                      y_pred_proba: np.ndarray,
                      model_name: str = "Model",
                      save_path: str = None) -> None:
        """
        Plot ROC curve for multi classification.
        
        Args:
            y_true (pd.Series): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # For multi-class, plot one-vs-rest
        if len(np.unique(y_true)) > 2:
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            
            # Binarize the output
            y_bin = label_binarize(y_true, classes=sorted(np.unique(y_true)))
            n_classes = y_bin.shape[1]
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot all ROC curves
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        else:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curve(self, 
                          model: Any,
                          X: pd.DataFrame,
                          y: pd.Series,
                          model_name: str = "Model",
                          cv: int = 5,
                          save_path: str = None) -> None:
        """
        Plot learning curve for a model.
        
        Args:
            model: Model to evaluate
            X (pd.DataFrame): Features
            y (pd.Series): Target
            model_name (str): Name of the model
            cv (int): Number of cross-validation folds
            save_path (str, optional): Path to save the plot
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.figure(figsize=(10, 6))
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curve saved to {save_path}")
        
        plt.show()
    
    def get_feature_importance(self, 
                             model: Any,
                             feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from a model.
        
        Args:
            model: Trained model
            feature_names (List[str]): Names of features
            
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        # Handle different model types
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            coef = model.coef_
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, 
                              model: Any,
                              feature_names: List[str],
                              model_name: str = "Model",
                              top_n: int = 15,
                              save_path: str = None) -> None:
        """
        Plot feature importance.
        
        Args:
            model: Trained model
            feature_names (List[str]): Names of features
            model_name (str): Name of the model
            top_n (int): Number of top features to show
            save_path (str, optional): Path to save the plot
        """
        importance_df = self.get_feature_importance(model, feature_names)
        
        if importance_df.empty:
            logger.warning("Cannot plot feature importance for this model")
            return
        
        # Select top features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()

def evaluate_model_performance(y_true: pd.Series,
                             y_pred: pd.Series,
                             y_pred_proba: np.ndarray = None,
                             model_name: str = "Model") -> Dict[str, Any]:
    """
    Convenience function to evaluate model performance.
    
    Args:
        y_true (pd.Series): True labels
        y_pred (pd.Series): Predicted labels
        y_pred_proba (np.ndarray, optional): Predicted probabilities
        model_name (str): Name of the model
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_classification_performance(y_true, y_pred, y_pred_proba, model_name)
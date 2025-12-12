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
from src.evaluation.ordinal_metrics import ordinal_mae, severe_error_rate, within_one_accuracy
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

        # Ensure arrays (avoids pandas index weirdness)
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        # Basic metrics
        accuracy = accuracy_score(y_true_arr, y_pred_arr)
        precision = precision_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
        recall = recall_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
        f1 = f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
        
        # Macro averages
        precision_macro = precision_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_arr, y_pred_arr)
        
        # Classification report
        class_report = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)

        # Ordinal metrics (assumes ordered labels like Low=0, Medium=1, High=2)
        try:
            ord_mae = ordinal_mae(y_true_arr, y_pred_arr)
            sev_rate = severe_error_rate(y_true_arr, y_pred_arr)
            within1 = within_one_accuracy(y_true_arr, y_pred_arr)
        except Exception as e:
            logger.warning(f"Could not calculate ordinal metrics: {e}")
            ord_mae, sev_rate, within1 = None, None, None
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,

            # Ordinal-aware metrics
            'ordinal_mae': ord_mae,
            'severe_error_rate': sev_rate,
            'within_1_accuracy': within1,

            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            try:
                # For multi-class, use one-vs-rest approach
                if len(np.unique(y_true_arr)) > 2:
                    roc_auc = roc_auc_score(y_true_arr, y_pred_proba, multi_class='ovr', average='weighted')
                    avg_precision = average_precision_score(y_true_arr, y_pred_proba, average='weighted')
                else:
                    roc_auc = roc_auc_score(y_true_arr, y_pred_proba[:, 1])
                    avg_precision = average_precision_score(y_true_arr, y_pred_proba[:, 1])
                
                results.update({
                    'roc_auc': roc_auc,
                    'average_precision': avg_precision
                })
            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {e}")
        
        # Store results
        self.evaluation_results[model_name] = results
        
        logger.info(
            f"{model_name} evaluation completed. "
            f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Ordinal_MAE: {ord_mae if ord_mae is not None else 'NA'}"
        )
        
        return results

    # ---- everything below stays exactly as you had it ----
    
    def cross_validate_model(self, 
                           model: Any,
                           X: pd.DataFrame,
                           y: pd.Series,
                           cv: int = 5,
                           scoring: List[str] = None) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
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
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
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
        """
        plt.figure(figsize=(8, 6))
        
        if len(np.unique(y_true)) > 2:
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            
            y_bin = label_binarize(y_true, classes=sorted(np.unique(y_true)))
            n_classes = y_bin.shape[1]
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        else:
            from sklearn.metrics import auc
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
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.figure(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
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
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
        
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
        """
        importance_df = self.get_feature_importance(model, feature_names)
        
        if importance_df.empty:
            logger.warning("Cannot plot feature importance for this model")
            return
        
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
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_classification_performance(y_true, y_pred, y_pred_proba, model_name)

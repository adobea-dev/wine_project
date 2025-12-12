"""
Model Visualization Module - Creates model evaluation plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

def plot_model_comparison(evaluation_results: Dict[str, Dict[str, Any]],
                         metrics: List[str] = None,
                         save_path: str = None) -> None:
    """Plot comparison of models across multiple metrics."""
    if metrics is None:
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    logger.info(f"Plotting model comparison for metrics: {metrics}")
    
    # Prepare data for plotting
    plot_data = []
    for model_name, results in evaluation_results.items():
        if 'error' not in results:
            for metric in metrics:
                if metric in results:
                    plot_data.append({
                        'model_name': model_name,
                        'metric': metric,
                        'score': results[metric]
                    })
    
    if not plot_data:
        logger.warning("No valid data for plotting")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            metric_data = plot_df[plot_df['metric'] == metric]
            if not metric_data.empty:
                sns.barplot(data=metric_data, x='model_name', y='score', ax=axes[i])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Model')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true: pd.Series,
                         y_pred: pd.Series,
                         model_name: str = "Model",
                         save_path: str = None) -> None:
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
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

def plot_roc_curve(y_true: pd.Series,
                  y_pred_proba: np.ndarray,
                  model_name: str = "Model",
                  save_path: str = None) -> None:
    """Plot ROC curve for binary classification."""
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(8, 6))
    
    # For multi-class, plot one-vs-rest
    if len(np.unique(y_true)) > 2:
        from sklearn.preprocessing import label_binarize
        
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
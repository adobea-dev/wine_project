"""
Model Comparison Module

Handles comparison of multiple models and selection of the best performing model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ModelComparator:
    """
    A comprehensive model comparator for wine quality classification.
    """
    
    def __init__(self):
        self.comparison_results = {}
        self.best_model = None
        self.ranking = None
    
    def compare_models(self, 
                      evaluation_results: Dict[str, Dict[str, Any]],
                      metric: str = 'f1_weighted',
                      ascending: bool = False) -> pd.DataFrame:
        """
        Compare multiple models based on evaluation results.
        
        Args:
            evaluation_results (Dict[str, Dict[str, Any]]): Results from model evaluation
            metric (str): Primary metric for comparison
            ascending (bool): Whether to sort in ascending order
            
        Returns:
            pd.DataFrame: Comparison results DataFrame
        """
        logger.info(f"Comparing models using metric: {metric}")
        
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            if 'error' in results:
                # Handle models with errors
                comparison_data.append({
                    'model_name': model_name,
                    'accuracy': np.nan,
                    'precision_weighted': np.nan,
                    'recall_weighted': np.nan,
                    'f1_weighted': np.nan,
                    'f1_macro': np.nan,
                    'roc_auc': np.nan if 'roc_auc' not in results else results['roc_auc'],
                    'error': results['error']
                })
            else:
                comparison_data.append({
                    'model_name': model_name,
                    'accuracy': results.get('accuracy', np.nan),
                    'precision_weighted': results.get('precision_weighted', np.nan),
                    'recall_weighted': results.get('recall_weighted', np.nan),
                    'f1_weighted': results.get('f1_weighted', np.nan),
                    'f1_macro': results.get('f1_macro', np.nan),
                    'roc_auc': results.get('roc_auc', np.nan),
                    'error': None
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric
        if metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(metric, ascending=ascending)
        
        # Store results
        self.comparison_results = comparison_df
        self.ranking = comparison_df['model_name'].tolist()
        
        # Identify best model
        if not comparison_df.empty and not comparison_df[metric].isna().all():
            self.best_model = comparison_df.iloc[0]['model_name']
            logger.info(f"Best model identified: {self.best_model}")
        
        logger.info("Model comparison completed")
        return comparison_df
    
    def plot_model_comparison(self, 
                            evaluation_results: Dict[str, Dict[str, Any]],
                            metrics: List[str] = None,
                            save_path: str = None) -> None:
        """
        Plot comparison of models across multiple metrics.
        
        Args:
            evaluation_results (Dict[str, Dict[str, Any]]): Results from model evaluation
            metrics (List[str], optional): Metrics to plot
            save_path (str, optional): Path to save the plot
        """
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
    
    def plot_metric_heatmap(self, 
                          evaluation_results: Dict[str, Dict[str, Any]],
                          metrics: List[str] = None,
                          save_path: str = None) -> None:
        """
        Plot heatmap of model performance across metrics.
        
        Args:
            evaluation_results (Dict[str, Dict[str, Any]]): Results from model evaluation
            metrics (List[str], optional): Metrics to include
            save_path (str, optional): Path to save the plot
        """
        if metrics is None:
            metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'f1_macro']
        
        logger.info("Creating metric heatmap...")
        
        # Prepare data for heatmap
        heatmap_data = []
        for model_name, results in evaluation_results.items():
            if 'error' not in results:
                row = {'model_name': model_name}
                for metric in metrics:
                    row[metric] = results.get(metric, np.nan)
                heatmap_data.append(row)
        
        if not heatmap_data:
            logger.warning("No valid data for heatmap")
            return
        
        heatmap_df = pd.DataFrame(heatmap_data).set_index('model_name')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
        plt.title('Model Performance Heatmap')
        plt.xlabel('Metrics')
        plt.ylabel('Models')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metric heatmap saved to {save_path}")
        
        plt.show()
    
    def get_model_ranking(self, 
                         evaluation_results: Dict[str, Dict[str, Any]],
                         metrics: List[str] = None,
                         weights: List[float] = None) -> pd.DataFrame:
        """
        Get comprehensive model ranking based on multiple metrics.
        
        Args:
            evaluation_results (Dict[str, Dict[str, Any]]): Results from model evaluation
            metrics (List[str], optional): Metrics to consider
            weights (List[float], optional): Weights for each metric
            
        Returns:
            pd.DataFrame: Model ranking with composite scores
        """
        if metrics is None:
            metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        if weights is None:
            weights = [1.0] * len(metrics)
        
        if len(weights) != len(metrics):
            weights = [1.0] * len(metrics)
            logger.warning("Weights length doesn't match metrics, using equal weights")
        
        logger.info(f"Creating model ranking with metrics: {metrics}")
        
        ranking_data = []
        
        for model_name, results in evaluation_results.items():
            if 'error' not in results:
                # Calculate composite score
                scores = []
                for metric in metrics:
                    if metric in results and not pd.isna(results[metric]):
                        scores.append(results[metric])
                    else:
                        scores.append(0.0)  # Default score for missing metrics
                
                # Weighted average
                composite_score = np.average(scores, weights=weights)
                
                ranking_data.append({
                    'model_name': model_name,
                    'composite_score': composite_score,
                    **{metric: results.get(metric, np.nan) for metric in metrics}
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('composite_score', ascending=False)
        
        logger.info("Model ranking completed")
        return ranking_df
    
    def get_best_model_info(self) -> Dict[str, Any]:
        """
        Get information about the best performing model.
        
        Returns:
            Dict[str, Any]: Best model information
        """
        if self.best_model is None:
            logger.warning("No best model identified")
            return {}
        
        if self.comparison_results.empty:
            logger.warning("No comparison results available")
            return {}
        
        best_model_info = self.comparison_results[
            self.comparison_results['model_name'] == self.best_model
        ].iloc[0].to_dict()
        
        return best_model_info
    
    def export_comparison_results(self, filepath: str) -> None:
        """
        Export comparison results to CSV.
        
        Args:
            filepath (str): Path to save the results
        """
        if self.comparison_results.empty:
            logger.warning("No comparison results to export")
            return
        
        self.comparison_results.to_csv(filepath, index=False)
        logger.info(f"Comparison results exported to {filepath}")

def compare_models(evaluation_results: Dict[str, Dict[str, Any]],
                  metric: str = 'f1_weighted') -> pd.DataFrame:
    """
    Convenience function to compare models.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): Results from model evaluation
        metric (str): Primary metric for comparison
        
    Returns:
        pd.DataFrame: Comparison results
    """
    comparator = ModelComparator()
    return comparator.compare_models(evaluation_results, metric)
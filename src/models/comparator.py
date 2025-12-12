"""
Model Comparison Module

Handles comparison of multiple models and selection of the best performing model.
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    A comprehensive model comparator for wine quality classification.
    """

    def __init__(self):
        self.comparison_results = pd.DataFrame()
        self.best_model: Optional[str] = None
        self.ranking: Optional[List[str]] = None

    def compare_models(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        metric: str = "f1_weighted",
        ascending: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models based on evaluation results.

        Args:
            evaluation_results: Results from model evaluation
            metric: Primary metric for comparison
            ascending: If None, will auto-select based on metric type.
                       Lower is better for error metrics like ordinal_mae.
        """
        logger.info(f"Comparing models using metric: {metric}")

        # Metrics where lower is better
        lower_is_better = {
            "ordinal_mae",
            "severe_error_rate",
        }

        # Auto choose sort direction if not provided
        if ascending is None:
            ascending = metric in lower_is_better

        comparison_data = []

        for model_name, results in evaluation_results.items():
            if "error" in results:
                comparison_data.append({
                    "model_name": model_name,
                    "accuracy": np.nan,
                    "precision_weighted": np.nan,
                    "recall_weighted": np.nan,
                    "f1_weighted": np.nan,
                    "f1_macro": np.nan,
                    "roc_auc": results.get("roc_auc", np.nan),

                    # Ordinal metrics (if your evaluator adds them)
                    "ordinal_mae": np.nan,
                    "severe_error_rate": np.nan,
                    "within_1_accuracy": np.nan,

                    "error": results["error"],
                })
            else:
                comparison_data.append({
                    "model_name": model_name,
                    "accuracy": results.get("accuracy", np.nan),
                    "precision_weighted": results.get("precision_weighted", np.nan),
                    "recall_weighted": results.get("recall_weighted", np.nan),
                    "f1_weighted": results.get("f1_weighted", np.nan),
                    "f1_macro": results.get("f1_macro", np.nan),
                    "roc_auc": results.get("roc_auc", np.nan),

                    # Ordinal metrics
                    "ordinal_mae": results.get("ordinal_mae", np.nan),
                    "severe_error_rate": results.get("severe_error_rate", np.nan),
                    "within_1_accuracy": results.get("within_1_accuracy", np.nan),

                    "error": None,
                })

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by chosen metric (if present)
        if metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
        else:
            logger.warning(f"Metric '{metric}' not found in comparison table. Returning unsorted results.")

        self.comparison_results = comparison_df
        self.ranking = comparison_df["model_name"].tolist()

        # Best model is first after sorting
        if (not comparison_df.empty) and (metric in comparison_df.columns) and (not comparison_df[metric].isna().all()):
            self.best_model = comparison_df.iloc[0]["model_name"]
            logger.info(f"Best model identified: {self.best_model}")
        else:
            self.best_model = None
            logger.warning("No best model identified (empty results or metric missing).")

        logger.info("Model comparison completed")
        return comparison_df

    def plot_model_comparison(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of models across multiple metrics.

        Args:
            evaluation_results: Results from model evaluation
            metrics: Metrics to plot
            save_path: Path to save the plot
        """
        if metrics is None:
            metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

        logger.info(f"Plotting model comparison for metrics: {metrics}")

        plot_data = []
        for model_name, results in evaluation_results.items():
            if "error" in results:
                continue
            for m in metrics:
                if m in results:
                    plot_data.append({
                        "model_name": model_name,
                        "metric": m,
                        "score": results[m],
                    })

        if not plot_data:
            logger.warning("No valid data for plotting")
            return

        plot_df = pd.DataFrame(plot_data)

        n = len(metrics)
        rows = 2 if n > 2 else 1
        cols = 2 if n > 1 else 1

        fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        for i, m in enumerate(metrics):
            if i >= len(axes):
                break
            metric_data = plot_df[plot_df["metric"] == m]
            if metric_data.empty:
                continue
            sns.barplot(data=metric_data, x="model_name", y="score", ax=axes[i])
            axes[i].set_title(m.replace("_", " ").title())
            axes[i].set_xlabel("Model")
            axes[i].set_ylabel("Score")
            axes[i].tick_params(axis="x", rotation=45)

        # Hide unused axes
        for j in range(len(metrics), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Model comparison plot saved to {save_path}")

        plt.show()

    def plot_metric_heatmap(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot heatmap of model performance across metrics.

        Args:
            evaluation_results: Results from model evaluation
            metrics: Metrics to include
            save_path: Path to save the plot
        """
        if metrics is None:
            metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro"]

        logger.info("Creating metric heatmap...")

        heatmap_data = []
        for model_name, results in evaluation_results.items():
            if "error" in results:
                continue
            row = {"model_name": model_name}
            for m in metrics:
                row[m] = results.get(m, np.nan)
            heatmap_data.append(row)

        if not heatmap_data:
            logger.warning("No valid data for heatmap")
            return

        heatmap_df = pd.DataFrame(heatmap_data).set_index("model_name")

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={"label": "Score"})
        plt.title("Model Performance Heatmap")
        plt.xlabel("Metrics")
        plt.ylabel("Models")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Metric heatmap saved to {save_path}")

        plt.show()

    def get_model_ranking(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        weights: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Get a composite ranking based on multiple metrics.

        Note: This assumes higher is better for the provided metrics.
        Do not include loss metrics like ordinal_mae unless you transform them.

        Args:
            evaluation_results: Results from model evaluation
            metrics: Metrics to consider
            weights: Weights for each metric

        Returns:
            Ranking DataFrame with composite scores
        """
        if metrics is None:
            metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

        if weights is None:
            weights = [1.0] * len(metrics)

        if len(weights) != len(metrics):
            logger.warning("Weights length does not match metrics, using equal weights")
            weights = [1.0] * len(metrics)

        ranking_data = []

        for model_name, results in evaluation_results.items():
            if "error" in results:
                continue

            scores = []
            for m in metrics:
                val = results.get(m, np.nan)
                if pd.isna(val):
                    scores.append(0.0)
                else:
                    scores.append(float(val))

            composite_score = float(np.average(scores, weights=weights))

            row = {"model_name": model_name, "composite_score": composite_score}
            for m in metrics:
                row[m] = results.get(m, np.nan)
            ranking_data.append(row)

        if not ranking_data:
            return pd.DataFrame()

        ranking_df = pd.DataFrame(ranking_data).sort_values("composite_score", ascending=False).reset_index(drop=True)
        return ranking_df

    def get_best_model_info(self) -> Dict[str, Any]:
        """
        Get info about the best model from the last compare_models() call.
        """
        if self.best_model is None:
            logger.warning("No best model identified")
            return {}

        if self.comparison_results.empty:
            logger.warning("No comparison results available")
            return {}

        row = self.comparison_results[self.comparison_results["model_name"] == self.best_model]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def export_comparison_results(self, filepath: str) -> None:
        """
        Export the last comparison table to CSV.
        """
        if self.comparison_results.empty:
            logger.warning("No comparison results to export")
            return

        self.comparison_results.to_csv(filepath, index=False)
        logger.info(f"Comparison results exported to {filepath}")


def compare_models(
    evaluation_results: Dict[str, Dict[str, Any]],
    metric: str = "f1_weighted"
) -> pd.DataFrame:
    """
    Convenience function to compare models.
    """
    comparator = ModelComparator()
    return comparator.compare_models(evaluation_results, metric)

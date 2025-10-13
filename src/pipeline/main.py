"""
Main Pipeline Module - Orchestrates the complete ML pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# ---- Project imports ----
from ..data_management import (
    download_wine_dataset, load_wine_data, validate_data_schema,
    clean_data, create_quality_categories, split_data
)
from ..features import (
    create_interaction_features, StandardScalerWrapper,
    select_features_correlation
)
from ..models import (
    ModelTrainer, ModelEvaluator, ModelComparator
)
from ..visualization import create_eda_plots

# ---- Utilities ----
from ..utils.runtime_info import save_runtime_info
from ..utils.io import read_yaml, write_yaml
from ..utils.tracking import save_metrics, save_model_metadata

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WineQualityPipeline:
    """Complete pipeline for wine quality classification."""

    def __init__(self, project_root: str = ".", outdir: str | Path = "reports/runs/local",
                 config: Dict[str, Any] | None = None, random_state: int = 42):
        self.project_root = Path(project_root)
        self.random_state = random_state

        # Output root for this run (figures/metrics/config/runtime go here)
        self.outdir = Path(outdir)
        (self.outdir / "figures").mkdir(parents=True, exist_ok=True)
        (self.outdir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.outdir / "configs").mkdir(parents=True, exist_ok=True)
        (self.outdir / "runtime").mkdir(parents=True, exist_ok=True)

        # Traditional shared folders
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.reports_dir = self.project_root / "reports"
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

        # Config
        self.cfg = config or {}

        # Initialize components
        self.trainer = ModelTrainer(random_state=self.random_state)
        self.evaluator = ModelEvaluator()
        self.comparator = ModelComparator()

        # Data storage
        self.raw_data: pd.DataFrame | None = None
        self.cleaned_data: pd.DataFrame | None = None
        self.train_data: tuple[pd.DataFrame, pd.Series] | None = None
        self.val_data: tuple[pd.DataFrame, pd.Series] | None = None
        self.test_data: tuple[pd.DataFrame, pd.Series] | None = None
        self.feature_columns: list[str] | None = None

    # ---------------------- Data ----------------------

    def download_and_load_data(self) -> pd.DataFrame:
        """Download and load the wine quality dataset."""
        logger.info("Starting data download and loading...")

        # Allow overriding by config (if you already have a processed CSV)
        data_cfg = self.cfg.get("data", {})
        csv_path = data_cfg.get("path")
        sep = data_cfg.get("sep", ",")

        if csv_path:
            logger.info(f"Loading dataset from {csv_path}")
            raw_file = Path(csv_path)
            self.raw_data = load_wine_data(raw_file)
        else:
            # Fallback: download
            raw_file = download_wine_dataset(self.data_dir / "raw")
            self.raw_data = load_wine_data(raw_file)

        # Validate schema
        is_valid, validation_info = validate_data_schema(self.raw_data)
        if not is_valid:
            logger.warning(f"Data validation issues: {validation_info}")

        logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
        return self.raw_data

    def preprocess_data(self,
                        handle_missing: str | None = None,
                        remove_outliers: bool | None = None,
                        quality_method: str | None = None,
                        quality_threshold: int | None = None) -> pd.DataFrame:
        """Preprocess the dataset."""
        logger.info("Starting data preprocessing...")

        handle_missing = handle_missing or self.cfg.get("preprocess", {}).get("handle_missing", "median")
        remove_outliers = (remove_outliers
                           if remove_outliers is not None
                           else self.cfg.get("preprocess", {}).get("remove_outliers", False))
        quality_method = quality_method or self.cfg.get("labeling", {}).get("method", "multi")
        quality_threshold = quality_threshold or self.cfg.get("labeling", {}).get("threshold", 6)

        self.cleaned_data = clean_data(self.raw_data,
                                       handle_missing=handle_missing,
                                       remove_outliers=remove_outliers)

        self.cleaned_data = create_quality_categories(self.cleaned_data,
                                                      method=quality_method,
                                                      threshold=quality_threshold)

        processed_file = self.data_dir / "processed" / "wine_quality_processed.csv"
        processed_file.parent.mkdir(exist_ok=True, parents=True)
        self.cleaned_data.to_csv(processed_file, index=False)

        logger.info(f"Data preprocessing completed. Shape: {self.cleaned_data.shape}")
        return self.cleaned_data

    def create_eda_plots(self) -> Dict[str, str]:
        """Create exploratory data analysis plots."""
        logger.info("Creating EDA plots...")
        plot_paths = create_eda_plots(
            self.cleaned_data,
            str(self.outdir / "figures"),
            target_column="quality_category"
        )
        logger.info(f"EDA plots created: {len(plot_paths)} plots")
        return plot_paths

    def engineer_features(self,
                          create_interactions: bool | None = None,
                          feature_selection_method: str | None = None,
                          n_features: int | None = None) -> pd.DataFrame:
        """Engineer features for the model."""
        logger.info("Starting feature engineering...")

        fe_cfg = self.cfg.get("preprocess", {})
        create_interactions = fe_cfg.get("create_interactions", True) if create_interactions is None else create_interactions
        feature_selection_method = feature_selection_method or fe_cfg.get("feature_selection_method", "correlation")
        n_features = n_features or fe_cfg.get("k_top", 10)

        if create_interactions:
            self.cleaned_data = create_interaction_features(self.cleaned_data)
            logger.info("Interaction features created")

        if feature_selection_method == "correlation":
            selected_features = select_features_correlation(
                self.cleaned_data,
                target_column="quality_category",
                threshold=0.1
            )
        else:
            numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ["quality", "quality_category", "quality_label"]
            selected_features = [c for c in numeric_cols if c not in exclude_cols]
            if n_features and len(selected_features) > n_features:
                selected_features = selected_features[:n_features]

        self.feature_columns = selected_features
        logger.info(f"Feature engineering completed. Selected {len(selected_features)} features")
        return self.cleaned_data

    def split_and_scale_data(self,
                             test_size: float | None = None,
                             val_size: float | None = None
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train/validation/test sets and apply scaling."""
        logger.info("Splitting and scaling data...")

        test_size = test_size or self.cfg.get("test_size", 0.2)
        val_size = val_size or self.cfg.get("val_size", 0.2)

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            self.cleaned_data,
            target_column="quality_category",
            test_size=test_size,
            val_size=val_size,
            random_state=self.random_state
        )

        scaler = StandardScalerWrapper(columns=self.feature_columns)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        self.train_data = (X_train_scaled, y_train)
        self.val_data = (X_val_scaled, y_val)
        self.test_data = (X_test_scaled, y_test)

        import joblib
        joblib.dump(scaler, self.models_dir / "scaler.pkl")

        logger.info("Data splitting and scaling completed")
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

    # ---------------------- Modeling ----------------------

    def train_models(self) -> Dict[str, Dict[str, Any]]:
        """Train multiple models and evaluate performance."""
        logger.info("Starting model training...")

        X_train, y_train = self.train_data
        X_val, y_val = self.val_data

        training_results = self.trainer.train_all_models(X_train, y_train, X_val, y_val)

        evaluation_results: Dict[str, Dict[str, Any]] = {}
        for model_name, results in training_results.items():
            if results.get("training_completed", False):
                model = results["model"]
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None

                eval_results = self.evaluator.evaluate_classification_performance(
                    y_val, y_pred, y_pred_proba, model_name
                )
                evaluation_results[model_name] = eval_results

                save_metrics(eval_results, self.outdir / "metrics" / f"{model_name}_val_metrics.json")

                model_path = self.models_dir / f"{model_name}.pkl"
                self.trainer.save_model(model_name, str(model_path))

        # --- Compare and select champion ---
        comparison_df = self.comparator.compare_models(evaluation_results)
        comparison_file = self.outdir / "metrics" / "model_comparison_val.csv"
        comparison_df.to_csv(comparison_file, index=False)
        self._save_results(evaluation_results, comparison_df)

        best_model_name = self.comparator.best_model
        if best_model_name:
            best_model = self.trainer.models.get(best_model_name)
            if best_model is not None:
                import joblib
                champion_path = self.models_dir / "champion.joblib"
                joblib.dump(best_model, champion_path)
                logger.info(f"Champion model '{best_model_name}' saved to {champion_path}")

                try:
                    feature_names = self.feature_columns or []
                    save_model_metadata(
                        best_model,
                        best_model_name,
                        feature_names,
                        label_names=["Low", "Medium", "High"],
                        path=self.models_dir / "champion_metadata.json"
                    )
                except Exception as meta_e:
                    logger.warning(f"Failed to save model metadata: {meta_e}")
            else:
                logger.warning(f"Best model '{best_model_name}' not found in trainer registry.")
        else:
            logger.warning("No best model identified during comparison.")

        logger.info("Model training and evaluation completed")
        return evaluation_results

    # ---------------------- Persistence helpers ----------------------

    def _save_results(self,
                      evaluation_results: Dict[str, Dict[str, Any]],
                      comparison_df: pd.DataFrame) -> None:
        """Save evaluation and comparison results."""
        eval_file = self.reports_dir / "evaluation_results.json"
        with open(eval_file, "w", encoding="utf-8") as f:
            json_results = {}
            for model_name, results in evaluation_results.items():
                json_results[model_name] = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        json_results[model_name][key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        json_results[model_name][key] = float(value)
                    else:
                        json_results[model_name][key] = value
            json.dump(json_results, f, indent=2)

        comparison_file = self.reports_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Results saved to {self.reports_dir}")

    # ---------------------- Orchestration ----------------------

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete machine learning pipeline."""
        logger.info("Starting complete wine quality classification pipeline...")

        try:
            save_runtime_info(self.outdir / "runtime" / "runtime_info.json")
            self.download_and_load_data()
            self.preprocess_data()
            plot_paths = self.create_eda_plots()
            self.engineer_features()
            self.split_and_scale_data()
            evaluation_results = self.train_models()
            best_model_info = self.comparator.get_best_model_info()

            logger.info("Complete pipeline finished successfully!")

            if self.cfg:
                write_yaml(self.cfg, self.outdir / "configs" / "config_used.yaml")

            return {
                "status": "success",
                "data_shape": None if self.cleaned_data is None else self.cleaned_data.shape,
                "feature_count": 0 if self.feature_columns is None else len(self.feature_columns),
                "plot_paths": plot_paths,
                "evaluation_results": evaluation_results,
                "best_model": best_model_info,
                "models_dir": str(self.models_dir),
                "outdir": str(self.outdir),
                "reports_dir": str(self.reports_dir),
            }

        except Exception as e:
            logger.exception("Pipeline failed")
            return {"status": "error", "error": str(e)}


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml",
                   help="Path to a YAML config with run settings.")
    p.add_argument("--outdir", type=str, default="reports/runs/local",
                   help="Where to save runtime/config/metrics/figures for THIS run.")
    p.add_argument("--project_root", type=str, default=".",
                   help="Project root (where data/, models/, reports/ live).")
    return p.parse_args()


def run_complete_pipeline(project_root: str = ".", outdir: str | Path = "reports/runs/local",
                          config: Dict[str, Any] | None = None,
                          random_state: int = 42) -> Dict[str, Any]:
    pipeline = WineQualityPipeline(project_root=project_root, outdir=outdir,
                                   config=config, random_state=random_state)
    return pipeline.run_complete_pipeline()


if __name__ == "__main__":
    args = cli()

    cfg = {}
    try:
        cfg = read_yaml(args.config)
    except Exception as e:
        logger.warning(f"Could not read config {args.config}: {e}. Proceeding with defaults.")

    results = run_complete_pipeline(
        project_root=args.project_root,
        outdir=args.outdir,
        config=cfg,
        random_state=cfg.get("random_state", 42)
    )
    print(f"Pipeline completed with status: {results.get('status')}")

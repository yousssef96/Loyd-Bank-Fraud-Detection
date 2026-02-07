
"""
Runs sequentially: load → validate → preprocess → feature engineering
"""

import os, sys, time
from pathlib import Path
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import(
    classification_report, precision_score,
    recall_score, f1_score, roc_auc_score
)
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules - Core pipeline components
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_Lloyd_data

def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.
    
    """
    
    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    # Configure MLflow to use local file-based tracking (not a tracking server)
    project_root = Path(__file__).resolve().parent.parent
    mlruns_dir = project_root / "mlruns"
    mlruns_path = args.mlflow_uri or mlruns_dir.as_uri()
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment) # Creates experiment if doesn't exist

    # Start MLflow run - all subsequent logging will be tracked under this run
    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        mlflow.log_param("model", "lightgbm") # Model type for comparison
        mlflow.log_param("test_size", args.test_size) # Train/test split ratio

        # === STAGE 1: Data Loading & Validation ===
        print("Loading data...")
        df = load_data(args.input) # Load raw Excel data with error handling
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")


        # === CRITICAL: Data Quality Validation ===
        # This step is ESSENTIAL for production ML - validates data quality before training
        print("Validating data quality with Great Expectations...")
        is_valid, failed = validate_Lloyd_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid)) # Track data quality over time

        if not is_valid:
            # Log validation failures for debugging
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("Data validation passed. Logged to MLflow.")

        # === STAGE 2: Data Preprocessing ===
        print("Preprocessing data...")
        df = preprocess_data(df) # Basic cleaning (handle missing values, fix data types)

        # Save processed dataset for reproducibility and debugging
        processed_path = os.path.join(project_root, "data", "processed", "Lloyd_loan_processed.xlsx")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_excel(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path} | Shape: {df.shape}")


        # === STAGE 3: Feature Engineering - CRITICAL for Model Performance ===
        print("Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        # Apply feature engineering transformations
        df_enc = build_features(df, target_col=target) # Binary encoding + one-hot encoding

        # IMPORTANT Casury step: Convert boolean columns to integers for LightGBM compatibility
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"Feature engineering completed: {df_enc.shape[1]} features")

        # === CRITICAL: Save Feature Metadata for Serving Consistency ===
        # This ensures serving pipeline uses exact same features in exact same order
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Get feature columns (exclude target)
        feature_cols = list(df_enc.drop(columns=[target]).columns)

        # Save locally for development serving
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)
        
         # Log to MLflow for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # ESSENTIAL: Save preprocessing artifacts for serving pipeline
        # These artifacts ensure training and serving use identical transformations
        preprocessing_artifact = {
            "feature_columns": feature_cols,  # Exact feature order
            "target": target                  # Target column name
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")

        # === STAGE 4: Train/Test Split ===
        print("Splitting data...")
        X = df_enc.drop(columns=[target])  # Feature matrix
        y = df_enc[target]                 # Target vector
        
        # Stratified split to maintain class distribution in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,    # Default: 20% for testing
            stratify=y,                  # Maintain class balance
            random_state=42              # Reproducible splits
        )
        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # === CRITICAL: Handle Class Imbalance ===
        # Calculate scale_pos_weight to handle imbalanced dataset
        # This tells LGBM to give more weight to the minority class (churners)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        # === STAGE 5: Model Training with Optimized Hyperparameters ===
        print("Training LightGBM model...")
        
        # IMPORTANT: These hyperparameters were optimized through hyperparameter tuning
        # In production, consider using hyperparameter optimization tools like Optuna
        model = LGBMClassifier(
            # Tree structure parameters
            n_estimators=372, # Number of trees (OPTIMIZED)
            learning_rate=0.01136707376225912, # Step size shrinkage (OPTIMIZED) 
            max_depth=4, # Maximum tree depth (OPTIMIZED)
            num_leaves=2323,

            # Regularization parameters 
            subsample=0.9853226652178365,  # Sample ratio of training instances
            colsample_bytree=0.916313213704182, # Sample ratio of features for each tree
            min_child_samples=67, 
            reg_alpha=0.002877503562818248, 
            reg_lambda=3.7081026468197853e-08,

            # Performance parameters
            n_jobs=-1,              # Use all CPU cores
            random_state=42,        # Reproducible results
            importance_type="gain",
            verbosity=-1,
        
            
            # ESSENTIAL: Handle class imbalance
            scale_pos_weight=scale_pos_weight  # Weight for positive class (churners)

        )
        # IMPORTANT: Pipeline to handle TargetEncoder with no data leakage
        pipe_lgbm = Pipeline([
        ('encoder', TargetEncoder(cols=['emp_title', 'addr_state'])),
        ('classifier', model)
        ])


        # === Train Model and Track Training Time ===
        t0 = time.time()
        pipe_lgbm.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)  # Track training performance
        print(f"Model trained in {train_time:.2f} seconds")

        # === STAGE 6: Model Evaluation ===
        print("Evaluating model performance...")
        
        # Generate predictions and track inference time
        t1 = time.time()
        proba = pipe_lgbm.predict_proba(X_test)[:, 1]  # Get probability of default (class 1)
        
        
        y_pred = pipe_lgbm.predict(X_test)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)  # Track inference performance

        # === CRITICAL: Log Evaluation Metrics to MLflow ===
        # These metrics are essential for model comparison and monitoring
        precision = precision_score(y_test, y_pred)    # Of predicted defaulters, how many actually defaulted?
        recall = recall_score(y_test, y_pred)          # Of actual defaulters, how many did we catch?
        f1 = f1_score(y_test, y_pred)                  # Harmonic mean of precision and recall
        roc_auc = roc_auc_score(y_test, proba)         # Area under ROC curve 
        
        # Log all metrics for experiment tracking
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall) 
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        print(f"Model Performance:")
        print(f"Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        # === STAGE 7: Model Serialization and Logging ===
        print("Saving model pipeline to MLflow...")
        # ESSENTIAL: Log model pipeline in MLflow's standard format for serving
        mlflow.sklearn.log_model(
            pipe_lgbm, 
            artifact_path="model_pipeline"  # This creates a 'model_pipeline/' folder in MLflow run artifacts
        )
        print("Model saved to MLflow for serving pipeline")

        # === Final Performance Summary ===
        print(f"\nPerformance Summary:")
        print(f"Training time: {train_time:.2f}s")
        print(f"Inference time: {pred_time:.4f}s")
        print(f"Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run loan status pipeline with LightGBM + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to Excel (e.g., data/raw/LBG Step Up Data Set.xlsx)")
    p.add_argument("--target", type=str, default="loan_status")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Lloyd Bank Loan status")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="override MLflow tracking URI, else use project_root/mlruns")
    
    args = p.parse_args()
    main(args)

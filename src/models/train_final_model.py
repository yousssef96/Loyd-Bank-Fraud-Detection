import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from mlflow.models import infer_signature
from src.utils.logger import logger


def train_final_logreg(preprocess, best_params, X_train, X_test, y_train, y_test,
                        tracking_uri: str = "http://127.0.0.1:5000",
                        experiment_name: str = "loan_default_logreg_tuning",
                        registered_model_name: str = "Best_loan_status_calssifier"):
    """
    Trains the final LogisticRegression pipeline using the best Optuna params,
    evaluates on X_test, and logs + registers the model in MLflow.

    Args:
        preprocess: Unfitted ColumnTransformer from build_preprocessor().
        best_params (dict): study.best_params from run_logreg_tuning().
        X_train, X_test, y_train, y_test: Train/test split.

    Returns:
        Pipeline: The fitted final pipeline.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info("Training final LogisticRegression model with tuned params")
        best_model = LogisticRegression(
            **best_params,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )

        pipe_lr = Pipeline(steps=[
            ('preprocess', preprocess),
            ('model', best_model)
        ])

        pipe_lr.fit(X_train, y_train)

        proba = pipe_lr.predict_proba(X_test)[:, 1]
        y_pred = pipe_lr.predict(X_test)

        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, proba)

        logger.info(f"Final model performance -> precision={precision:.3f}, recall={recall:.3f}, "
                    f"f1={f1:.3f}, auc={auc:.3f}")

        signature = infer_signature(X_train, pipe_lr.predict(X_train))

        with mlflow.start_run(run_name="best_LR_model"):
            mlflow.log_params({**best_params, "solver": "liblinear", "class_weight": "balanced"})
            mlflow.log_metrics({"precision": precision, "recall": recall, "f1": f1, "auc": auc})
            mlflow.sklearn.log_model(
                sk_model=pipe_lr, name="LR_Pipeline",
                registered_model_name=registered_model_name,
                signature=signature, input_example=X_train
            )

        logger.info(f"Model registered as '{registered_model_name}'")
        return pipe_lr
    except Exception:
        logger.exception("Failed to train and log final model")
        raise


if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_data
    from src.data.preprocess import preprocess_data
    from src.data.split import split_data
    from src.data.transformation import build_preprocessor
    from src.models.tune_logreg import run_logreg_tuning

    file_path = Path("data/raw/LBG Step Up Data Set.xlsx")
    df = load_data(file_path=file_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocess = build_preprocessor(X_train)


    
    study = run_logreg_tuning(preprocess, X_train, y_train)
    pipe_lr = train_final_logreg(preprocess, study.best_params, X_train, X_test, y_train, y_test)
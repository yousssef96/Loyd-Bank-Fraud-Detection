import mlflow
import optuna
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from src.utils.logger import logger


def run_logreg_tuning(preprocess, X_train, y_train, n_trials: int = 30,
                       tracking_uri: str = "http://127.0.0.1:5000",
                       experiment_name: str = "loan_default_logreg_tuning"):
    """
    Runs an Optuna study tuning LogisticRegression hyperparameters, logging each
    trial as a nested MLflow run.

    Returns:
        optuna.Study: The completed study, with study.best_params / study.best_value.
    """
    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42
        }

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            model = LogisticRegression(**params)
            pipe_lr = Pipeline(steps=[
                ('preprocess', preprocess),
                ('classifier', model)
            ])

            cv_scores = cross_val_score(
                pipe_lr, X_train, y_train,
                cv=5, scoring="recall", n_jobs=-1,
                error_score="raise"
            )

            avg_recall = cv_scores.mean()

            mlflow.log_params(params)
            mlflow.log_metrics({"recall": avg_recall})

        return avg_recall

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info(f"Starting Optuna study: {n_trials} trials")
        with mlflow.start_run(run_name="logreg_optuna_search"):
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_recall", study.best_value)

        logger.info(f"Best Params: {study.best_params}")
        logger.info(f"Best Recall: {study.best_value}")
        return study
    except Exception:
        logger.exception("Failed during Optuna tuning")
        raise


if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_data
    from src.data.preprocess import preprocess_data
    from src.data.split import split_data
    from src.data.transformation import build_preprocessor

    file_path = Path("data/raw/LBG Step Up Data Set.xlsx")
    df = load_data(file_path=file_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    

    preprocess = build_preprocessor(X_train)
    study = run_logreg_tuning(preprocess, X_train, y_train)
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, recall_score


def train_model(df: pd.DataFrame, target_col: str, high_card_cols: list):
    """
    Trains an LightGBM model and logs with MLflow
    
    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of target column.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42
        )
    
    pipe_lgbm = Pipeline([
        ('encoder', TargetEncoder(cols=['emp_title', 'addr_state'])),
        ('classifier', model)
    ])

    with mlflow.start_run():
        # calculate model average recall score
        scores = cross_val_score(
            estimator=pipe_lgbm,
            X=X_train,
            y=y_train,
            scoring="recall",
            cv=5,
            n_jobs=-1
        )
        avg_recall = scores.mean()

        # Train model
        pipe_lgbm.fit(X_train, y_train)
        preds = pipe_lgbm.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)


        
        # Log params, metrics, and pipeline
        mlflow.log_param("n_estimator", 500)
        mlflow.log_metric("cv_avg_recall", avg_recall)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_recall", rec)
        mlflow.sklearn.log_model(pipe_lgbm, "pipeline")

        # Log dataset
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. CV Recall: {avg_recall:.4f}, Test Acc: {acc:.4f}, Test Recall: {rec:.4f}")

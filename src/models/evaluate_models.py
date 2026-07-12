import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score
from src.utils.logger import logger


def evaluate(true, predicted, proba):
    recall = recall_score(true, predicted)
    f1 = f1_score(true, predicted)
    auc = roc_auc_score(true, proba)
    precision = precision_score(true, predicted)
    return recall, f1, auc, precision


def modelEvaluation(preprocess, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Fits LogisticRegression, RandomForest, and XGBoost with default params,
    evaluates each on X_test at the default 0.5 threshold.

    Args:
        preprocess: Unfitted ColumnTransformer from build_preprocessor().
        X_train, X_test, y_train, y_test: Train/test split.

    Returns:
        pd.DataFrame: Results sorted by recall, descending.
    """
    try:
        negative_cases = (y_train == 0).sum()
        positive_cases = (y_train == 1).sum()
        ratio = negative_cases / positive_cases

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "RandomForestClassifier": RandomForestClassifier(class_weight='balanced', random_state=42),
            "XGBClassifier": XGBClassifier(scale_pos_weight=ratio, random_state=42)
        }

        results = []
        for name, model in models.items():
            logger.info(f"Evaluating {name}")
            full_pipe = Pipeline(steps=[
                ("preprocess", preprocess),
                ('model', model)
            ])
            full_pipe.fit(X_train, y_train)

            y_pred = full_pipe.predict(X_test)
            y_proba = full_pipe.predict_proba(X_test)[:, 1]

            recall, f1, auc, precision = evaluate(true=y_test, predicted=y_pred, proba=y_proba)

            results.append({
                "model": name,
                "recall": recall,
                "f1": f1,
                'auc': auc,
                'precision': precision
            })
            logger.info(f"{name} -> recall={recall:.3f}, f1={f1:.3f}, auc={auc:.3f}, precision={precision:.3f}")

        return pd.DataFrame(results).sort_values(by='recall', ascending=False)
    except Exception:
        logger.exception("Failed during model evaluation")
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

    
    results_df = modelEvaluation(preprocess, X_train, X_test, y_train, y_test)
    print(results_df)
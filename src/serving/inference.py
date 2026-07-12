from pathlib import Path
import pandas as pd
import mlflow
from src.utils.logger import logger


MODEL_PATH = Path(__file__).parent / "model"

_model = None  # loaded once, reused across requests


def load_model():
    """
    Loads the MLflow pipeline model from disk. Cached after first call.
    """
    global _model
    if _model is None:
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            _model = mlflow.sklearn.load_model(str(MODEL_PATH))
            logger.info("Model loaded successfully")
        except Exception:
            logger.exception("Failed to load model")
            raise
    return _model


def predict(input_df: pd.DataFrame) -> dict:
    """
    Runs prediction on a single-row (or multi-row) DataFrame of raw features.

    Args:
        input_df (pd.DataFrame): Raw feature data, same columns as training X.

    Returns:
        dict: predictions and probabilities as lists.
    """
    try:
        model = load_model()
        proba = model.predict_proba(input_df)[:, 1]
        pred = model.predict(input_df)

        return {
            "prediction": pred.tolist(),
            "probability": proba.tolist()
        }
    except Exception:
        logger.exception("Failed during prediction")
        raise



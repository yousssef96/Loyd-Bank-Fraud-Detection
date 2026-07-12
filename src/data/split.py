import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import logger


def split_data(
    df: pd.DataFrame,
    target_col: str = "loan_status",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits a DataFrame into train/test features and target, stratified on target_col.

    Args:
        df (pd.DataFrame): Preprocessed dataset including target column.
        target_col (str): Name of the target column.
        test_size (float): Fraction of data to hold out for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    try:
        logger.info(f"Splitting data with test_size={test_size}, stratify on '{target_col}'")
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Split complete: X_train={X_train.shape}, X_test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception:
        logger.exception("Failed to split data")
        raise


if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_data
    from src.data.preprocess import preprocess_data

    file_path = Path("data/raw/LBG Step Up Data Set.xlsx")
    df = load_data(file_path=file_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(X_train.head())
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from feature_engine.outliers import Winsorizer
from src.utils.logger import logger


ONE_HOT_COLS = ['home_ownership', 'purpose', 'term']

SKEWED_COLS = ['total_bal_ex_mort', 'max_bal_bc', 'avg_cur_bal',
               'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl']


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Builds the full preprocessing ColumnTransformer.

    Skewed numeric columns: impute -> log1p -> clip to 5th/95th percentile
    (learned from train only) -> scale.
    Other numeric columns: impute -> scale.
    Categorical columns: one-hot encode.

    Args:
        X_train (pd.DataFrame): Raw training features (before any manual
            transformation) — used only to infer column lists.

    Returns:
        ColumnTransformer: Unfitted preprocessor, ready to go inside a model Pipeline.
    """
    try:
        logger.info("Building preprocessing ColumnTransformer")

        numerical_cols = X_train.select_dtypes(include=[int, float]).columns.tolist()
        other_numeric_cols = [c for c in numerical_cols if c not in SKEWED_COLS]

        skewed_pipeline = Pipeline([
            ("impute", IterativeImputer(max_iter=10, random_state=42, skip_complete=True, min_value=0)),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("clip", Winsorizer(capping_method="quantiles", tail="both", fold=0.05)),
            ("scale", StandardScaler())
        ])

        other_numeric_pipeline = Pipeline([
            ("impute", IterativeImputer(max_iter=10, random_state=42, skip_complete=True, min_value=0)),
            ("scale", StandardScaler())
        ])

        preprocess = ColumnTransformer(transformers=[
            ("onehot", OneHotEncoder(drop='first', handle_unknown="ignore"), ONE_HOT_COLS),
            ("skewed_num", skewed_pipeline, SKEWED_COLS),
            ("other_num", other_numeric_pipeline, other_numeric_cols)
        ], remainder='drop')

        logger.info(f"Preprocessor built: {len(ONE_HOT_COLS)} onehot cols, "
                    f"{len(SKEWED_COLS)} skewed cols, {len(other_numeric_cols)} other numeric cols")
        return preprocess
    except Exception:
        logger.exception("Failed to build preprocessor")
        raise


if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_data
    from src.data.preprocess import preprocess_data
    from src.data.split import split_data

    file_path = Path("data/raw/LBG Step Up Data Set.xlsx")
    df = load_data(file_path=file_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocess = build_preprocessor(X_train)

    # sanity check: fit on train, transform both, confirm shapes and no errors
    X_train_transformed = preprocess.fit_transform(X_train)
    X_test_transformed = preprocess.transform(X_test)

    print("X_train shape after preprocessing:", X_train_transformed.shape)
    print("X_test shape after preprocessing:", X_test_transformed.shape)
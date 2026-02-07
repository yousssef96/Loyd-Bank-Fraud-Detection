import pandas as pd

def map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    
    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic and must be consistent between training and serving.
    
    """
    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # === DETERMINISTIC BINARY MAPPINGS ===
    # CRITICAL: These exact mappings are hardcoded in serving pipeline
    
    # term months mapping
    if valset == {" 36 months", " 60 months"}:
        return s.map({' 36 months': 0, ' 60 months': 1}).astype("Int64")
    
    # === GENERIC BINARY MAPPING ===
    # For any other 2-category feature, use stable alphabetical ordering
    if len(vals) == 2:
        # Sort values to ensure consistent mapping across runs
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")
    
    # === NON-BINARY FEATURES ===
    # Return unchanged - will be handled by one-hot encoding or TargetEncoding
    return s


def build_features(df: pd.DataFrame, target_col: str = "loan_status") -> pd.DataFrame:
    """
    Apply complete features engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the 
    serving pipeline to ensure prediction accuracy.
    
    """
    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns...")

    # Identify Feature Types
    # Find categorical columns (object dtype) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]). columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    # Split Categorical by Cardinality
    # Binary features (exactly 2 unique values) get binary encoding
    # Multi-category features (>2 to 15 unique values) get one-hot encoding
    # Multi-category features (>15 unique values) get target encoder
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if 2 < df[c].dropna().nunique() <= 15]
    high_card_cols = [c for c in obj_cols if df[c].dropna().nunique() > 15]

    print(f"Binary features: {len(binary_cols)} | "
          f"Multi-category features: {len(multi_cols)} | "
          f"High-cardinality features: {len(high_card_cols)}")
    
    if binary_cols:
        print(f"Binary: {binary_cols}")
    if multi_cols:
        print(f"Multi-category: {multi_cols}")
    if high_card_cols:
        print(f"High-cardinality: {high_card_cols}")
    
    # Apply Binary Encoding
    # Convert 2-category features to 0/1 using deterministic mappings
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = map_binary_series(df[c].astype(str))
        print(f"{c}: {original_dtype} -> binary (0/1)")

    # Convert Boolean Columns
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"Converted {len(bool_cols)} boolean columns to int: {bool_cols}")
    
    # One-Hot Encoding for Multi-Category Features
    # drop_first=True prevents multicollinearity
    if multi_cols:
        print(f"Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape
        
        # Apply one-hot encoding with drop_first=True (same as serving)
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True, dtype="int64")
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"Created {new_features} new features from {len(multi_cols)} categorical columns")
    
    if high_card_cols:
        print(f"Keeping {len(high_card_cols)} columns for Target Encoding in training script.")
        
        
    
    # Data Type Cleanup
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(-1).astype(int)
    
    print(f"Feature engineering complete: {df.shape[1]} final features")
    return df

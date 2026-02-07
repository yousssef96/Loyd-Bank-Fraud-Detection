import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "loan_status") -> pd.DataFrame:
    """"
    Basic cleaning for Lloyd Bank loans.
    - trim column names
    - drop ID cols
    - map target loan_status to 0/1
    - map emp_length to numeric 0 -> 10
    - simple NA handling
    """

    # tidy headers
    df.columns = df.columns.str.strip() # Remove leading/trailing whitespace

    # drop ids if present
    for col in ["id", "CustomerID", "customer_id", "customerID"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # target to 0/1 if it's Fully Paid/Charged Off
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"Fully Paid" : 0, "Charged Off" : 1})
    
    # clean and transform Emp length to numeric 0 -> 10
    if "emp_length" in df.columns:
        emp_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
            }
        df['emp_length'] = df['emp_length'].map(emp_map)

    # simple NA strategy:
    # - numeric: fill with 0
    # - others: leave for encoders to handle
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(-1)

    return df
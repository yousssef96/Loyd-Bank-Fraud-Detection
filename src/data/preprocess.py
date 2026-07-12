import pandas as pd
from pathlib import Path
from src.utils.logger import logger
from src.data.load_data import load_data

def preprocess_data(df: pd.DataFrame, target_col: str = "loan_status") -> pd.DataFrame:
    """"
    Basic cleaning for Lloyd Bank loans.
    - trim column names
    - drop ID cols
    - map target loan_status to 0/1
    - map emp_length to numeric 0 -> 10
    
    """
    try:

        logger.info("Removing leading/trailing whitespace from column names")
        # tidy headers
        df.columns = df.columns.str.strip() # Remove leading/trailing whitespace

        logger.info("Dropping the id column")
        # drop id if present
        df = df.drop(columns=['id'])

        logger.info("Mapping the target column to 0 and 1")
        # target to 0/1 if it's Fully Paid/Charged Off
        if target_col in df.columns and df[target_col].dtype == "object":
            df[target_col] = df[target_col].str.strip().map({"Fully Paid" : 0, "Charged Off" : 1})


        logger.info("Cleaning and transforming Emp length to numeric 0 -> 10")
        # clean and transform Emp length to numeric 0 -> 10
        if "emp_length" in df.columns:
            emp_map = {
                '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
                '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
                }
            df['emp_length'] = df['emp_length'].map(emp_map)

        logger.info("Preprocess Completed")
        return df
    except Exception:
        logger.exception("Failed to preprocess the data")



if __name__ == "__main__":
    file_path = Path("data/raw/LBG Step Up Data Set.xlsx")
    df = load_data(file_path=file_path)
    df = preprocess_data(df)
    print(df.head())
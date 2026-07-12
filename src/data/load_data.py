import pandas as pd
from pathlib import Path
from src.utils.logger import logger


def load_data(file_path : str | Path) -> pd.DataFrame:
    """
    Loads Excel data into a pandas DataFrame.
    
    Args:
        file_path (str) : Path to Exel file
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        logger.info(f"Attempting to load data from: {file_path}")
        df = pd.read_excel(file_path)
        logger.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception:
        logger.exception("Failed to load Excel dataset")
        raise



if __name__ == "__main__":
    file_path = Path("data/raw/LBG Step Up Data Set.xlsx")
    df = load_data(file_path=file_path)
    print(df.head())
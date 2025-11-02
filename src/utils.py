import pandas as pd

def load_dataset(path: str):
    """
    Load dataset CSV that must contain columns: 'text' and 'lang'
    Returns pandas DataFrame with those columns.
    """
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'lang' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'lang' columns.")
    # Drop NaNs
    return df[['text', 'lang']].dropna().reset_index(drop=True)
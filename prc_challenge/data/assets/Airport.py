import pandas as pd


def get_Airport() -> pd.DataFrame:
    """_summary_

    Returns:
        pd.DataFrame: _description_
    """

    asset = pd.read_parquet(f"~/prc-challenge-2025/data/apt.parquet")

    return asset

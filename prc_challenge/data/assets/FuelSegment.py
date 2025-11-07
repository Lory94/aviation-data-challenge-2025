import pandas as pd


def get_FuelSegment(variant: str) -> pd.DataFrame:
    """_summary_

    Args:
        variant (str): _description_

    Returns:
        pd.DataFrame: _description_
    """

    assert variant in ("train", "rank")

    flag = {"train": "train", "rank": "rank_submission"}[variant]

    asset = pd.read_parquet(f"~/prc-challenge-2025/data/fuel_{flag}.parquet")

    return asset

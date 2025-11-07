import pandas as pd


def get_FlightList(variant: str) -> pd.DataFrame:
    """_summary_

    Args:
        variant (str): _description_

    Returns:
        pd.DataFrame: _description_
    """

    # variant
    assert variant in ("train", "rank")

    asset = pd.read_parquet(f"~/prc-challenge-2025/data/flightlist_{variant}.parquet")

    return asset

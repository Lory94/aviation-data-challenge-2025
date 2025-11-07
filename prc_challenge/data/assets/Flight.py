import pandas as pd
import os

class Flight(object):

    def __init__(self, variant: str):

        self.directory = f"~/prc-challenge-2025/data/flights_{variant}"
        filenames = os.listdir(os.path.expanduser(self.directory))
        self.flight_ids = map(lambda filename: filename.split(".")[0], filenames)

    def __getitem__(self, flight_id):
        flight = self.load_flight(flight_id)
        return flight
    
    def load_flight(self, flight_id):
        filename = f"{flight_id}.parquet"
        asset = pd.read_parquet(os.path.join(self.directory, filename))
        return asset
    
    def keys(self):
        return list(self.flight_ids)
    
def get_Flight(variant: str) -> Flight:
    """_summary_

    Args:
        variant (str): _description_

    Returns:
        Flight: _description_
    """

    # variant
    assert variant in ("train", "rank")

    return Flight(variant=variant)

import pandas as pd
from tqdm import tqdm

from .BaseFeatureEngineering import BaseFeatureEngineering


class AddTimeSinceTakeOff_fromFlightList_v0_0_0(BaseFeatureEngineering):

    def __call__(
        self,
        FuelSegment_X,
        FuelSegment_Y,
        FlightList,
        Airport,
        Flight,
        column_functions,
    ):
        merged_df = pd.merge(
            FuelSegment_X, FlightList, how="left", on="flight_id", suffixes=("", "_fl")
        ).set_index(FuelSegment_X.index)

        FuelSegment_X["time_since_take_off[seconds]"] = (
            merged_df["start"] - merged_df["takeoff"]
        ) / pd.Timedelta(seconds=1)
        column_functions["numerical"].append("time_since_take_off[seconds]")
        FuelSegment_X["time_until_landed[seconds]"] = (
            merged_df["landed"] - merged_df["start"]
        ) / pd.Timedelta(seconds=1)
        column_functions["numerical"].append("time_until_landed[seconds]")
        FuelSegment_X["time_fraction"] = (merged_df["start"] - merged_df["takeoff"]) / (
            merged_df["landed"] - merged_df["takeoff"]
        )
        column_functions["numerical"].append("time_fraction")

        return FuelSegment_X, column_functions

import pandas as pd
from tqdm import tqdm
from .BaseFeatureEngineering import BaseFeatureEngineering


class AddTimeSinceTakeOff_v0_0_0(BaseFeatureEngineering):

    def __init__(self, method):
        assert method in ("first_flight_timestamp", "earliest_fuel_segment_start")
        self.method = method

    def __call__(self, FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight, column_functions):

        match self.method:
            case "first_flight_timestamp":

                mapping = {}
                for flight_id in tqdm(Flight.flight_ids):
                    mapping[flight_id] = Flight[flight_id]["timestamp"].iloc[0]
                take_off_time = FuelSegment_X["flight_id"].map(mapping)

            case "earliest_fuel_segment_start":
                FuelSegment_X.groupby("flight_id").agg({"start": "min"})

            case _:
                raise NotImplementedError

        FuelSegment_X["time_since_take_off[seconds]"] = (FuelSegment_X["start"] - take_off_time) / pd.Timedelta(seconds=1)
        column_functions["numerical"].append('time_since_take_off[seconds]')

        return FuelSegment_X, column_functions
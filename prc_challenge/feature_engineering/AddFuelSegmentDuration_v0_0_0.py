import pandas as pd
from .BaseFeatureEngineering import BaseFeatureEngineering


class AddFuelSegmentDuration_v0_0_0(BaseFeatureEngineering):

    def __call__(self, FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight, column_functions):

        FuelSegment_X['segment_duration[seconds]'] = (FuelSegment_X["end"] - FuelSegment_X["start"]) / pd.Timedelta(seconds=1)
        column_functions["numerical"].append('segment_duration[seconds]')

        return FuelSegment_X, column_functions
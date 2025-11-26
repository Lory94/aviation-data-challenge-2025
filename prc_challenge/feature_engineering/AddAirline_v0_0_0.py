from pathlib import Path

import pandas as pd

from .BaseFeatureEngineering import BaseFeatureEngineering


class AddAirline_v0_0_0(BaseFeatureEngineering):

    def __call__(
        self,
        FuelSegment_X,
        FuelSegment_Y,
        FlightList,
        Airport,
        Flight,
        column_functions,
    ):
        csv_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "airline_per_flight_flightradar.csv"
        )
        airlines = pd.read_csv(csv_path, usecols=("flight_id", "airline"))
        FuelSegment_X = FuelSegment_X.merge(airlines, on="flight_id")
        column_functions["categorical"].append("airline")

        return FuelSegment_X, column_functions

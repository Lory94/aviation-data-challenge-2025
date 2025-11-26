import pandas as pd
from airports import airport_data

from .BaseFeatureEngineering import BaseFeatureEngineering


class AddAirportElevation_v0_0_0(BaseFeatureEngineering):

    def __call__(
        self,
        FuelSegment_X,
        FuelSegment_Y,
        FlightList,
        Airport,
        Flight,
        column_functions,
    ):
        FuelSegment_X = FuelSegment_X.merge(
            FlightList[["flight_id", "origin_icao", "destination_icao"]],
            on="flight_id",
            how="left",
        )

        origins = set(FuelSegment_X["origin_icao"].unique())
        dests = set(FuelSegment_X["destination_icao"].unique())
        all_icaos = {x for x in origins.union(dests) if pd.notna(x)}

        elevation_map = {}

        for icao in all_icaos:
            try:
                data = airport_data.get_airport_by_icao(icao)
                if data:
                    elevation_map[icao] = data[0]["elevation_ft"]
                else:
                    elevation_map[icao] = None
            except Exception:
                elevation_map[icao] = None

        FuelSegment_X["elevation_origin_ft"] = FuelSegment_X["origin_icao"].map(
            elevation_map
        )
        FuelSegment_X["elevation_destination_ft"] = FuelSegment_X[
            "destination_icao"
        ].map(elevation_map)

        columns_to_drop = ["origin_icao", "destination_icao"]
        FuelSegment_X.drop(columns=columns_to_drop, inplace=True)

        new_features = [
            "elevation_origin_ft",
            "elevation_destination_ft",
        ]
        column_functions["numerical"].extend(new_features)

        return FuelSegment_X, column_functions

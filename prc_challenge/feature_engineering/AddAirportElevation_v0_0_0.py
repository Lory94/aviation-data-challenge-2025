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
        all_icaos = origins.union(dests)

        elevation_map = {}

        for icao in all_icaos:

            data = airport_data.get_airport_by_icao(icao)

            elevation_map[icao] = data[0]["elevation_ft"]

        # Application du mapping
        FuelSegment_X["elevation_origin_ft"] = FuelSegment_X["origin_icao"].map(
            elevation_map
        )
        FuelSegment_X["elevation_destination_ft"] = FuelSegment_X[
            "destination_icao"
        ].map(elevation_map)

        new_features = [
            "elevation_origin_ft",
            "elevation_destination_ft",
        ]
        column_functions["numerical"].extend(new_features)
        return FuelSegment_X, column_functions

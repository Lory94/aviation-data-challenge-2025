from .BaseFeatureEngineering import BaseFeatureEngineering


class AddAirportIcao_v0_0_0(BaseFeatureEngineering):

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
        new_features = [
            "origin_icao",
            "destination_icao",
        ]
        column_functions["categorical"].extend(new_features)

        return FuelSegment_X, column_functions

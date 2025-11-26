import numpy as np
import pandas as pd
from haversine import Unit, haversine
from tqdm import tqdm

from .BaseFeatureEngineering import BaseFeatureEngineering


def hv_distance(lat1, long1, lat2, long2, unit=Unit.FEET):
    """Compute the distance between two points (from their coordinates)"""
    return haversine((lat1, long1), (lat2, long2), unit=unit)


class AddDistanceAirports_v0_0_0(BaseFeatureEngineering):
    def __call__(
        self,
        FuelSegment_X,
        FuelSegment_Y,
        FlightList,
        Airport,
        Flight,
        column_functions,
    ):
        flightlist_with_distances = (
            FlightList[["flight_id", "origin_icao", "destination_icao"]]
            .merge(Airport.add_prefix("origin_"), on="origin_icao")
            .merge(Airport.add_prefix("destination_"), on="destination_icao")
            .assign(
                origin_destination_distance=lambda df: df.apply(
                    lambda row: haversine(
                        (row.origin_latitude, row.origin_longitude),
                        (row.destination_latitude, row.destination_longitude),
                        unit=Unit.FEET,
                    ),
                    axis=1,
                )
            )
        )

        FuelSegment_X = FuelSegment_X.merge(
            flightlist_with_distances[["flight_id", "origin_destination_distance"]],
            on="flight_id",
        )
        column_functions["numerical"].append("origin_destination_distance")

        return FuelSegment_X, column_functions

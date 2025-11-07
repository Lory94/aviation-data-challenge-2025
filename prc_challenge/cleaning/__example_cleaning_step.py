import pandas as pd

def __example_cleaning_step(FuelSegment_X, FlightList, Airport, Flight):
    return FuelSegment_X, FlightList, Airport, Flight

def remove_rows_with_missing_aircraft_type_in_FlightList(FuelSegment_X, FlightList, Airport, Flight):
    FlightList = FlightList.dropna(subset=["aircraft_type"])

    return FuelSegment_X, FlightList, Airport, Flight


def add_aircraft_type(FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight):

    FuelSegment_X = pd.merge(
        FuelSegment_X,
        FlightList[
            ["flight_id", "aircraft_type"]
        ],  # , "origin_icao", "destination_icao"]],
        on="flight_id",
        how="left",
    )

    return FuelSegment_X

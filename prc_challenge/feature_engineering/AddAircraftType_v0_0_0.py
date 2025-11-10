from .BaseFeatureEngineering import BaseFeatureEngineering


class AddAircraftType_v0_0_0(BaseFeatureEngineering):

    def __call__(self, FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight, column_functions):
        # Create the mapping Series: 'flight_id' -> 'aircraft_type' and apply it
        aircraft_map_series = FlightList.set_index('flight_id')['aircraft_type']
        FuelSegment_X['aircraft_type'] = FuelSegment_X['flight_id'].map(aircraft_map_series)
        column_functions["categorical"].append('aircraft_type')

        return FuelSegment_X, column_functions

from collections import defaultdict
from .BaseFeatureEngineering import BaseFeatureEngineering


class DropIfExists_v0_0_0(BaseFeatureEngineering):

    def __init__(self, columns):
        self.drop_columns = columns

    def __call__(self, FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight, column_functions):

        non_present_columns = [
            column for column in self.drop_columns
            if column not in FuelSegment_X.columns
        ]
        if len(non_present_columns) > 0:
            print(f"[DropIfExists] Attempted to drop columns {non_present_columns}, but they are not present.")
        
        FuelSegment_X = FuelSegment_X.drop(columns=[
            col for col in FuelSegment_X 
            if col in self.drop_columns
        ])

        # Correcting column_functions
        new_column_functions = defaultdict(list)
        for feature_type, columns in column_functions.items():
            for column in columns:
                if column not in self.drop_columns:
                    new_column_functions[feature_type].append(column)

        return FuelSegment_X, column_functions

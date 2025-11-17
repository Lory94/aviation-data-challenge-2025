import pandas as pd
from openap import prop
from typing import Dict, Any, Union, List


from .BaseFeatureEngineering import BaseFeatureEngineering


# Dictionary Flattening
class AddingAircraftFeatures(BaseFeatureEngineering):
    def __call__(self, FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight, column_functions):
        """
            Adds detailed, flattened aircraft-specific features to the fuel consumption DataFrame.

            Args:
                fuel_file_path: Path to the main fuel data Parquet file.
                flightlist_path: Path to the flight list Parquet file for mapping.

            Returns:
                A Pandas DataFrame (fuel data) enriched with aircraft characteristics.
        """
        NOT_AVAILABLE_AIRCRAFTS: tuple = ('MD11', 'B77L', 'A306')

        intermediate_aircraft_type = FlightList['aircraft_type'].str.upper()

        # Create the mapping Series: 'flight_id' -> 'aircraft_type'
        # Use the intermediate variable instead of modifying the DataFrame column directly.
        aircraft_map_series = FlightList.set_index('flight_id')[intermediate_aircraft_type.name]

        # Add the 'aircraft_type' column to the X segment.
        
        FuelSegment_X['aircraft_type'] = FuelSegment_X['flight_id'].map(aircraft_map_series)
        
        # 2. Prepare the Aircraft Parameters Mapping Table (df_params_map)

        # Get unique aircraft types from the fuel data (which now has 'aircraft_type').
        unique_aircraft_types = FuelSegment_X['aircraft_type'].dropna().unique().tolist()
        
        params_data: List[Dict[str, Any]] = []
        
    
        for ac_type in unique_aircraft_types:
            # Skip unavailable aircrafts.
            if ac_type in NOT_AVAILABLE_AIRCRAFTS:
                continue
                
            # Fetch nested parameters (conversion to lowercase assumed by 'prop' library).
            ac_params = prop.aircraft(ac_type.lower()) 
            
            # Flatten the parameters into a single dictionary.
            flat_params = flatten_aircraft_params(ac_params)
      
            #Removing useless features
            flat_params = {
                key: value
                for key, value in flat_params.items()
                if not key.startswith("engine_") and not key.startswith("clean_")
            }
        
            
            # Standardize the keys for consistency and the merge operation:
            # Set the join key.
            flat_params['aircraft_type'] = ac_type
            # Overwrite the 'aircraft' name (e.g., 'Airbus A320') with the standard ID ('A320').
            flat_params['aircraft'] = ac_type
            
            params_data.append(flat_params)
        
        # Create the small reference DataFrame from the list of flattened parameter dictionaries.
        if not params_data:
            return FuelSegment_X 
            
        df_params_map = pd.DataFrame(params_data)
        
        # 3. Join the detailed parameters back to the main DataFrame using pd.merge.
        # The left merge ensures all rows from the original fuel data are preserved.
        FuelSegment_X = pd.merge(
            FuelSegment_X,
            df_params_map,
            on="aircraft_type",
            how="left"
        )


        
        return FuelSegment_X, column_functions


def flatten_aircraft_params(params: Dict[str, Any]) -> Dict[str, Union[str, float, int, None]]:
    """
    Flattens a deeply nested dictionary into a single-level dictionary.

    Keys are concatenated using an underscore (e.g., 'drag_cd0'). This converts
    specifications into column names.

    Args:
        params: The nested dictionary containing aircraft specifications.

    Returns:
        A flattened dictionary with combined keys and terminal values.
    """
    flat_dict = {}

    def _flatten(data: Any, prefix: str = ""):
        """ Internal recursive function for dictionary flattening. """
        
        # Recurse for nested dictionaries.
        if isinstance(data, dict):
            for key, value in data.items():
                # Build the new key prefix.
                new_prefix = f"{prefix}_{key}" if prefix else key
                _flatten(value, new_prefix)
        
        # Skip lists or tuples to avoid complex column expansion.
        elif isinstance(data, (list, tuple)):
            return 
        
        # Store terminal values (int, float, str, None).
        else:
            final_key = prefix.lstrip('_')
            flat_dict[final_key] = data

    _flatten(params)
    return flat_dict

# --- Main Feature Engineering Function ---




if __name__ == '__main__':

    # --- Execution Example ---
    FLIGHT_LIST_PATH = '~/prc-challenge-2025/data/flightlist_train.parquet'
    FLIGHT_LIST_PATH = '../../../DB/flightlist_train.parquet'

    Flightlist = pd.read_parquet(FLIGHT_LIST_PATH)

    FuelSegment_X_path = '~/prc-challenge-2025/data/flights_train/prc770835414.parquet'
    FuelSegment_X_path = '../../../DB/flights_train/prc770835414.parquet'
    FuelSegment_X = pd.read_parquet(FuelSegment_X_path)
    df_feats_added, _ = AddingAircraftFeatures.__call__("",FuelSegment_X, pd.DataFrame, Flightlist, "airport", "flight", "")


    # Display the first few rows, showing the features in chunks of 5 columns.
    print("--- Enriched DataFrame Head (Pandas) ---")
    for i in range(0, len(df_feats_added.columns), 5):
        # Use iloc for slicing columns
        print(df_feats_added.iloc[:, i:i+5].head())
    



import pandas as pd
from .BaseFeatureEngineering import BaseFeatureEngineering

from openap import prop, FuelFlow
import matplotlib.pyplot as plt
class AddOpenapPred_v0_0_0(BaseFeatureEngineering):

    def __call__(self, FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight, column_functions):
            
        NOT_AVAILABLE_AIRCRAFTS: tuple = ('MD11', 'B77L', 'A306', 'A21N')

        unique_aircraft_types = FlightList["aircraft_type"].unique()
        
        mtow_map = {}
        for atype in unique_aircraft_types:
            if atype not in NOT_AVAILABLE_AIRCRAFTS:
                try:
                    mtow_map[atype] = prop.aircraft(atype)["mtow"]
                except Exception as e:
                    print(f"Alerte OpenAP pour MTOW de {atype}: {e}")
                    mtow_map[atype] = None
            else:
                mtow_map[atype] = None 

        predictions = pd.Series(index=FuelSegment_X.index, dtype=float)

        FuelSegment_X_copy = FuelSegment_X.copy()
        FuelSegment_X_copy['MTOW_kg'] = FuelSegment_X_copy['aircraft_type'].map(mtow_map)

        grouped_by_flight_id = FuelSegment_X_copy.groupby('flight_id')

        for flight_id, group in grouped_by_flight_id:
            ac_type_str = group["aircraft_type"].iloc[0]
            
            try:
                fuelflow_model = FuelFlow(ac=ac_type_str, use_synonym=True)
            except ValueError as e:
                print(f"Not available aircraft : {e}, skipping this computing")
                continue

            ff_kg_s_array = fuelflow_model.enroute(
                    alt=group["mean_altitude"],
                    tas=group["mean_TAS"],
                    vs=group["mean_vertical_rate"],
                    mass=group["MTOW_kg"]
                )
            
            predictions.loc[group.index] = ff_kg_s_array * group['segment_duration[seconds]']
            
        FuelSegment_X["openap_pred"] = predictions

        return FuelSegment_X, column_functions
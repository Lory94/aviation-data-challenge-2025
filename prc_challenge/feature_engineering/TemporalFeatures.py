import pandas as pd
import numpy as np

from .BaseFeatureEngineering import BaseFeatureEngineering



class TemporalFeatures(BaseFeatureEngineering):

    def _add_sincos_local_hour(self, trajectory, fuel):
        trajectory["timestamp"] = pd.to_datetime(trajectory["timestamp"])
        initial_columns = list(fuel.columns)
        fuel = pd.merge_asof(
        fuel.sort_values('start'),
        trajectory.sort_values('timestamp'),
        left_on='start',        # merge on datetime
        right_on='timestamp',
        by='flight_id',         # group by key before merging
        direction='nearest'   # choose the closest timestamp
        )
        fuel = fuel.assign(
            solar_timestamp=lambda df: df["timestamp"]
            + (df["longitude"] / 180 * 12).apply(pd.to_timedelta, unit="hours")
        )
        fuel["solar_hour"] = fuel["solar_timestamp"].dt.hour
        fuel["solar_hour"] = (fuel["solar_hour"].fillna(fuel['start'].dt.hour)).astype(int)
        return fuel[initial_columns+["solar_hour"]]

    def _add_day_of_week(self, fuel):
        fuel['day_of_week'] = fuel["start"].dt.dayofweek
        return fuel

    def _add_sincos_day_of_year(self, fuel):
        fuel['sin_day'] = np.sin(2 * np.pi * pd.to_datetime(fuel['start']).dt.dayofyear / 365)
        fuel['cos_day'] = np.cos(2 * np.pi * pd.to_datetime(fuel['start']).dt.dayofyear / 365)
        return fuel

    def _add_segment_duration(self, fuel):
        fuel["segment_duration"] =  (pd.to_datetime(fuel["end"])-pd.to_datetime(fuel["start"])).dt.seconds
        return fuel

    def add_temporal_features(self, trajectory, fuel):
        # Add the following temporal features to the data: sin_day, cos_day, segment_duration, solar_hour, day_of_week
        fuel["start"] = pd.to_datetime(fuel["start"])
        fuel = self._add_segment_duration(fuel)
        fuel = self._add_sincos_day_of_year(fuel)
        #fuel = self._add_sincos_local_hour(trajectory, fuel)
        fuel = self._add_day_of_week(fuel)
        return trajectory, fuel

    def __call__(self, FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight, column_functions):

        Flight, FuelSegment_X = self.add_temporal_features(Flight, FuelSegment_X)
        column_functions["numerical"].append('segment_duration')
        column_functions["numerical"].append('sin_day')
        column_functions["numerical"].append('cos_day')
        #column_functions["numerical"].append('solar_hour')
        column_functions["numerical"].append('day_of_week')
        return FuelSegment_X, column_functions
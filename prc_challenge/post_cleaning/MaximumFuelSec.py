import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .BasePostCleaning import BasePostCleaning


class MaximumFuelSec_v0_0_0(BasePostCleaning):

    def __init__(self, save: bool = False):
        self.method = None
        self.save = save

    def __call__(
        self,
        FuelSegment_train,
        FlightList_train,
        Predictions
    ):
        enhanced_flight = pd.merge(FuelSegment_train, FlightList_train,on="flight_id")
        enhanced_flight["segment_duration"] = (pd.to_datetime(enhanced_flight["end"])-pd.to_datetime(enhanced_flight["start"])).dt.seconds
        enhanced_flight["fuel_kg_sec"] = enhanced_flight["fuel_kg"]/enhanced_flight["segment_duration"]
        max_fuel_cons = enhanced_flight.groupby("aircraft_type")["fuel_kg_sec"].max().to_dict()
        # Aircrafts should consume more than 10% more than the max consumption in the testset and at least 0
        return Predictions.apply(lambda x: max(0,min(x.y_pred, 1.1*(max_fuel_cons[x.aircraft_type]*x["segment_duration[seconds]"]))),axis=1)

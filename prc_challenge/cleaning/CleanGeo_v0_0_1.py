import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from geopy.distance import geodesic

from .BaseCleaning import BaseCleaning


class CleanGeo_v0_0_1(BaseCleaning):

    def __init__(self, save: bool = False):
        self.method = None
        self.save = save

    def __call__(
        self,
        FuelSegment_X,
        FuelSegment_Y,
        FlightList,
        Airport,
        Flight,
    ):
        if "clean" not in str(Flight.directory):
            output_dir = Flight.directory + "_clean"
        else:
            output_dir = Flight.directory
        if self.save:
            # Create the dir if it doesn't exist to prevent OSErrors
            Path(output_dir).expanduser().mkdir(exist_ok=True)

        for flight_id in tqdm(Flight.flight_ids):
            file_path = Path(f"{output_dir}/{flight_id}.parquet").expanduser()
            if file_path.exists():
                continue
            traj = Flight[flight_id][
                [
                    "altitude",
                    "latitude",
                    "longitude",
                    "mach",
                    "groundspeed",
                    "vertical_rate",
                    "TAS",
                    "timestamp",
                ]
            ]
            traj["timestamp"] = pd.to_datetime(traj["timestamp"]) #
            condition = (
                (traj["altitude"] >= 0)
                & (traj["altitude"] <= 60000)
                & (traj["latitude"].between(-90, 90))
                & (traj["longitude"].between(-180, 180))
                & (traj["vertical_rate"].between(-6000, 6000))
            )
            traj = traj[condition]

            traj['latitude_prev'] = traj['latitude'].shift(1)
            traj['longitude_prev'] = traj['longitude'].shift(1)
            traj['timestamp_prev'] = traj['timestamp'].shift(1)
            traj['speed_kmh'] = traj.apply(compute_speed, axis=1)

            traj = traj[(traj['speed_kmh'].isna()) | (traj['speed_kmh'] < 1300)]


            traj["time_bin"] = (
                traj["timestamp"].astype("int64") // 5_000_000_000
            )  # 5 seconds in nanoseconds
            traj = (
                traj.groupby("time_bin")
                .agg(
                    {
                        "altitude": "mean",
                        "latitude": "mean",
                        "longitude": "mean",
                        "mach": "mean",
                        "groundspeed": "mean",
                        "vertical_rate": "mean",
                        "TAS": "mean",
                        "timestamp": "mean",
                    }
                )
                .reset_index(drop=True)
            )

            start = traj["timestamp"].min()
            end = traj["timestamp"].max()
            full_range = pd.date_range(start=start, end=end, freq="10s")

            traj_interp = pd.DataFrame(
                {
                    "timestamp": full_range,
                }
            )
            traj_interp = pd.concat([traj, traj_interp])
            traj_interp = traj_interp.sort_values("timestamp").set_index("timestamp")
            traj_interp = traj_interp.interpolate(
                method="time", limit_direction="forward"
            )
            traj_interp = traj_interp.reset_index()
            traj_interp["flight_id"] = flight_id
            if self.save:
                traj_interp.to_parquet(f"{output_dir}/{flight_id}.parquet", index=False)

        Flight.directory = output_dir
        return FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight


def compute_speed(row):
    if pd.isnull(row['latitude']) or pd.isnull(row['latitude_prev']):
        return np.nan
    dist_km = geodesic(
        (row['latitude_prev'], row['longitude_prev']),
        (row['latitude'], row['longitude'])
    ).km
    time_diff_sec = (row['timestamp'] - row['timestamp_prev']).total_seconds()
    if time_diff_sec == 0:
        return np.nan
    return dist_km / (time_diff_sec / 3600)


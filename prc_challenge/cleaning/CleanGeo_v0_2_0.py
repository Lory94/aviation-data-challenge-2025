import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from .BaseCleaning import BaseCleaning

# --- Helper: Vectorized Haversine ---
R_EARTH_KM = 6371.0088

def vectorized_haversine_speed(lat1, lon1, lat2, lon2, time_diff_sec):
    # Vectorized calculation (releases GIL, so it works well with threading)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist_km = R_EARTH_KM * c
    
    speed_kmh = np.divide(
        dist_km, 
        (time_diff_sec / 3600.0), 
        out=np.zeros_like(dist_km), 
        where=time_diff_sec != 0
    )
    return speed_kmh

def process_single_flight(flight_id, source_dir, output_dir, save_mode):
    # Construct paths
    save_path = Path(f"{output_dir}/{flight_id}.parquet").expanduser()
    
    if save_path.exists():
        return

    # Load data
    input_path = Path(f"{source_dir}/{flight_id}.parquet").expanduser()
    
    if not input_path.exists():
        return 

    # Read necessary columns
    cols = ["altitude", "latitude", "longitude", "mach", 
            "groundspeed", "vertical_rate", "TAS", "timestamp"]
    
    try:
        traj = pd.read_parquet(input_path, columns=cols)
    except Exception:
        return 

    traj["timestamp"] = pd.to_datetime(traj["timestamp"])

    # 1. Vectorized Filtering
    condition = (
        (traj["altitude"].between(0, 60000) | traj["altitude"].isna())
        & (traj["latitude"].between(-90, 90) | traj["latitude"].isna())
        & (traj["longitude"].between(-180, 180) | traj["longitude"].isna())
        & (traj["vertical_rate"].between(-6000, 6000) | traj["vertical_rate"].isna())

    )
    traj = traj[condition].copy()

    if traj.empty:
        return

    # 2. Vectorized Speed Calc
    lat = traj['latitude'].values
    lon = traj['longitude'].values
    ts = traj['timestamp'].values
    
    # Check for empty arrays after filter to prevent errors
    if len(lat) < 2:
        return

    lat_prev = np.roll(lat, 1)
    lon_prev = np.roll(lon, 1)
    ts_prev = np.roll(ts, 1)
    
    time_diff = (ts - ts_prev).astype('timedelta64[ms]').astype(float) / 1000.0
    speeds = vectorized_haversine_speed(lat_prev, lon_prev, lat, lon, time_diff)
    speeds[0] = 0 
    traj['speed'] = speeds
    
    traj = traj[(np.isnan(speeds)) | (speeds < 1300)]

    if traj.empty:
        return

    # 3. Resample & Interpolate
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
    if pd.notna(start) and pd.notna(end):
        full_range = pd.date_range(start=start, end=end, freq="10s")
        traj_interp = pd.DataFrame(
            {
                "timestamp": full_range,
            }
        )
        traj_interp = pd.concat([traj, traj_interp])
        traj_interp = traj_interp.sort_values("timestamp").set_index("timestamp")
        traj_interp = traj_interp.interpolate(method="time", limit_direction="forward")
        traj_interp.index.name = "timestamp"
        traj_interp = traj_interp.reset_index()
        traj_interp["flight_id"] = flight_id

        if save_mode:
            traj_interp.to_parquet(save_path, index=False)


class CleanGeo_v0_2_0(BaseCleaning):

    def __init__(self, save: bool = True):
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
        source_dir = str(Flight.directory)
        
        if "clean" not in source_dir:
            output_dir = source_dir + "_clean"
        else:
            output_dir = source_dir

        if self.save:
            Path(output_dir).expanduser().mkdir(exist_ok=True, parents=True)

        # n_jobs = max(1, cpu_count() - 1)
        n_jobs = 100
        
        print(f"Processing with {n_jobs} threads...")
        
        # --- FIXED: Added backend="threading" ---
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_single_flight)(
                fid, 
                source_dir, 
                output_dir, 
                self.save
            ) for fid in tqdm(Flight.flight_ids)
        )

        Flight.directory = output_dir
        return FuelSegment_X, FuelSegment_Y, FlightList, Airport, Flight
import numpy as np
import pandas as pd
from haversine import Unit, haversine
from tqdm import tqdm

from .BaseFeatureEngineering import BaseFeatureEngineering


def hv_distance(lat1, long1, lat2, long2, unit=Unit.FEET):
    """Compute the distance between two points (from their coordinates)"""
    return haversine((lat1, long1), (lat2, long2), unit=unit)


class AddTrajectoryFeatures_v0_0_1(BaseFeatureEngineering):

    def __call__(
        self,
        FuelSegment_X,
        FuelSegment_Y,
        FlightList,
        Airport,
        Flight,
        column_functions,
    ):
        """
        Aggregates trajectory features by iterating through flights, loading
        their specific trajectory data, and aggregating onto FuelSegment_X intervals.
        """

        # 1. Define Columns
        agg_features_cols = [
            "mean_altitude",
            "max_altitude",
            "mean_TAS",
            "max_TAS",
            "mean_groundspeed",
            "mean_vertical_rate",
            "n_points",
        ]
        delta_features_cols = ["altitude_change", "time_in_interval_s"]
        init_features_cols = [
            # altitude [ft] at the start of the fuel segment
            "init_altitude",
            # distance from the origin/destination airport at the start of the fuel
            # segment (not cumulative: depends too much on the data quality)
            "init_distance_origin",
            "init_distance_destination",
            # initial position of the fuel segment, as a fraction of the total flight
            # (computed as origin-current-destination)
            "init_position_fraction",
        ]
        all_new_cols = agg_features_cols + delta_features_cols + init_features_cols

        # 2.a. Prepare FuelSegment_X with a temporary ID for later merging
        FuelSegment_X = FuelSegment_X.copy()
        FuelSegment_X["interval_temp_id"] = FuelSegment_X.index

        # 2.b. Prepare coordinates of airports for each flight
        for ref in ("origin", "destination"):
            FlightList[[f"{ref}_latitude", f"{ref}_longitude"]] = FlightList.merge(
                Airport, left_on=f"{ref}_icao", right_on="icao"
            )[["latitude", "longitude"]]

        # List to store the calculated feature dataframes for each flight
        results_list = []

        # 3. Iterate over unique flights present in this segment batch
        unique_flight_ids = FuelSegment_X["flight_id"].unique()

        print(
            f"Processing trajectory features for {len(unique_flight_ids)} unique flights..."
        )

        for fid in tqdm(unique_flight_ids):
            try:
                # LOAD TRAJECTORY: Use the Flight class __getitem__ method
                # This loads the specific parquet file for this flight_id
                traj_df = Flight[fid].copy()
            except Exception as e:
                # Handle cases where file might be missing or corrupt
                continue

            if traj_df.empty:
                continue

            # Filter FuelSegment_X for only this flight
            try:
                segments_subset = FuelSegment_X[FuelSegment_X["flight_id"] == fid]
            except Exception as e:
                print("issue with id", fid)
                continue

            # 4.a. Merge Logic (Per Flight)
            # We merge the single flight's trajectory with its specific segments
            merged_traj = pd.merge(
                traj_df,
                segments_subset[["flight_id", "start", "end", "interval_temp_id"]],
                on="flight_id",
            )

            # Filter points strictly within the start/end window
            filtered_traj = merged_traj[
                (merged_traj["timestamp"] >= merged_traj["start"])
                & (merged_traj["timestamp"] <= merged_traj["end"])
            ].copy()

            if filtered_traj.empty:
                continue

            FlightList_row = FlightList[FlightList["flight_id"] == fid]

            # 5. Aggregation (Per Flight)
            filtered_traj.sort_values(
                by=["interval_temp_id", "timestamp"], inplace=True
            )
            gb = filtered_traj.groupby("interval_temp_id")

            # 5.a. Standard Aggregations
            agg_features = gb.agg(
                mean_altitude=("altitude", "mean"),
                max_altitude=("altitude", "max"),
                mean_TAS=("TAS", "mean"),
                max_TAS=("TAS", "max"),
                mean_groundspeed=("groundspeed", "mean"),
                mean_vertical_rate=("vertical_rate", "mean"),
                n_points=("timestamp", "count"),
            )

            # 5.b. Delta Features
            first_points = gb.first()
            last_points = gb.last()

            delta_features = pd.DataFrame(index=agg_features.index)
            delta_features["altitude_change"] = (
                last_points["altitude"] - first_points["altitude"]
            )
            delta_features["time_in_interval_s"] = (
                pd.to_datetime(last_points["timestamp"])
                - pd.to_datetime(first_points["timestamp"])
            ).dt.total_seconds()

            # 5.c. Init Features
            init_features = pd.DataFrame(index=agg_features.index)
            init_features["init_altitude"] = first_points["altitude"]
            init_features["init_distance_origin"] = first_points.apply(
                lambda row: hv_distance(
                    FlightList_row["origin_latitude"].iloc[0],
                    FlightList_row["origin_longitude"].iloc[0],
                    row["latitude"],
                    row["longitude"],
                ),
                axis=1,
            )
            init_features["init_distance_destination"] = first_points.apply(
                lambda row: hv_distance(
                    row["latitude"],
                    row["longitude"],
                    FlightList_row["destination_latitude"].iloc[0],
                    FlightList_row["destination_longitude"].iloc[0],
                ),
                axis=1,
            )
            init_features["init_position_fraction"] = init_features[
                "init_distance_origin"
            ] / (
                init_features["init_distance_origin"]
                + init_features["init_distance_destination"]
            )

            # 5.d. Combine for this flight and add to list
            combined_flight_features = pd.concat(
                [agg_features, delta_features, init_features], axis=1
            )
            results_list.append(combined_flight_features)

        # 6. Compile all results
        if results_list:
            all_features = pd.concat(results_list)

            # Merge back into the main dataframe using the index (interval_temp_id)
            FuelSegment_X = FuelSegment_X.merge(
                all_features, left_on="interval_temp_id", right_index=True, how="left"
            )
        else:
            print(
                "Warning: No features could be calculated (no matching trajectory data)."
            )
            for col in all_new_cols:
                FuelSegment_X[col] = np.nan

        # 7. Clean up NaN values (for segments that had no matching trajectory points)
        # Ensure columns exist even if merge failed entirely for some rows
        for col in all_new_cols:
            if col not in FuelSegment_X.columns:
                FuelSegment_X[col] = np.nan

        # Drop temp ID
        if "interval_temp_id" in FuelSegment_X.columns:
            FuelSegment_X.drop(columns=["interval_temp_id"], inplace=True)

        # 8. Update Metadata
        if "numerical" not in column_functions:
            column_functions["numerical"] = []
        column_functions["numerical"].extend(all_new_cols)

        return FuelSegment_X, column_functions

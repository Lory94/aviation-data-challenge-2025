import pandas as pd
from tqdm import tqdm
from haversine import Unit, haversine
from pathlib import Path
import numpy as np

from .BaseFeatureEngineering import BaseFeatureEngineering


def hv_distance(lat1, long1, lat2, long2, unit=Unit.FEET):
    """Compute the distance between two points (from their coordinates)"""
    return haversine((lat1, long1), (lat2, long2), unit=unit)


class AddGeoInfo_v0_0_3(BaseFeatureEngineering):

    def __init__(self, method):
        self.method = method
        self.directory = f"~/prc-challenge-2025/data/GeoFeature"
        Path(self.directory).expanduser().mkdir(exist_ok=True, parents=True)
        # method = [
        # "mean_altitude[ft]",
        # max_altitude[ft]
        # "range_altitude[ft]",
        # "delta_altitude[ft]",
        # "mean_TAS[kt]",
        # "max_TAS[kt]",
        # "mean_groundspeed[kt]",
        # "mean_vertical_rate[ft/min]",
        # "n_points",
        # "mean_wind_proxy",
        # "min_wind_proxy",
        # "max_wind_proxy",
        # "mean_acc_TAS",
        # "mean_acc_groundspeed",
        # "min_acc_groundspeed",
        # "max_acc_groundspeed",
        # "init_altitude",
        # "init_distance_origin",
        # "init_distance_destination",
        # "init_position_fraction"
        # ]

    def __call__(
        self,
        FuelSegment_X,
        FuelSegment_Y,
        FlightList,
        Airport,
        Flight,
        column_functions,
    ):
        FuelSegment_X = FuelSegment_X.copy()
        FuelSegment_X["interval_id"] = FuelSegment_X.index

        grouped_segments = FuelSegment_X.groupby("flight_id")

        for ref in ("origin", "destination"):
            FlightList[[f"{ref}_latitude", f"{ref}_longitude"]] = FlightList.merge(
                Airport, left_on=f"{ref}_icao", right_on="icao"
            )[["latitude", "longitude"]]

        all_flight_features = []
        for flight_id in tqdm(grouped_segments.groups.keys()):

            segment = grouped_segments.get_group(flight_id)
            FlightList_row = FlightList[FlightList["flight_id"] == flight_id].iloc[0]
            trajectory = Flight[flight_id]
            trajectory["wind_proxy"] = trajectory["TAS"] - trajectory["groundspeed"]
            trajectory["delta_t"] = trajectory["timestamp"].diff().dt.total_seconds()
            trajectory["delta_TAS"] = trajectory["TAS"].diff()
            trajectory["acceleration_TAS"] = (
                trajectory["delta_TAS"] / trajectory["delta_t"]
            )
            trajectory["delta_groundspeed"] = trajectory["groundspeed"].diff()
            trajectory["acceleration_groundspeed"] = (
                trajectory["delta_groundspeed"] / trajectory["delta_t"]
            )

            merged = self._assign_segment_id(segment, trajectory)
            if not merged.empty:
                features = self._get_geo_features(merged, FlightList_row)
                all_flight_features.append(features)

        all_flight_features = pd.concat(all_flight_features, ignore_index=True)
        FuelSegment_X = FuelSegment_X.merge(
            all_flight_features, on="interval_id", how="left"
        )
        FuelSegment_X.set_index("interval_id", inplace=True)
        FuelSegment_X.index.name = None
        #FuelSegment_X.fillna(0.0, inplace=True)

        for m in self.method:
            column_functions["numerical"].append(m)

        FuelSegment_X.to_csv(Path(self.directory).expanduser() / "features_v0_0_3.csv")
        return FuelSegment_X, column_functions

    def _assign_segment_id(self, segment, flight):
        merged = pd.merge(
            flight,
            segment[["flight_id", "start", "end", "interval_id"]],
            on="flight_id",
        )
        filtered_merged = merged[
            (merged["timestamp"] >= merged["start"])
            & (merged["timestamp"] <= merged["end"])
        ].copy()
        filtered_merged.sort_values(by=["interval_id", "timestamp"], inplace=True)
        return filtered_merged

    def _get_geo_features(self, df: pd.DataFrame, flight_info) -> pd.DataFrame:
        agg_dict = {}

        if "mean_altitude[ft]" in self.method:
            agg_dict["mean_altitude[ft]"] = ("altitude", "mean")
        if "max_altitude[ft]" in self.method:
            agg_dict["max_altitude[ft]"] = ("altitude", "max")
        if "range_altitude[ft]" in self.method:
            agg_dict["range_altitude[ft]"] = (
                "altitude",
                lambda x: x.max() - x.min() if len(x) > 1 else np.nan,
            )
        if "delta_altitude[ft]" in self.method:
            agg_dict["delta_altitude[ft]"] = (
                "altitude",
                lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else np.nan,
            )
        if "mean_TAS[kt]" in self.method:
            agg_dict["mean_TAS[kt]"] = ("TAS", "mean")
        if "max_TAS[kt]" in self.method:
            agg_dict["max_TAS[kt]"] = ("TAS", "max")
        if "mean_groundspeed[kt]" in self.method:
            agg_dict["mean_groundspeed[kt]"] = ("groundspeed", "mean")
        if "mean_vertical_rate[ft/min]" in self.method:
            agg_dict["mean_vertical_rate[ft/min]"] = ("vertical_rate", "mean")
        if "n_points" in self.method:
            agg_dict["n_points"] = ("altitude", "count")

        if "mean_wind_proxy" in self.method:
            agg_dict["mean_wind_proxy"] = ("wind_proxy", "mean")
        if "min_wind_proxy" in self.method:
            agg_dict["min_wind_proxy"] = ("wind_proxy", "min")
        if "max_wind_proxy" in self.method:
            agg_dict["max_wind_proxy"] = ("wind_proxy", "max")

        if "mean_acc_TAS" in self.method:
            agg_dict["mean_acc_TAS"] = ("acceleration_TAS", "mean")
        if "min_acc_TAS" in self.method:
            agg_dict["min_acc_TAS"] = ("acceleration_TAS", "min")
        if "max_acc_TAS" in self.method:
            agg_dict["max_acc_TAS"] = ("acceleration_TAS", "max")

        if "mean_acc_groundspeed" in self.method:
            agg_dict["mean_acc_groundspeed"] = ("acceleration_groundspeed", "mean")
        if "min_acc_groundspeed" in self.method:
            agg_dict["min_acc_groundspeed"] = ("acceleration_groundspeed", "min")
        if "max_acc_groundspeed" in self.method:
            agg_dict["max_acc_groundspeed"] = ("acceleration_groundspeed", "max")

        if "init_altitude" in self.method:
            agg_dict["init_altitude"] = ("altitude", "first")

        agg_dict["init_latitude"] = ("latitude", "first")
        agg_dict["init_longitude"] = ("longitude", "first")

        df = (
            df.groupby("interval_id")[
                [
                    "latitude",
                    "longitude",
                    "altitude",
                    "TAS",
                    "groundspeed",
                    "vertical_rate",
                    "wind_proxy",
                    "acceleration_TAS",
                    "acceleration_groundspeed",
                ]
            ]
            .agg(**agg_dict)
            .reset_index()
        )

        df["init_distance_origin"] = df.apply(
            lambda row: hv_distance(
                flight_info["origin_latitude"],
                flight_info["origin_longitude"],
                row["init_latitude"],
                row["init_longitude"],
            ),
            axis=1,
        )
        df["init_distance_destination"] = df.apply(
            lambda row: hv_distance(
                row["init_latitude"],
                row["init_longitude"],
                flight_info["destination_latitude"],
                flight_info["destination_longitude"],
            ),
            axis=1,
        )
        df["init_position_fraction"] = df["init_distance_origin"] / (
            df["init_distance_origin"] + df["init_distance_destination"]
        )
        return df

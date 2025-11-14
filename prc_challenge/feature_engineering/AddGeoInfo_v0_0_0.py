import pandas as pd
from tqdm import tqdm

from .BaseFeatureEngineering import BaseFeatureEngineering


class AddGeoInfo_v0_0_0(BaseFeatureEngineering):

    def __init__(self, method):
        self.method = method
        # method = ["mean_altitude[ft]", "range_altitude[ft]", "delta_altitude[ft]", "mean_TAS[kt]", "max_TAS[kt]", "mean_groundspeed[kt]", "mean_vertical_rate[ft/min]", "n_points"]

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

        all_flight_features = []
        for flight_id in tqdm(grouped_segments.groups.keys()):

            segment = grouped_segments.get_group(flight_id)
            merged = self._assign_segment_id(segment, Flight[flight_id])
            if not merged.empty:
                features = self._get_geo_features(merged)
                all_flight_features.append(features)

        all_flight_features = pd.concat(all_flight_features, ignore_index=True)
        FuelSegment_X = FuelSegment_X.merge(
            all_flight_features, on="interval_id", how="left"
        )
        FuelSegment_X.set_index("interval_id", inplace=True)
        FuelSegment_X.index.name = None
        FuelSegment_X.fillna(0.0, inplace=True)

        for m in self.method:
            column_functions["numerical"].append(m)
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

    def _get_geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        agg_dict = {}

        if "mean_altitude[ft]" in self.method:
            agg_dict["mean_altitude[ft]"] = ("altitude", lambda x: x.mean() if len(x) > 1 else 0.0)
        if "range_altitude[ft]" in self.method:
            agg_dict["range_altitude[ft]"] = ("altitude", lambda x: x.max() - x.min() if len(x) > 1 else 0.0)
        if "delta_altitude[ft]" in self.method:
            agg_dict["delta_altitude[ft]"] = ("altitude", lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0.0)
        if "mean_TAS[kt]" in self.method:
            agg_dict["mean_TAS[kt]"] = ("TAS", lambda x: x.mean() if len(x) > 1 else 0.0)
        if "max_TAS[kt]" in self.method:
            agg_dict["max_TAS[kt]"] = ("TAS", lambda x: x.max() if len(x) > 1 else 0.0)
        if "mean_groundspeed[kt]" in self.method:
            agg_dict["mean_groundspeed[kt]"] = ("groundspeed", lambda x: x.mean() if len(x) > 1 else 0.0)
        if "mean_vertical_rate[ft/min]" in self.method:
            agg_dict["mean_vertical_rate[ft/min]"] = ("vertical_rate", lambda x: x.mean() if len(x) > 1 else 0.0)
        if "n_points" in self.method:
            agg_dict["n_points"] = ("altitude", "count")

        return df.groupby("interval_id")[["altitude", "TAS", "groundspeed", "vertical_rate"]].agg(**agg_dict).reset_index()

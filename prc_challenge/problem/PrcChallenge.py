import pandas as pd
from sklearn.metrics import root_mean_squared_error
from ..data import SupervisedTabularDataset, TabularDataset
from ..utils.load_object_in_file import load_object_in_file
from ..utils.split_dataset import split_train_val
from sklearn.metrics import root_mean_squared_error
import datetime
import os
import json
from typing import Dict
import types
from ..data.assets import get_FuelSegment, get_Airport, get_FlightList, get_Flight


class PrcChallenge(object):

    def __init__(
        self,
        seed: int,
    ):
        
        train_FuelSegment = get_FuelSegment(variant="train")
        train_FlightList = get_FlightList(variant="train")
        self.Airport = get_Airport()
        self.Flight = get_Flight(variant="train")

        (self.train_FlightList, 
         self.valid_FlightList, 
         train_FuelSegment, 
         valid_FuelSegment,
        ) = split_train_val(
            train_frac=0.8, 
            flightlist=train_FlightList, 
            fuel=train_FuelSegment, 
            seed=seed,
        )
        self.seed = seed
        self.train_FuelSegment_X = train_FuelSegment.drop(columns=["fuel_kg"])
        self.train_FuelSegment_Y = train_FuelSegment["fuel_kg"]
        self.test_FuelSegment_X = valid_FuelSegment.drop(columns=["fuel_kg"])
        self.test_FuelSegment_Y = valid_FuelSegment["fuel_kg"]
        self.column_functions = {
            "timestamp": ["start", "end"],
            "incremental_identifier": ["idx"],
            "identifier": ["flight_id"],
            "numerical": [],
            "categorical": [],
        }

    @classmethod
    def load_config(cls, config: Dict) -> Dict:
        """_summary_

        Args:
            config (Dict): _description_

        Returns:
            Dict: _description_
        """

        return {
            "cleaning": [
                load_object_in_file(
                    file_path="../cleaning/__init__.py",
                    namespace="cleaning",
                    object_name=approach[0],
                )(**approach[1])
                for approach in config["cleaning"]
            ],
            "feature_engineering": [
                load_object_in_file(
                    file_path="../feature_engineering/__init__.py",
                    namespace="feature_engineering",
                    object_name=approach[0],
                )(**approach[1])
                for approach in config["feature_engineering"]
            ],
            "model": load_object_in_file(
                file_path="../model/__init__.py",
                namespace="model",
                object_name=config["model"][0],
            )(**config["model"][1]),
        }
    
    def solve_using(self, config):

        run_seed = self.seed
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
        model = f"{config['model'][0]}({config['model'][1]})"
        run_name = f"{run_seed}_{run_timestamp}_{model}"
        os.makedirs(f"../../results/{run_name}", exist_ok=True)

        # config contains the string version of the configuration of the run
        # loaded_config is its counterpart where associated objects are loaded and instanciated
        loaded_config = self.load_config(config)

        print("Available features before any processing:")
        print(self.train_FuelSegment_X.dtypes)
        print()

        for cleaning_step in loaded_config["cleaning"]:
            if not isinstance(cleaning_step, types.FunctionType) and hasattr(cleaning_step, "fit"):
                cleaning_step.fit(
                    self.train_FuelSegment_X, self.train_FuelSegment_Y, self.train_FlightList, self.Airport, self.Flight, **step_kwargs,
                )
            self.train_FuelSegment_X, self.train_FlightList, self.Airport, self.Flight = cleaning_step(
                self.train_FuelSegment_X, self.train_FlightList, self.Airport, self.Flight, **step_kwargs,
            )

        print("Available features after cleanup:")
        print(self.train_FuelSegment_X.dtypes)
        print()

        for feature_engineering_step in loaded_config["feature_engineering"]:
            feature_engineering_step.fit(
                self.train_FuelSegment_X, self.train_FuelSegment_Y, self.train_FlightList, self.Airport, self.Flight,
            )
            self.train_FuelSegment_X, self.column_functions = feature_engineering_step(
                self.train_FuelSegment_X, self.train_FuelSegment_Y, self.train_FlightList, self.Airport, self.Flight, self.column_functions,
            )

        print("Available features after feature engineering:")
        print(self.train_FuelSegment_X.dtypes)
        print()

        loaded_config["model"].fit(self.train_FuelSegment_X, self.train_FuelSegment_Y, self.column_functions)

        evaluation = self.evaluate(
            loaded_config["model"],
        )

        with open(f"../../results/{run_name}/config.json", 'w') as fp:
            json.dump(config, fp, indent=4)
        with open(f"../../results/{run_name}/evaluation.json", 'w') as fp:
            json.dump(evaluation, fp, indent=4)

        return model, evaluation

    def evaluate(
        self,
        model,
    ):

        metrics = {}

        # Train RMSE
        y_pred = model.predict(self.train_FuelSegment_X)
        y_true = self.train_FuelSegment_Y
        metrics["mse(test)"] = root_mean_squared_error(y_pred=y_pred, y_true=y_true)

        # Test RMSE
        y_pred = model.predict(self.test_FuelSegment_X)
        y_true = self.test_FuelSegment_Y
        metrics["rmse(test)"] = root_mean_squared_error(y_pred=y_pred, y_true=y_true)

        return metrics

    def build_submission(
        self, 
        model, 
        ranking_dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        assert set(ranking_dataframe.columns) == {
            "idx",
            "flight_id",
            "start",
            "end",
            "fuel_kg",
        }
        ranking_x = ranking_dataframe.drop(columns=["fuel_kg"])
        y_pred = model.predict(ranking_x)
        return ranking_x.assign(fuel_kg=y_pred)

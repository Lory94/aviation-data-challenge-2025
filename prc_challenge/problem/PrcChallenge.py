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
from collections import Counter


class PrcChallenge(object):

    def __init__(
        self,
        seed: int=1,
        train_frac: float=0.8,
    ):

        FuelSegment = get_FuelSegment(variant="train")
        FlightList = get_FlightList(variant="train")
        self.Airport = get_Airport()
        self.Flight = get_Flight(variant="train")

        (
            self.train_FlightList,
            self.valid_FlightList,
            train_FuelSegment,
            valid_FuelSegment,
        ) = split_train_val(
            train_frac=train_frac, 
            flightlist=FlightList, 
            fuel=FuelSegment, 
            seed=seed,
        )
        self.seed = seed
        self.train_FuelSegment_X = train_FuelSegment.drop(columns=["fuel_kg"])
        self.train_FuelSegment_Y = train_FuelSegment["fuel_kg"]
        self.valid_FuelSegment_X = valid_FuelSegment.drop(columns=["fuel_kg"])
        self.valid_FuelSegment_Y = valid_FuelSegment["fuel_kg"]
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
            "post_cleaning": [
                load_object_in_file(
                    file_path="../post_cleaning/__init__.py",
                    namespace="post_cleaning",
                    object_name=approach[0],
                )(**approach[1])
                for approach in config["post_cleaning"]
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
        self.run_name = f"{run_seed}_{run_timestamp}_{model}"
        os.makedirs(f"../../results/{self.run_name}", exist_ok=True)

        # config contains the string version of the configuration of the run
        # loaded_config is its counterpart where associated objects are loaded and instanciated
        self.loaded_config = self.load_config(config)

        print("Available features before any processing:")
        print(self.train_FuelSegment_X.dtypes)
        print()

        for cleaning_step in self.loaded_config["cleaning"]:
            print(f"Applying on the data the cleaning {cleaning_step.__class__.__name__}.")
            if not isinstance(cleaning_step, types.FunctionType) and hasattr(
                cleaning_step, "fit"
            ):
                cleaning_step.fit(
                    self.train_FuelSegment_X,
                    self.train_FuelSegment_Y,
                    self.train_FlightList,
                    self.Airport,
                    self.Flight,
                )
            (
                self.train_FuelSegment_X,
                self.train_FuelSegment_Y,
                self.train_FlightList,
                self.Airport,
                self.Flight,
            ) = cleaning_step(
                self.train_FuelSegment_X,
                self.train_FuelSegment_Y,
                self.train_FlightList,
                self.Airport,
                self.Flight,
            )

        print("Available features after cleanup:")
        print(self.train_FuelSegment_X.dtypes)
        print()

        for feature_engineering_step in self.loaded_config["feature_engineering"]:
            print(f"Processing the data with {feature_engineering_step.__class__.__name__}.")
            self.train_FuelSegment_X, self.column_functions = feature_engineering_step(
                self.train_FuelSegment_X,
                self.train_FuelSegment_Y,
                self.train_FlightList,
                self.Airport,
                self.Flight,
                self.column_functions,
            )

            # Checks
            declared_columns = []
            for feature_type, columns in self.column_functions.items():
                declared_columns += columns
            ## Checking the last feature engineering step didn't produce a double in column_functions
            if len(declared_columns) != len(set(declared_columns)):
                duplicates = [
                    column_name
                    for column_name, count in Counter(declared_columns).items()
                    if count > 1
                ]
                raise Exception(f"{feature_engineering_step.__class__.__name__} seems to have added to column_functions some columns which were already there: {duplicates}.")
            ## Checking the feature engineering step didn't produce a feature not declared in column_functions
            undeclared_columns = [
                column_name
                for column_name in self.train_FuelSegment_X.columns
                if column_name not in declared_columns
            ]
            if len(undeclared_columns) > 0:
                raise Exception(f"{feature_engineering_step.__class__.__name__} seems to have created a new columns without declaring it in column_functions: {undeclared_columns}")
 

        print("Available features after feature engineering:")
        print(self.train_FuelSegment_X.dtypes)
        print()

        print("Starting training...")
        self.loaded_config["model"].fit(
            self.train_FuelSegment_X, self.train_FuelSegment_Y, self.column_functions
        )
        print("Finished training.")

        print("Starting validation evaluation...")
        evaluation = self.evaluate(
            self.loaded_config["model"],
        )
        print("Finished validation evaluation.")

        with open(f"../../results/{self.run_name}/config.json", "w") as fp:
            json.dump(config, fp, indent=4)
        with open(f"../../results/{self.run_name}/evaluation.json", "w") as fp:
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
        metrics["rmse(train)"] = root_mean_squared_error(y_pred=y_pred, y_true=y_true)

        # Test RMSE
        for cleaning_step in self.loaded_config["cleaning"]:
            (
                self.valid_FuelSegment_X,
                _,
                self.valid_FlightList,
                self.Airport,
                self.Flight,
            ) = cleaning_step(
                self.valid_FuelSegment_X,
                None,
                self.valid_FlightList,
                self.Airport,
                self.Flight,
            )
        for feature_engineering_step in self.loaded_config["feature_engineering"]:
            self.valid_FuelSegment_X, _ = feature_engineering_step(
                self.valid_FuelSegment_X,
                None,
                self.valid_FlightList,
                self.Airport,
                self.Flight,
                self.column_functions,
            )
        y_pred = model.predict(self.valid_FuelSegment_X)
        y_true = self.valid_FuelSegment_Y
        metrics["rmse(valid)"] = root_mean_squared_error(y_pred=y_pred, y_true=y_true)

        # Building submission file
        print("Starting submission file creation...")
        self.build_submission_with_trained_model()
        print("Finished submission file creation.")

        return metrics

    def build_submission_with_trained_model(self) -> pd.DataFrame:

        FuelSegment = get_FuelSegment(variant="rank")
        FlightList = get_FlightList(variant="rank")
        Airport = get_Airport()
        Flight = get_Flight(variant="rank")

        FuelSegment_train = get_FuelSegment(variant="train")
        FlightList_train = get_FlightList(variant="train")
        
        FuelSegment_X = FuelSegment.drop(columns=["fuel_kg"])

        for cleaning_step in self.loaded_config["cleaning"]:
            FuelSegment_X, _, FlightList, Airport, Flight = cleaning_step(
                FuelSegment_X,
                None,
                FlightList,
                Airport,
                Flight,
            )
        for feature_engineering_step in self.loaded_config["feature_engineering"]:
            FuelSegment_X, _ = feature_engineering_step(
                FuelSegment_X,
                None,
                FlightList,
                Airport,
                Flight,
                self.column_functions,
            )

        y_pred = self.loaded_config["model"].predict(FuelSegment_X)

        FuelSegment_X["y_pred"] = y_pred
        for post_cleaning_step in self.loaded_config["post_cleaning"]:
            FuelSegment_X["y_pred"] = post_cleaning_step(
                FuelSegment_train,
                FlightList_train,
                FuelSegment_X
            )
        y_pred = FuelSegment_X["y_pred"]

        FuelSegment.drop(columns=["fuel_kg"]).assign(fuel_kg=y_pred).to_parquet(
            f"../../results/{self.run_name}/submission_with_trained_model.parquet"
        )

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


class PrcChallenge(object):

    def __init__(
        self,
        supervised_fuel_dataset: SupervisedTabularDataset,
        enrichment_dataset: TabularDataset,
        seed: int,
    ):

        train_flights, valid_flights, train_fuel, valid_fuel = split_train_val(
            0.8, enrichment_dataset, supervised_fuel_dataset, seed=seed,
        )
        self.seed = seed
        self.train_fuel_X = train_fuel.drop(columns=["fuel_kg"])
        self.train_fuel_Y = train_fuel["fuel_kg"]
        self.test_fuel_X = valid_fuel.drop(columns=["fuel_kg"])
        self.test_fuel_Y = valid_fuel["fuel_kg"]
        self.train_enrichment = train_flights
        self.test_enrichment = valid_flights

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
        run_timestamp = datetime.datetime.now().strftime()
        model = f"{config['model'][0]}({config['model'][1]})"
        run_name = f"{run_seed}_{run_timestamp}_{model}"

        loaded_config = self.load_config(config)

        train_fuel_X, train_enrichment = self.train_fuel_X, self.train_enrichment

        for cleaning_step in loaded_config["cleaning"]:
            train_fuel_X, train_enrichment = cleaning_step(
                train_fuel_X, train_enrichment,
            )

        for feature_engineering_step in loaded_config["feature_engineering"]:
            train_fuel_X, train_enrichment = feature_engineering_step(
                train_fuel_X, train_enrichment,
            )

        model = config["model"].fit(train_fuel_X, train_enrichment)

        evaluation = self.evaluate(model)

        os.makedirs("../../results/{run_name}", exist_ok=True)
        with open("../../results/{run_name}/config.json", 'w') as fp:
            json.dump(config, fp)
        with open("../../results/{run_name}/evaluation.json", 'w') as fp:
            json.dump(evaluation, fp)

        return model, evaluation

    def evaluate(
        self,
        model,
    ):

        metrics = {}

        # Train RMSE
        y_pred = model.predict(self.train_fuel_X)
        y_true = self.train_fuel_Y
        metrics["mse(test)"] = root_mean_squared_error(y_pred=y_pred, y_true=y_true)

        # Test RMSE
        y_pred = model.predict(self.test_fuel_X)
        y_true = self.test_fuel_Y
        metrics["rmse(test)"] = root_mean_squared_error(y_pred=y_pred, y_true=y_true)

        return metrics

    def predict_for_submission(
        self, model, ranking_dataframe: pd.DataFrame
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

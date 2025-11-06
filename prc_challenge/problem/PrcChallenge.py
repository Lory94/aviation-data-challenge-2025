import pandas as pd
from sklearn.metrics import root_mean_squared_error
from ..data import SupervisedTabularDataset, TabularDataset
from ..utils.load_object_in_file import load_object_in_file
from ..utils.split_dataset import split_train_val
from .SupervisedRegression import SupervisedRegression
from sklearn.metrics import root_mean_squared_error


class PrcChallenge(SupervisedRegression):

    def __init__(
        self,
        supervised_fuel_dataset: SupervisedTabularDataset,
        enrichment_dataset: TabularDataset,
    ):

        train_flights, valid_flights, train_fuel, valid_fuel = split_train_val(
            0.8, enrichment_dataset, supervised_fuel_dataset
        )
        self.train_fuel_X = train_fuel.drop(columns=["fuel_kg"])
        self.train_fuel_Y = train_fuel["fuel_kg"]
        self.test_fuel_X = valid_fuel.drop(columns=["fuel_kg"])
        self.test_fuel_Y = valid_fuel["fuel_kg"]
        self.train_enrichment = train_flights
        self.test_enrichment = valid_flights

    @classmethod
    def load_config(cls, config):

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

        train_fuel_X, train_enrichment = self.train_fuel_X, self.train_enrichment

        for cleaning_step in config["cleaning"]:
            train_fuel_X, train_enrichment = cleaning_step(
                train_fuel_X, train_enrichment
            )

        for feature_engineering_step in config["feature_engineering"]:
            train_fuel_X, train_enrichment = feature_engineering_step(
                train_fuel_X, train_enrichment
            )

        model = config["model"].fit(train_fuel_X, train_enrichment)
        return model

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

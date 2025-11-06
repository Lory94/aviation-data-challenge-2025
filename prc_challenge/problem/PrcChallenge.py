from ..data import SupervisedTabularDataset, TabularDataset
from ..utils.split_dataset import split_train_val
from .SupervisedRegression import SupervisedRegression


class PrcChallenge(SupervisedRegression):

    def __init__(
        self,
        supervised_fuel_dataset: SupervisedTabularDataset,
        enrichment_dataset: TabularDataset,
    ):

        train_flights, valid_flights, train_fuel, valid_fuel = split_train_val(
            0.8, enrichment_dataset, supervised_fuel_dataset
        )
        self.train_x = train_fuel.drop(columns=["fuel_kg"])
        self.train_y = train_fuel["fuel_kg"]
        self.test_x = valid_fuel.drop(columns=["fuel_kg"])
        self.test_y = valid_fuel["fuel_kg"]

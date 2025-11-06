from .SupervisedRegression import SupervisedRegression
from ..data import SupervisedTabularDataset, TabularDataset


class PrcChallenge(SupervisedRegression):

    def __init__(
        self,
        supervised_fuel_dataset: SupervisedTabularDataset,
        enrichment_dataset: TabularDataset,
    ):
        
        train_flights, valid_flights = TODO
        self.train_x, self.test_x, self.train_y, self.test_y = TODO

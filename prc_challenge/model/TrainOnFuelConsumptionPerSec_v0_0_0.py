import pandas as pd

from prc_challenge.utils import load_object_in_file


class TrainOnFuelConsumptionPerSec_v0_0_0:
    """A wrapper model that feeds fuel consumption rate in [kg/s] as the target to a
    submodel and post-processes the infered data to get fuel consumption in [kg]."""

    def __init__(self, **kwargs) -> None:
        """
        kwargs:
            model: the model to train and use
            model_params: the parameters to pass to the model
        """
        self.kwargs = kwargs
        if "model" not in kwargs:
            raise ValueError(f"A model must be passed to {self.__class__.__name__!r}")
        self.model_name = kwargs["model"]
        self.model_params = kwargs.get("model_params", {})

        self.model = load_object_in_file(
            "../model/__init__.py", namespace="model", object_name=self.model_name
        )(**self.model_params)

    def fit(self, X, y, column_functions):
        # Pre-process the target: kg -> kg/s
        y = y.reset_index(drop=True)
        segment_durations = (X["end"] - X["start"]) / pd.Timedelta(seconds=1)
        y = y / segment_durations
        assert not y.isna().any()

        self.model.fit(X=X, y=y, column_functions=column_functions)

    def predict(self, X):
        predicted = self.model.predict(X)

        # Post-process the predictions: kg/s -> kg
        segment_durations = (X["end"] - X["start"]) / pd.Timedelta(seconds=1)
        return predicted * segment_durations

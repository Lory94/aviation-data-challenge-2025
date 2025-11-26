import numpy as np

from prc_challenge.utils import load_object_in_file


class AverageEnsemble_v0_0_0:
    """A wrapper model that feeds fuel consumption rate in [kg/s] as the target to a
    submodel and post-processes the infered data to get fuel consumption in [kg]."""

    def __init__(self, **kwargs) -> None:
        """
        kwargs:
            models: the [model, model_kwargs]s to train and use
        """
        self.kwargs = kwargs
        if "models" not in kwargs:
            raise ValueError(f"A 'models' argument must be passed to {self.__class__.__name__!r}")

        self.models = [
            load_object_in_file(
                "../model/__init__.py", 
                namespace="model", 
                object_name=model,
            )(**model_kwargs)
            for model, model_kwargs in kwargs["models"]
        ]

    def fit(self, X, y, column_functions):

        for model in self.models:
            model.fit(X, y, column_functions)

    def predict(self, X):
        predictions = [
            model.predict(X)
            for model in self.models
        ]  # List of 1D arrays
        return np.mean(predictions, axis=0)

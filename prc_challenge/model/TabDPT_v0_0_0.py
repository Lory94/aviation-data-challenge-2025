from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder
from tabdpt import TabDPTRegressor


class TabDPT_v0_0_0(object):
    """
    A wrapper for the TabDPT model

    git clone git@github.com:layer6ai-labs/TabDPT.git
    cd TabDPT-inference
    uv sync
    uv pip install -e .
    comment out line 47 in TabDPT-inference/src/tabdpt/regressor.py

    Running this with this config took 90 minutes
    problem = PrcChallenge(seed=1, train_frac=0.99)

    config = {
        "cleaning": [["CleanGeo_v0_0_0", {"save": True}]],  #  need to save the first time
        "feature_engineering": [
            ["AddAircraftType_v0_0_0", {}],
            ["AddingAircraftFeatures", {}],
            ["AddFuelSegmentDuration_v0_0_0", {}],
            ["AddTimeSinceTakeOff_fromFlightList_v0_0_0", {}],
            ["TemporalFeatures", {}],
            [
                "AddGeoInfo_v0_0_0",
                {
                    "method": [
                        "mean_altitude[ft]",
                        "range_altitude[ft]",
                        "delta_altitude[ft]",
                        "mean_TAS[kt]",
                        "max_TAS[kt]",
                        "mean_groundspeed[kt]",
                        "mean_vertical_rate[ft/min]",
                        "n_points",
                    ]
                },
            ],
            ["AddTrajectoryFeatures_v0_0_1", {}],
            ["DropIfExists_v0_0_0", {"columns": ["idx", "flight_id"]}],
        ],
        "model": ["TabDPT_v0_0_0", {}],
    }
    """

    def __init__(self, **kwargs):
        """
        Initializes the TabDPT_v0_0_0 model.
        Kwargs are passed to the TabDPTRegressor constructor.
        Special kwargs for predict ('n_ensembles', 'context_size', 'seed') are also handled here.
        """
        self.kwargs = kwargs
        # Set defaults for memory-intensive parameters to avoid OOM errors.
        # These can be overridden by the user's configuration.
        self.kwargs.setdefault("inf_batch_size", 128)

        # Capture predict-time arguments from kwargs.
        # These are removed from self.kwargs so they are not passed to TabDPTRegressor constructor.
        self.predict_kwargs = {
            "n_ensembles": self.kwargs.pop("n_ensembles", 2),
            "context_size": self.kwargs.pop("context_size", 1024),
            "seed": self.kwargs.pop("seed", None),
        }
        self.preprocessor = None
        self.model = None

    def fit(self, X, y, column_functions):
        """
        Fits the model, including preprocessing of the data.
        """

        numerical_transformer = SimpleImputer(strategy="mean")

        categorical_transformer = SklearnPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, column_functions["numerical"]),
                ("cat", categorical_transformer, column_functions["categorical"]),
            ],
            remainder="drop",
        )

        # Fit preprocessor and transform X
        X_processed = self.preprocessor.fit_transform(X, y)

        # Initialize and fit the TabDPT model
        self.model = TabDPTRegressor(**self.kwargs)
        self.model.fit(X_processed, y.to_numpy())

        return self

    def predict(self, X):
        """
        Makes predictions on new data.
        """
        if not self.model or not self.preprocessor:
            raise RuntimeError("The model has not been fitted yet.")

        # Transform X using the fitted preprocessor
        X_processed = self.preprocessor.transform(X)

        # Predict using the TabDPT model
        return self.model.predict(X_processed, **self.predict_kwargs)

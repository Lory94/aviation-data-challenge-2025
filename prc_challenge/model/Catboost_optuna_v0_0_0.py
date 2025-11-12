import datetime
import sys

import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


class Catboost_v0_0_0(object):
    """
    A wrapper for CatBoost that integrates Optuna hyperparameter tuning
    into a Scikit-Learn compatible Pipeline framework.
    """

    def __init__(self, **kwargs):
        """
        kwargs:
            group_col (str): Name of the column in X to use for GroupShuffleSplit.
            n_trials (int): Number of Optuna trials.
            study_name_prefix (str): Prefix for the Optuna study.
            storage (str): Database URL for Optuna.
            device_type (str): "cpu" or "gpu".
            to_load (str): Name of an existing study to resume.
        """
        self.kwargs = kwargs
        self.pipeline = None

        # Extract config from kwargs with defaults
        self.group_col = kwargs.get("group_col", None)
        self.n_trials = kwargs.get("n_trials", 1)  # Reduced default for safety
        self.study_name_prefix = kwargs.get("study_name_prefix", "CatBoostModel")
        self.storage = kwargs.get("storage", "sqlite:///fuel_models.sqlite3")
        self.device_type = kwargs.get("device_type", "cpu")
        self.to_load = kwargs.get("to_load", None)

    def fit(self, X, y, column_functions):

        # 1. Define Preprocessing
        def to_string(x):
            return x.astype(str)

        preprocessing = ColumnTransformer(
            [
                (
                    "num_imputer",
                    SimpleImputer(strategy="mean"),
                    column_functions["numerical"],
                ),
                (
                    "cat_imputer",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant", fill_value="MISSING"
                                ),
                            ),
                            ("caster", FunctionTransformer(to_string, validate=False)),
                        ]
                    ),
                    column_functions["categorical"],
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        print("Preprocessing data for tuning...")
        # Fit/Transform X once to use for the Optuna process
        X_processed = preprocessing.fit_transform(X, y)

        cat_features = column_functions["categorical"]
        cat_features = [c for c in cat_features if c in X_processed.columns]

        # 2. Extract Groups for Splitting (if configured)
        groups = None
        if self.group_col and self.group_col in X.columns:
            groups = X[self.group_col]

        # 3. Run Optuna Tuning
        best_params = self._run_optuna(X_processed, y, groups, cat_features)

        # 4. Build Final Pipeline
        print("\nFitting final model with best parameters...")

        final_model = CatBoostRegressor(
            **best_params,
            task_type="GPU" if self.device_type == "gpu" else "CPU",
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            logging_level="Silent",
            cat_features=cat_features,
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("model", final_model),
            ]
        )

        # Fit the final pipeline on the whole dataset
        self.pipeline.fit(X, y)

        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def _run_optuna(self, X, y, groups, categorical_features):
        """
        Internal method to run the Optuna search.
        X here is already preprocessed (imputed).
        """
        print("Splitting data for tuning (internal train/validation split)...")

        # Setup Split logic
        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.9, random_state=42)
            try:
                train_idx, val_idx = next(gss.split(X, y, groups=groups))
            except ValueError:
                print(
                    "Warning: Not enough groups. Falling back to random split.",
                    file=sys.stderr,
                )
                train_idx, val_idx = self._get_random_split(X)
        else:
            print("No groups provided. Using random split.")
            train_idx, val_idx = self._get_random_split(X)

        if len(val_idx) == 0 and len(train_idx) > 0:
            print("No validation samples created. Using training set as validation.")
            val_idx = train_idx

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(
            f"Internal split: {len(X_train)} train samples, {len(X_val)} val samples."
        )

        train_pool = Pool(X_train, y_train, cat_features=categorical_features)
        val_pool = Pool(X_val, y_val, cat_features=categorical_features)

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 500, 2000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", 0.0, 1.0
                ),
                "random_strength": trial.suggest_float(
                    "random_strength", 1e-9, 10.0, log=True
                ),
            }

            model = CatBoostRegressor(
                **params,
                task_type="GPU" if self.device_type == "gpu" else "CPU",
                loss_function="RMSE",
                eval_metric="RMSE",
                random_seed=42,
                logging_level="Silent",
                early_stopping_rounds=100,
            )

            model.fit(train_pool, eval_set=val_pool)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))

            # Set user attr for retrieval of optimal iterations later
            trial.set_user_attr("best_iteration", model.get_best_iteration())

            return rmse

        print(
            f"\n--- Starting Optuna Hyperparameter Search ({self.device_type.upper()}) ---"
        )

        if self.to_load:
            study = optuna.load_study(study_name=self.to_load, storage=self.storage)
            print(f"Loaded existing study '{self.to_load}'")
        else:
            study_name = f"{self.study_name_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

            pruner = optuna.pruners.MedianPruner(n_warmup_steps=20, n_startup_trials=20)

            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                storage=self.storage,
                load_if_exists=True,
                pruner=pruner,
            )

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        print("\nOptuna search finished.")
        print(f"Best Tuning (Validation) RMSE: {study.best_value:.4f}")

        best_params = study.best_params
        # Use the specific iteration count where early stopping occurred
        if "best_iteration" in study.best_trial.user_attrs:
            best_params["iterations"] = study.best_trial.user_attrs["best_iteration"]

        return best_params

    def _get_random_split(self, X):
        if len(X) > 1:
            indices = np.arange(len(X))
            return train_test_split(indices, test_size=0.2, random_state=42)
        else:
            return [0], []

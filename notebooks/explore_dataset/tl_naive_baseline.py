# # Flight Fuel Consumption Prediction Pipeline
#
# This file loads flight, fuel, airport, and trajectory data.
#
# The goal is to train a SINGLE global CatBoost model for all aircraft types
# and evaluate its performance on unseen data.
#
# This script can:
# 1. Train a model from "train" data.
# 2. Immediately use that model to generate a prediction file for "rank" data.
# Airport elevation features are not used in this version.

import datetime
import glob
import os
import sys
import time
from typing import Optional, Tuple, Dict, Any
import warnings

import numpy as np
from catboost import CatBoostRegressor, Pool
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# connect to optuna dashboard via "uv run optuna-dashboard sqlite:///fuel_models.sqlite3"

# Suppress Optuna's trial pruning warning for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)
# Suppress Pandas warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings(
    "ignore",
    message="A value is trying to be set on a copy of a slice from a DataFrame.",
)


def load_data():
    """Loads all TRAINING parquet files and performs initial type conversion."""
    print("Loading TRAINING data...")
    base_dir = ""  # ../../
    try:
        # Load the single-file datasets
        flights_df = pd.read_parquet(f"{base_dir}data/flightlist_train.parquet")
        fuel_df = pd.read_parquet(f"{base_dir}data/fuel_train.parquet")
        airports_df = pd.read_parquet(f"{base_dir}data/apt.parquet")
        print("Loaded flight list (train), fuel (train), and airport data.")

        # --- Load all trajectory files from the 'flights_train' directory ---
        traj_path = f"{base_dir}data/flights_train"
        if not os.path.isdir(traj_path):
            raise FileNotFoundError(f"Trajectory directory not found: '{traj_path}'.")

        all_traj_files = glob.glob(os.path.join(traj_path, "*.parquet"))
        if not all_traj_files:
            raise FileNotFoundError(f"No .parquet files found in '{traj_path}'.")

        print(f"Found {len(all_traj_files)} train trajectory files. Loading...")

        # Using a generator for potentially lower memory usage during list creation
        traj_dfs = (pd.read_parquet(f) for f in all_traj_files)
        traj_df = pd.concat(traj_dfs, ignore_index=True)
        print("Train trajectory data concatenated.")

        # Data Preprocessing and Cleaning
        print("Preprocessing timestamps...")
        flights_df["takeoff"] = pd.to_datetime(flights_df["takeoff"])
        flights_df["landed"] = pd.to_datetime(flights_df["landed"])
        traj_df["timestamp"] = pd.to_datetime(traj_df["timestamp"])
        fuel_df["start"] = pd.to_datetime(fuel_df["start"])
        fuel_df["end"] = pd.to_datetime(fuel_df["end"])

        # Drop flights with missing aircraft_type, as it's key for our analysis
        flights_df = flights_df.dropna(subset=["aircraft_type"])
        return flights_df, traj_df, fuel_df, airports_df

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during data loading: {e}", file=sys.stderr)
        print("You may need to install a parquet engine", file=sys.stderr)
        sys.exit(1)


def load_ranking_data(
    base_dir: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads all RANKING (submission) parquet files."""
    print("Loading RANKING data for submission...")
    try:
        # Load the single-file datasets
        flights_df = pd.read_parquet(f"{base_dir}data/flightlist_rank.parquet")
        fuel_df_submission = pd.read_parquet(
            f"{base_dir}data/fuel_rank_submission.parquet"
        )
        airports_df = pd.read_parquet(f"{base_dir}data/apt.parquet")
        print("Loaded flight list (rank), fuel submission template, and airport data.")

        # Load all trajectory files from the 'flights_rank' directory
        traj_path = f"{base_dir}data/flights_rank"
        if not os.path.isdir(traj_path):
            raise FileNotFoundError(f"Trajectory directory not found: '{traj_path}'.")

        all_traj_files = glob.glob(os.path.join(traj_path, "*.parquet"))
        if not all_traj_files:
            raise FileNotFoundError(f"No .parquet files found in '{traj_path}'.")

        print(f"Found {len(all_traj_files)} rank trajectory files. Loading...")

        traj_dfs = (pd.read_parquet(f) for f in all_traj_files)
        traj_df = pd.concat(traj_dfs, ignore_index=True)
        print("Rank trajectory data concatenated.")

        # Data Preprocessing and Cleaning
        print("Preprocessing timestamps...")
        flights_df["takeoff"] = pd.to_datetime(flights_df["takeoff"])
        flights_df["landed"] = pd.to_datetime(flights_df["landed"])
        traj_df["timestamp"] = pd.to_datetime(traj_df["timestamp"])
        fuel_df_submission["start"] = pd.to_datetime(fuel_df_submission["start"])
        fuel_df_submission["end"] = pd.to_datetime(fuel_df_submission["end"])

        # Don't drop NAs here, need to predict on everything
        return flights_df, traj_df, fuel_df_submission, airports_df

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure 'flightlist_rank.parquet', ", file=sys.stderr)
        print("'fuel_rank_submission.parquet', and 'flights_rank/' ", file=sys.stderr)
        print("directory exist in 'data/'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during data loading: {e}", file=sys.stderr)
        sys.exit(1)


def create_features(
    flights_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
    airports_df: pd.DataFrame,
    is_prediction: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, list]:
    """
    Engineers a model-ready feature set (X) and target (y) from the raw data.

    If `is_prediction` is True, it will NOT drop rows with missing trajectory
    data, which is crucial for building the submission file.
    """
    print("Engineering features...")
    if is_prediction:
        print("Mode: PREDICTION (will not drop rows with missing trajectory)")
    else:
        print("Mode: TRAINING (will drop rows with missing trajectory)")

    # 1: Create base table from fuel data
    data = fuel_df.copy()

    # 2: Add flight-level features
    data = pd.merge(
        data,
        flights_df[
            ["flight_id", "aircraft_type"]
        ],  # , "origin_icao", "destination_icao"]],
        on="flight_id",
        how="left",
    )

    # # 3: Add airport features (elevation)
    # airports_df_simple = airports_df[["icao", "elevation"]].rename(
    #     columns={"elevation": "origin_elevation"}
    # )
    # data = pd.merge(
    #     data, airports_df_simple, left_on="origin_icao", right_on="icao", how="left"
    # )
    # airports_df_simple = airports_df_simple.rename(
    #     columns={"origin_elevation": "dest_elevation"}
    # )
    # data = pd.merge(
    #     data,
    #     airports_df_simple,
    #     left_on="destination_icao",
    #     right_on="icao",
    #     how="left",
    # )

    # 4: Engineer interval-based trajectory features
    if traj_df.empty:
        print("Warning: Trajectory DataFrame is empty. Features will be sparse.")
        # Create empty columns for features so the function doesn't crash
        agg_features_cols = [
            "mean_altitude",
            "max_altitude",
            "mean_TAS",
            "max_TAS",
            "mean_groundspeed",
            "mean_vertical_rate",
            "n_points",
        ]
        delta_features_cols = ["altitude_change", "time_in_interval_s"]
        for col in agg_features_cols + delta_features_cols:
            data[col] = np.nan
    else:
        try:
            import duckdb

            print("Using DuckDB for efficient trajectory join...")
            data["interval_id"] = data.index
            data_to_join = data[["flight_id", "start", "end", "interval_id"]]
            query = """
            SELECT t.*, d.interval_id
            FROM traj_df AS t
            JOIN data_to_join AS d
                ON t.flight_id = d.flight_id
                AND t.timestamp BETWEEN d.start AND d.end
            """
            filtered_traj = duckdb.query(query).to_df()

        except ImportError:
            print("Warning: DuckDB not found. Falling back to slow pandas merge.")
            data["interval_id"] = data.index
            merged_traj = pd.merge(
                traj_df,
                data[["flight_id", "start", "end", "interval_id"]],
                on="flight_id",
            )
            filtered_traj = merged_traj[
                (merged_traj["timestamp"] >= merged_traj["start"])
                & (merged_traj["timestamp"] <= merged_traj["end"])
            ].copy()

        # 5: Aggregate trajectory features by interval
        print("Aggregating trajectory features...")
        if not filtered_traj.empty:
            filtered_traj.sort_values(by=["interval_id", "timestamp"], inplace=True)

            gb = filtered_traj.groupby("interval_id")
            agg_features = gb.agg(
                mean_altitude=("altitude", "mean"),
                max_altitude=("altitude", "max"),
                mean_TAS=("TAS", "mean"),
                max_TAS=("TAS", "max"),
                mean_groundspeed=("groundspeed", "mean"),
                mean_vertical_rate=("vertical_rate", "mean"),
                n_points=("timestamp", "count"),
            )
            first_points = gb.first()
            last_points = gb.last()

            delta_features = pd.DataFrame(index=agg_features.index)
            delta_features["altitude_change"] = (
                last_points["altitude"] - first_points["altitude"]
            )
            delta_features["time_in_interval_s"] = (
                last_points["timestamp"] - first_points["timestamp"]
            ).dt.total_seconds()

            data = data.merge(agg_features, on="interval_id", how="left")
            data = data.merge(delta_features, on="interval_id", how="left")
        else:
            print(
                "Warning: No matching trajectory points found for any fuel intervals."
            )
            # Create empty columns
            agg_features_cols = [
                "mean_altitude",
                "max_altitude",
                "mean_TAS",
                "max_TAS",
                "mean_groundspeed",
                "mean_vertical_rate",
                "n_points",
            ]
            delta_features_cols = ["altitude_change", "time_in_interval_s"]
            for col in agg_features_cols + delta_features_cols:
                data[col] = np.nan

    # 6: Add simple features
    data["interval_duration_s"] = (data["end"] - data["start"]).dt.total_seconds()

    # 7. Final Processing
    print("Finalizing dataset...")

    # --- MODIFICATION ---
    # Moved all filtering logic into the 'if not is_prediction' block.
    # For prediction mode, we must NOT drop any rows.

    if not is_prediction:
        # For training, we only want to learn from valid, complete data.

        # 'fuel_kg' will be present in training (actuals) and submission (0.0s)
        data = data.dropna(subset=["fuel_kg"])

        # --- Debugging line ---
        pre_duration_drop_count = len(data)

        data = data[data["interval_duration_s"] > 0]

        # --- Debugging line ---
        post_duration_drop_count = len(data)
        if pre_duration_drop_count > 0 and post_duration_drop_count == 0:
            print(
                "WARNING: All rows were dropped because 'interval_duration_s' was <= 0.",
                file=sys.stderr,
            )

        # For training, we only want to learn from intervals
        # where we have trajectory data.
        data = data.dropna(subset=["n_points"])

    # For prediction, we keep ALL rows, and the fillna(0)
    # below will handle the missing trajectory data.
    # -----------------------------

    data["aircraft_type"] = data["aircraft_type"].fillna("UNKNOWN").astype("category")

    features = [
        "interval_duration_s",
        "aircraft_type",
        "origin_elevation",
        "dest_elevation",
        "mean_altitude",
        "max_altitude",
        "mean_TAS",
        "max_TAS",
        "mean_groundspeed",
        "mean_vertical_rate",
        "n_points",
        "altitude_change",
        "time_in_interval_s",
    ]
    categorical_features = ["aircraft_type"]
    target = "fuel_kg"

    # Ensure all feature columns exist, even if empty
    for col in features:
        if col not in data.columns:
            data[col] = np.nan
            if col == "aircraft_type":
                data[col] = data[col].astype("category")

    numeric_features = [f for f in features if f not in categorical_features]
    data[numeric_features] = (
        data[numeric_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    X = data[features]
    y = data[target]
    g_groups = data["flight_id"]

    print(f"Dataset finalized. Total samples: {len(X)}")
    return X, y, g_groups, categorical_features


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    categorical_features: list,
    study_name_prefix: str = "CatBoostModel",
    n_trials: int = 50,
    to_load: Optional[str] = None,
    device_type: str = "cpu",
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Uses Optuna to find the best CatBoost hyperparameters.
    This function splits X/y into its *own* internal train/validation set
    for the purpose of tuning.

    Returns:
        study_name (str): The name of the completed Optuna study.
        best_rmse (float): The best *validation* RMSE achieved during tuning.
        best_params (dict): The best hyperparameter dictionary.
    """

    print("Splitting data for tuning (internal train/validation split)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.6, random_state=42)

    try:
        train_idx, val_idx = next(gss.split(X, y, groups=groups))
    except ValueError:
        print(
            "Warning: Not enough groups to split. Falling back to random split.",
            file=sys.stderr,
        )
        if len(X) > 1:
            indices = np.arange(len(X))
            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, random_state=42
            )
        else:
            print("Only 1 sample. Cannot split. Using for training only.")
            train_idx, val_idx = [0], []

    if len(val_idx) == 0 and len(train_idx) > 0:
        print("No validation samples created. Using training set as validation.")
        val_idx = train_idx

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_flights = set(groups.iloc[train_idx])
    val_flights = set(groups.iloc[val_idx])
    print(
        f"Internal split: {len(X_train)} train samples ({len(train_flights)} flights), "
        f"{len(X_val)} val samples ({len(val_flights)} flights)."
    )

    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, y_val, cat_features=categorical_features)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-9, 10.0, log=True
            ),
        }
        model = CatBoostRegressor(
            **params,
            task_type="GPU" if device_type == "gpu" else "CPU",
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            logging_level="Silent",
            early_stopping_rounds=100,
        )
        model.fit(train_pool, eval_set=val_pool)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    print(f"\n--- Starting Optuna Hyperparameter Search ({device_type.upper()}) ---")
    storage_name = "sqlite:///fuel_models.sqlite3"
    # study_name_prefix = X["aircraft_type"].iloc[0] if not X.empty else "UNKNOWN_AC"

    if to_load:
        study = optuna.load_study(study_name=to_load, storage=storage_name)
        print(f"Loaded existing study '{to_load}' with {len(study.trials)} trials.")
    else:
        study_name = (
            f"CatBoost Fuel {study_name_prefix} {datetime.datetime.now().isoformat()}"
        )
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
        )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nOptuna search finished.")
    print(f"Best Tuning (Validation) RMSE: {study.best_value:.4f}")

    # Add final model params
    best_params = study.best_params
    best_params["iterations"] = study.best_trial.last_step  # Use optimal iterations

    return study.study_name, study.best_value, best_params


def run_training_pipeline(
    n_trials_per_model: int, device_type: str = "cpu"
) -> Optional[Tuple[CatBoostRegressor, list]]:
    """
    Main pipeline to load, split, tune, and test ONE GLOBAL model.

    Returns:
        The trained CatBoostRegressor model and list of categorical features
        if successful, otherwise None, None.
    """

    # 1. Load all data
    try:
        flights, traj, fuel, airports = load_data()
    except Exception as e:
        print(f"Failed to load data. Exiting. Error: {e}", file=sys.stderr)
        return None, None

    # 2. Create the "Truly Unseen" Holdout Test Set
    print("\n" + "=" * 50)
    print("Creating Train/Test split by flight_id...")
    gss_main = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    pool_idx, test_idx = next(gss_main.split(flights, groups=flights["flight_id"]))

    flights_pool = flights.iloc[pool_idx]
    flights_test = flights.iloc[test_idx]

    pool_flight_ids = flights_pool["flight_id"].unique()
    test_flight_ids = flights_test["flight_id"].unique()

    # Filter all other dataframes based on this split
    traj_pool = traj[traj["flight_id"].isin(pool_flight_ids)]
    fuel_pool = fuel[fuel["flight_id"].isin(pool_flight_ids)]

    traj_test = traj[traj["flight_id"].isin(test_flight_ids)]
    fuel_test = fuel[fuel["flight_id"].isin(test_flight_ids)]

    print(
        f"Data split: {len(flights_pool)} flights in Training Pool, "
        f"{len(flights_test)} flights in Holdout Test Set."
    )
    print("=" * 50)

    # 3. --- PHASE 1: TUNING ---
    print("\n" + "#" * 50)
    print("PHASE 1: TUNING SINGLE GLOBAL MODEL")
    print("#" * 50)

    print("🚀 --- Tuning for ALL Aircraft Types --- 🚀")
    start_time_ac = time.perf_counter()

    # 3a. Engineer features from ALL POOL data
    try:
        # Note: is_prediction=False (default)
        X_pool, y_pool, g_pool, cat_features = create_features(
            flights_pool, traj_pool, fuel_pool, airports
        )
    except Exception as e:
        print(f"Error during feature engineering for ALL data: {e}", file=sys.stderr)
        return None, None  # Exit if we can't even build the features

    # 3b. Check for valid data
    if X_pool.empty or y_pool.empty:
        print("No valid data found after feature engineering. Exiting.")
        return None, None

    print(f"Tuning model on {len(X_pool)} total samples...")

    # 3c. Tune model
    try:
        study_name, best_rmse, best_params = tune_model(
            X_pool,
            y_pool,
            g_pool,
            cat_features,
            study_name_prefix="GLOBAL_MODEL",
            n_trials=n_trials_per_model,
            device_type=device_type,
        )
        duration_ac = time.perf_counter() - start_time_ac
        print(f"--- ✅ Finished Tuning. Best Tuning RMSE: {best_rmse:.4f} ---")
        print(f"--- Time taken: {duration_ac:.2f} seconds ---")

    except Exception as e:
        print(f"Error during model tuning: {e}", file=sys.stderr)
        return None, None  # Can't continue if tuning failed

    # 4. --- PHASE 2: FINAL TESTING ---
    print("\n\n" + "#" * 50)
    print("      PHASE 2: FINAL TESTING ON HOLDOUT SET")
    print("#" * 50)

    if best_params is None:
        print("No valid parameters found from tuning. Exiting.")
        return None, None

    # --- We already have the full training data (X_pool, y_pool)
    #     So we just need to create the test set
    print("Creating features for unseen holdout test set...")

    try:
        # Note: is_prediction=False (default)
        X_test, y_test, _, _ = create_features(
            flights_test, traj_test, fuel_test, airports
        )
    except Exception as e:
        print(f"Error creating test set: {e}", file=sys.stderr)
        return None, None

    if X_test.empty or y_test.empty:
        print("No valid test samples found after feature engineering. Exiting.")
        return None, None

    # --- Train Final Model ---
    print(f"Training final model on {len(X_pool)} pool samples...")
    # We need `cat_features` from the pool data engineering step
    train_pool_full = Pool(X_pool, y_pool, cat_features=cat_features)

    final_model = CatBoostRegressor(
        **best_params,
        task_type="GPU" if device_type == "gpu" else "CPU",
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        logging_level="Silent",
    )

    # Note: No early stopping here, we use the iteration count from tuning.
    final_model.fit(train_pool_full)

    # --- Evaluate on Test Set ---
    print(f"Evaluating on {len(X_test)} unseen test samples...")
    test_preds = final_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    # 5. Report final summary
    print("\n" + "=" * 80)
    print("🏁 FINAL RESULTS SUMMARY (HOLDOUT TEST) 🏁")
    print("=" * 80)
    print(f" ✈️  Aircraft Types: ALL (Single Global Model)")
    print(f" 📊 Tuning (Validation) RMSE: {best_rmse:.4f}")
    print(f" 🧪 Final (Holdout Test) RMSE: {test_rmse:.4f}")
    print(f" 📈 Samples (Pool / Test): {len(X_pool)} / {len(X_test)}")
    print("=" * 80)

    # Return the trained model and the feature list
    return final_model, cat_features


def get_dummy_model(cat_features_list: list) -> CatBoostRegressor:
    """Creates and fits a dummy model for testing the prediction pipeline."""
    print("Creating dummy model for testing...")
    # Create minimal fake data that matches the features
    X_dummy = pd.DataFrame(
        {
            "interval_duration_s": [600, 1200],
            "aircraft_type": ["A320", "B738"],
            "origin_elevation": [100, 500],
            "dest_elevation": [500, 100],
            "mean_altitude": [30000, 35000],
            "max_altitude": [31000, 35000],
            "mean_TAS": [400, 450],
            "max_TAS": [410, 460],
            "mean_groundspeed": [420, 430],
            "mean_vertical_rate": [0, 0],
            "n_points": [10, 12],
            "altitude_change": [0, 0],
            "time_in_interval_s": [600, 1200],
        }
    )
    y_dummy = pd.Series([150, 250])

    # Ensure categorical feature is set correctly
    X_dummy["aircraft_type"] = X_dummy["aircraft_type"].astype("category")

    dummy_model = CatBoostRegressor(
        iterations=1,
        random_seed=42,
        logging_level="Silent",
    )

    # Find the index of the categorical feature
    cat_features_indices = [
        i for i, col in enumerate(X_dummy.columns) if col in cat_features_list
    ]

    dummy_model.fit(X_dummy, y_dummy, cat_features=cat_features_indices)
    print("Dummy model created and fitted.")
    return dummy_model


def run_prediction_only(team_name: str, version: int):
    """Runs only the submission creation part with a dummy model."""
    print("--- RUNNING IN PREDICTION TEST MODE ---")

    # We need the list of categorical features. We'll hardcode it
    # based on create_features.
    cat_features_list = ["aircraft_type"]

    # 1. Get a dummy model
    #    Alternatively, you could try to load a real one:
    #    try:
    #        dummy_model = CatBoostRegressor().load_model("my_final_model.cbm")
    #        print("Loaded real model from 'my_final_model.cbm'")
    #    except:
    #        print("Could not load real model, creating dummy.")
    dummy_model = get_dummy_model(cat_features_list)

    # 2. Run submission creation
    create_submission_file(
        trained_model=dummy_model,
        team_name=team_name,
        version=version,
    )


def create_submission_file(
    trained_model: CatBoostRegressor,
    team_name: str,
    version: int,
):
    """
    Loads the ranking data, generates predictions using the
    trained model, and saves the submission file.
    """
    print("\n" + "#" * 50)
    print("   PHASE 3: CREATING SUBMISSION FILE")
    print("#" * 50)

    # 1. Load ranking data
    try:
        flights_rank, traj_rank, fuel_submission, airports_static = load_ranking_data()
    except Exception as e:
        print(f"Could not load ranking data: {e}", file=sys.stderr)
        return

    # 2. Engineer features from ranking data
    #    IMPORTANT: Set is_prediction=True
    try:
        X_rank, _, _, _ = create_features(
            flights_rank,
            traj_rank,
            fuel_submission,
            airports_static,
            is_prediction=True,
        )
    except Exception as e:
        print(f"Could not engineer features for ranking data: {e}", file=sys.stderr)
        return

    if X_rank.empty:
        print(
            "Feature engineering for ranking data produced no samples.", file=sys.stderr
        )
        return

    print(f"Generating predictions for {len(X_rank)} submission samples...")

    # 3. Generate predictions
    predictions = trained_model.predict(X_rank)

    # 4. Post-process predictions (e.g., ensure no negative fuel)
    predictions[predictions < 0] = 0

    # 5. Create final submission DataFrame
    #    The `fuel_submission` DataFrame has the exact format we need.
    #    Since `create_features` (with is_prediction=True) preserves
    #    the index, we can just assign the 'fuel_kg' column.
    submission_df = fuel_submission.copy()
    submission_df["fuel_kg"] = predictions

    # 6. Save to parquet
    filename = f"{team_name}_v{version}.parquet"
    try:
        submission_df.to_parquet(filename, index=False)
        print("\n" + "=" * 80)
        print(f"✅ SUCCESSFULLY CREATED SUBMISSION FILE: {filename}")
        print(submission_df.head())
        print("=" * 80)
    except Exception as e:
        print(f"Failed to save submission file '{filename}': {e}", file=sys.stderr)


if __name__ == "__main__":
    # --- Configuration ---
    MODE = "FULL_PIPELINE"  # Train and then predict
    # MODE = "PREDICTION_TEST"  # Only test the prediction step

    N_TRIALS_PER_MODEL = 20  # 50-100 provides better tuning. 20 is a quick test.
    DEVICE = "cpu"  # "cpu" or "gpu"
    TEAM_NAME = "my_awesome_team"
    SUBMISSION_VERSION = 1

    pipeline_start_time = time.perf_counter()

    if MODE == "FULL_PIPELINE":
        print(
            f"Starting FULL PIPELINE with {N_TRIALS_PER_MODEL} trials on {DEVICE.upper()}."
        )
        # --- 1. Run Training ---
        final_model, training_cat_features = run_training_pipeline(
            n_trials_per_model=N_TRIALS_PER_MODEL, device_type=DEVICE
        )

        # --- 2. Create Submission (if training was successful) ---
        if final_model is not None:
            create_submission_file(
                trained_model=final_model,
                team_name=TEAM_NAME,
                version=SUBMISSION_VERSION,
            )
        else:
            print("Training pipeline failed. No model was trained.", file=sys.stderr)
            print("Submission file was NOT created.", file=sys.stderr)

    elif MODE == "PREDICTION_TEST":
        print("Starting PREDICTION TEST MODE.")
        run_prediction_only(
            team_name=TEAM_NAME,
            version=SUBMISSION_VERSION,
        )

    # --- 3. Report Total Time ---
    pipeline_end_time = time.perf_counter()
    total_duration = pipeline_end_time - pipeline_start_time
    print("\n" + "=" * 60)
    print(f"Total pipeline execution time: {total_duration / 60:.2f} minutes")
    print("=" * 60)

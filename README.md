Reminder: The README should allow anyone to jump into the project within a few minutes. Keeping it updated is an ABSOLUTE PRIORITY.

# PRC Challenge 2025

This repository contains the codebase of Euranova's contribution to Eurocontrol's 2025 challenge.

## Getting started as a contributor

Use uv.
If you need the explicit creation of a virtual environment, for example to run notebooks in VS Code, run within the directory which contains this README:

```bash
uv venv
```

You can now select the environment related to the project in VS Code.

### Accessing the data

```bash
export ACCESS_KEY=xxxxxxxxxxxxxxxx
export SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
bash load_data.sh
```

### Adding a new dependency

```bash
uv add package_name
```

If everything goes right, you'll just have to make other devs aware that they need to pull and check that what they work on is still working.

If uv can't manage to find a proper version of the package because of current dependencies, see with the others if the blocking packages requirements can be relaxed.

### Splitting the data

Split the data with the `split.py` script, e.g. for an 80-20 split with the default seed (0) and target location (`split/`):

```python
python scripts/split.py 0.8
```

Run

```python
python scripts/split.py -h
```

to check the options

## Run the code

A submission file can be generated running the notebook : `notebooks/pipeline_run_demo/pipeline_run_demo.ipynb`

In the notebook the desired configuration should be chosen.For each step, cleaning, feature_engineering, post_cleaning and model you can select with steps to run. This is an example of a configuration file:

```python
config = {
    "cleaning": [
        ["CleanGeo_v0_0_0", {}]
    ],
    "feature_engineering": [
        ["AddAircraftType_v0_0_0", {}],
        ["AddingAircraftFeatures", {}],
        ["AddFuelSegmentDuration_v0_0_0", {}],
        ["AddTimeSinceTakeOff_fromFlightList_v0_0_0", {}],
        ["TemporalFeatures", {}],
        ["AddGeoInfo_v0_0_0", {"method":["mean_altitude[ft]", "range_altitude[ft]", "delta_altitude[ft]", "mean_TAS[kt]", "max_TAS[kt]", "mean_groundspeed[kt]", "mean_vertical_rate[ft/min]", "n_points"]}],
        ["DropIfExists_v0_0_0", {"columns": ["idx", "flight_id"]}],
    ],
    "model": ["Catboost_v0_0_0", {}],
}
```
Once the notebook is executed, the submission file can be found in `reults/model_name/submission_with_trained_model.parquet`

## Architecture

Our solution is based on 3 steps, and uses Catboost as the predicted model (the hyperparameter tuning is performed with Optuna):

- cleaning: which removes from the training and rank set outlier values, such as non realistic latitude or longitudes

- feature_engineering: which add different new features to the model:
    - AddAircraftType_v0_0_0: adds the aircraft type;
    - AddingAircraftFeatures: adds aircraft type informations such as MTOW and aerodynamic information retrived from openap;
    - AddFuelSegmentDuration_v0_0_0: adds the segment duration in seconds;
    - AddTimeSinceTakeOff_fromFlightList_v0_0_0: adds the seconds passed from the take off time;
    - TemporalFeatures: adds temporal feature information such as the day of the week;
    - AddGeoInfo_v0_0_0: adds trajectory information such as the mean_altitude or the mean_vertical_rate
    - DropIfExists_v0_0_0: removes the flight_id from the features of the model.


- post_cleaning: which clips the the predictions by setting the maximum consumption per minute to 10% more of the maximum consumption experienced in the training set


## Submission

Competition runs in two phases:

- phase 1 until 9 Nov 2025 (23:59)
- phase 2 from 10 Nov to 30 Nov 2025

### Phase 1

Ranking based on `fuel_rank_submission.parquet`: file available, with columns `idx`, `flight_id`, `start` and `end` before `fuel_kg`. Column `fuel_kg` is set to 0. The values must be replaced with our predictions (for *every* row).

The file must be named `<team-name>_v<incremental integer>.parquet` and submitted to our S3 bucket (using Minio: see Accessing the data, above, and [last year's challenge instructions](https://ansperformance.eu/study/data-challenge/dc2024/data.html#using-minio-client)).

To use Minio do the following:

```bash
brew install minio/stable/mc ## install Minio
mc alias set dc25 https://s3.opensky-network.org/ ACCESS_KEY SECRET_KEY ## Setup access

mc cp outspoken-tornado_v@.parquet dc25/prc-2025-outspoken-tornado/ ## submit
mc ls dc25/prc-2025-outspoken-tornado ## list all submissions
mc cp --recursive dc25/prc-2025-outspoken-tornado prc-2025-outspoken-tornado_submissions ## download all submissions
```

### Phase 2

Ranking based on `fuel_final_submission.parquet` (**one** submission allowed).


# Submissions

Euranova team: outspoken-tornado

## Phase 1 submissions

Results can be found at [this url](https://datacomp.opensky-network.org/api/competitions/71c49292-6139-425f-803a-52ee8730ba58/leaderboard?limit=50&teamName=outspoken-tornado).

Sub version | Description | Score
---|---|---
1 | Thibault: avg trajectory features, CatBoost, fine-tuning by Optuna (20 iter), trained on 70% of the train set | 277.1618
2 | Thibault: avg trajectory features, CatBoost, fine-tuning by Optuna (100 iter), trained on 100% of the train set (80/20 split) | 256.2298
3 | Thibault: avg trajectory features, CatBoost, fine-tuning by Optuna (150 iter), trained on 100% of the train set (99/1 split) | 246.4086
4 | Thomas: avg trajectory features + distance features, CatBoost, fine-tuning by Optuna, trained on 100% of the train set (80/20 split) | 264.1956
5 | Thomas: avg trajectory features + distance features, CatBoost, fine-tuning by Optuna, trained on 100% of the train set (99/1 split) | 253.3667
6 | ? | 245.8032
7 | Thomas: fixed trajectory features + fixed time since takeoff, CatBoost+Optuna, 99/1 split | 227.2324
8 | Thomas: 7 without GeoFeatures, CatBoost+Optuna, 99/1 split | 231.3615
9 | Thibault: same config as 7, TabDPT, trained on 100% of the train set (99/1 split) | 230.3649
10 | Lucas: same config as 7 + openAP predictions on 100% of the train set (99/1 split) with 1 Optuna's trial | 237.666
11 | Mathilde: Fix aircraftFeatures, without TrajectoryFeatures, GeoInfo_v0_0_0 (80/20 split) with 1 Optuna's trial | 227.2522
12 | Mathilde: Fix aircraftFeatures, without TrajectoryFeatures, GeoInfo_v0_0_0 (99/1 split) with 1 Optuna's trial | 237.9804
13 | ? | 225.0856
14 | ? | 247.8558
15 | Mathilde: GeoInfo_v0_0_2 (80/20 split) with 1 Optuna's trial | 235.6303
16 | Mathilde: GeoInfo_v0_0_3 (80/20 split) with 1 Optuna's trial | 227.332
17 | ? | 297.8765
18 | Lucas: config with every feature (Add_geo_info_V3 and cleanGeo_V0) on 100% of the train set (99/1 split) with 160 Optuna's trial (including 40 warm-up trials) | 303.5219
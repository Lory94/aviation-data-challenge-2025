import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from prc_challenge.utils.split_dataset import split_train_val


def main(train_frac: float, target: str, seed: int = 0):
    assert 0 < train_frac < 1
    target = Path(target)

    print("Loading pandas...")
    import pandas as pd

    print("Loading data...")
    flightlist = pd.read_parquet("data/flightlist_train.parquet")
    fuel = pd.read_parquet("data/fuel_train.parquet")

    flightlist_train, flightlist_val, fuel_train, fuel_val = split_train_val(
        train_frac=train_frac, flightlist=flightlist, fuel=fuel, seed=seed
    )

    print(f"Split flight list ({len(flightlist)}) into:")
    print(
        f" - train: {len(flightlist_train)} ({len(flightlist_train) / len(flightlist):.1%})"
    )
    print(
        f" - val: {len(flightlist_val)} ({len(flightlist_val) / len(flightlist):.1%})"
    )

    print(f"Split fuel data ({len(fuel)}) into:")
    print(f" - train: {len(fuel_train)} ({len(fuel_train) / len(fuel):.1%})")
    print(f" - val: {len(fuel_val)} ({len(fuel_val) / len(fuel):.1%})")

    flightlist_train_fname = (
        f"flightlist_split_train_{train_frac:.2f}_seed_{seed}.parquet"
    )
    flightlist_val_fname = f"flightlist_split_val_{train_frac:.2f}_seed_{seed}.parquet"
    fuel_train_fname = f"fuel_split_train_{train_frac:.2f}_seed_{seed}.parquet"
    fuel_val_fname = f"fuel_split_val_{train_frac:.2f}_seed_{seed}.parquet"

    print(
        f"Writing files to {target}:\n"
        f" - {flightlist_train_fname}\n"
        f" - {flightlist_val_fname}\n"
        f" - {fuel_train_fname}\n"
        f" - {fuel_val_fname}"
    )

    flightlist_train.to_parquet(target / flightlist_train_fname)
    flightlist_val.to_parquet(target / flightlist_val_fname)
    fuel_train.to_parquet(target / fuel_train_fname)
    fuel_val.to_parquet(target / fuel_val_fname)

    print("Done 🚀")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the training data into training and validation sets"
    )
    parser.add_argument("train_frac", type=float)
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument(
        "-t", "--target", default="split/", help="Target location for the split files"
    )
    args = parser.parse_args()

    Path(args.target).mkdir(parents=True, exist_ok=True)

    main(train_frac=args.train_frac, target=args.target, seed=args.seed)

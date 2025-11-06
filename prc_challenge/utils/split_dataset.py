def split_train_val(train_frac: float, flightlist, fuel, seed: int = 0):
    assert 0 < train_frac < 1

    flightlist_train = flightlist.sample(frac=train_frac, random_state=seed)
    flightlist_val = flightlist[~flightlist.index.isin(flightlist_train.index)]

    fuel_train = fuel[fuel["flight_id"].isin(flightlist_train["flight_id"])]
    fuel_val = fuel[fuel["flight_id"].isin(flightlist_val["flight_id"])]

    return flightlist_train, flightlist_val, fuel_train, fuel_val

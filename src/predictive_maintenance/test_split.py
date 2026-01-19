import pandas as pd
from predictive_maintenance.split import split_by_unit_id


def test_unit_splits_are_disjoint():
    df = pd.DataFrame({"unit_id": [1, 1, 2, 2, 3, 3], "cycle": [1, 2, 1, 2, 1, 2]})
    res = split_by_unit_id(df, 0.34, 0.33, 0.33, seed=42)

    assert res.train_units.isdisjoint(res.valid_units)
    assert res.train_units.isdisjoint(res.test_units)
    assert res.valid_units.isdisjoint(res.test_units)

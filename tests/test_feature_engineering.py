from src import feature_engineering
import pandas as pd
import numpy as np
from pandas import _testing


def test_drop_missing_data():

    df = pd.DataFrame({"number": [1.0, 1.5, 2.0, 0.0, np.nan]})

    expected = pd.DataFrame({"number": [1.0, 1.5, 2.0, 0.0]})

    result = feature_engineering.drop_missing_data(
        train=df,
        valid=df,
        test=df,
        cols=[
            "number",
        ],
    )["train"]

    _testing.assert_frame_equal(result, expected)


def test_numerical_missing_imputation_twofeatures():

    df = pd.DataFrame(
        {
            "a": [1.0, 1.5, 2.0, 0.0, 1.25, np.nan],
            "b": [1.0, 1.5, 2.0, 0.0, 0.0, np.nan],
            "c": [0.0, 0.0, 0.0, 0.0, 0, 0.0],
            "d": ["apple", "apple", "pear", "apple", "pear", "pear"],
        }
    )

    expected = pd.DataFrame(
        {
            "a": [1.0, 1.5, 2.0, 0.0, 1.25, 1.25],
            "b": [1.0, 1.5, 2.0, 0.0, 0.0, 1.0],
        }
    )

    train, valid, test = feature_engineering.numerical_missing_imputation(
        train=df,
        valid=df,
        test=df,
        cols=[
            "a",
            "b",
        ],
    )

    _testing.assert_frame_equal(train, expected)


def test_one_hot_encoding():

    df = pd.DataFrame({"class": ["a", "b", "c", "a", np.nan]})

    expected = pd.DataFrame(
        {
            "class_a": [1, 0, 0, 1, 0],
            "class_b": [0, 1, 0, 0, 0],
            "class_c": [0, 0, 1, 0, 0],
            "class_na": [0, 0, 0, 0, 1],
        }
    )

    train, valid, result = feature_engineering.one_hot_encoding(
        train=df,
        valid=df,
        test=df,
        cols=[
            "class",
        ],
    )

    _testing.assert_frame_equal(train, expected)

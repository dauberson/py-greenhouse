from feature_engine import encoding

# import pandas as pd


def one_hot_encoding(train, valid, test, cols):

    enc = encoding.OneHotEncoder(variables=cols)

    for col in cols:
        train[col] = train[col].fillna("na")
        valid[col] = valid[col].fillna("na")
        test[col] = test[col].fillna("na")

    enc.fit(train[cols])

    return {
        "train": enc.transform(train[cols]),
        "valid": enc.transform(valid[cols]),
        "test": enc.transform(test[cols]),
    }

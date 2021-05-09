from feature_engine import encoding, imputation


def drop_missing_data(train, valid, test, cols):

    fe = imputation.DropMissingData()

    fe.fit(train[cols])

    return {
        "train": fe.transform(train[cols]),
        "valid": fe.transform(valid[cols]),
        "test": fe.transform(test[cols]),
    }


def numerical_missing_imputation(train, valid, test, cols, imputation_method="median"):

    fe = imputation.MeanMedianImputer(imputation_method=imputation_method)

    fe.fit(train[cols])

    return (
        fe.transform(train[cols]),
        fe.transform(valid[cols]),
        fe.transform(test[cols]),
    )


def one_hot_encoding(train, valid, test, cols):

    print(train)

    fe = encoding.OneHotEncoder(variables=cols)

    for col in cols:
        train[col] = train[col].fillna("na")
        valid[col] = valid[col].fillna("na")
        test[col] = test[col].fillna("na")

    print(train)

    fe.fit(train[cols])

    return (
        fe.transform(train[cols]),
        fe.transform(valid[cols]),
        fe.transform(test[cols]),
    )

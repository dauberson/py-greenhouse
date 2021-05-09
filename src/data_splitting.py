def split(df, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=0):

    train = df.sample(frac=train_ratio, random_state=seed)

    rest = df.copy().drop(train.index)

    ratio = valid_ratio / (valid_ratio + test_ratio)

    valid = rest.sample(frac=ratio, random_state=seed)

    test = rest.drop(valid.index)

    return train, valid, test

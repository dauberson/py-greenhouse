import data_sourcing
import data_splitting
import data_preprocessing
import feature_engineering
from prefect import Flow, task, context
import pandas as pd

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)


@task
def sourcing():

    return data_sourcing.get()


@task
def cleansing(df):

    return data_preprocessing.clean(df)


@task
def normalizing(df):

    return data_preprocessing.normalize(df)


@task(nout=3)
def splitting(df):

    return data_splitting.split(df)


@task(nout=3)
def one_hot(train, valid, test, cols):

    logger = context.get("logger")

    # logger.info(train.head())

    train_hot, valid_hot, test_hot = feature_engineering.one_hot_encoding(
        train=train, valid=valid, test=test, cols=cols
    )

    train = train.join(train_hot)
    valid = valid.join(train_hot)
    test = test.join(train_hot)

    logger.info(train.head())

    return train, valid, test


@task(nout=3)
def numerical_missing(train, valid, test, cols):

    logger = context.get("logger")

    mask = train[cols].isna()

    logger.info(train[mask])

    (
        train_miss,
        valid_miss,
        test_miss,
    ) = feature_engineering.numerical_missing_imputation(
        train=train,
        valid=valid,
        test=test,
        cols=cols,
        imputation_method="median",
    )

    train = train.join(train_miss, rsuffix="_imputed")
    valid = valid.join(train_miss, rsuffix="_imputed")
    test = test.join(train_miss, rsuffix="_imputed")

    logger.info(train[mask])

    return train, valid, test


@task(log_stdout=True)
def task_print(x):

    print(x)

    pass


with Flow("greenhouse") as flow:

    df = sourcing()
    df = cleansing(df)
    df = normalizing(df)
    train, valid, test = splitting(df)

    categorical_cols = [
        "sex",
    ]

    train, valid, test = one_hot(
        train=train, valid=valid, test=test, cols=categorical_cols
    )

    numerical_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    train, valid, test = numerical_missing(
        train=train, valid=valid, test=test, cols=numerical_cols
    )

if __name__ == "__main__":

    flow.run()

    flow.visualize(filename="flow/prefect_flow")

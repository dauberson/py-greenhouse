import data_sourcing
import data_splitting
import data_preprocessing
import feature_engineering
import monitoring
import modeling
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

    logger.info(train)

    train_hot, valid_hot, test_hot = feature_engineering.one_hot_encoding(
        train=train,
        valid=valid,
        test=test,
        cols=cols,
    )

    train = train.join(train_hot)
    valid = valid.join(valid_hot)
    test = test.join(test_hot)

    logger.info(train)

    return train, valid, test


@task(nout=3)
def imputation(train, valid, test, cols, imputation_method):

    logger = context.get("logger")

    # Find rows where the numerical variables are nan
    mask = train[cols].isna()

    logger.info(train[mask])

    train_imp, valid_imp, test_imp = feature_engineering.numerical_missing_imputation(
        train=train,
        valid=valid,
        test=test,
        cols=cols,
        imputation_method=imputation_method,
    )

    train = train.join(train_imp, rsuffix="_imputed")
    valid = valid.join(valid_imp, rsuffix="_imputed")
    test = test.join(test_imp, rsuffix="_imputed")

    logger.info(train[mask])

    return train, valid, test


@task
def monitor(df, path):

    monitoring.monitor(df, path)

    pass


@task
def model(train, valid, test, y_col, x_col):

    return modeling.fit_transform(train, valid, test, y_col, x_col)


@task(log_stdout=True)
def task_print(x):

    print(x)

    pass


with Flow("greenhouse") as flow:

    df = sourcing()
    df = cleansing(df)
    df = normalizing(df)
    train, valid, test = splitting(df)

    monitor(train, "monitor/monitor_before_feat_eng.html")

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

    train, valid, test = imputation(
        train=train,
        valid=valid,
        test=test,
        cols=numerical_cols,
        imputation_method="median",
    )

    monitor(train, "monitor/monitor_afeterfeat_eng.html")

    y_col = ["species"]

    x_col = [
        "sex_male",
        "sex_female",
        "sex_na",
        "bill_length_mm_imputed",
        "bill_depth_mm_imputed",
        "flipper_length_mm_imputed",
        "body_mass_g_imputed",
    ]

    model(train, valid, test, y_col, x_col)

if __name__ == "__main__":

    flow.run()

    flow.visualize(filename="flow/prefect_flow")

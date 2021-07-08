import data_sourcing
import data_splitting
import data_preprocessing
import feature_engineering
import monitoring
import modeling
import performance_monitoring
from prefect import Flow, task, context
import pandas as pd
import time

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

start_time = time.strftime("%Y%m%d%H%M%S")


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


@task(nout=5)
def model(train, valid, test, obs, y_col, x_col):

    mo = modeling.model()

    mo.fit(train=train, y_col=y_col, x_col=x_col)

    lst = list(mo.transform_sets(train=train, valid=valid, test=test))

    lst.append(mo.transform_new(obs=obs))

    return lst


@task
def threshold(y_true, y_score):

    return performance_monitoring.optimal_threshold(y_true, y_score)


@task
def performance(y_true, y_score, best_hyperparams, path, opt_thr, suffix):

    return performance_monitoring.report_performance(
        y_true=y_true,
        y_score=y_score,
        best_hyperparams=best_hyperparams,
        path=path,
        opt_thr=opt_thr,
        suffix=suffix,
    )


@task
def binarize(binary_map, series):

    return series.map(binary_map)


@task(log_stdout=True)
def task_print(x):

    print(x)

    pass


@task
def df_to_csv(df, filename):

    df.to_csv(filename)

    pass


with Flow("greenhouse") as flow:

    df = sourcing()
    df = cleansing(df)
    df = normalizing(df)
    train, valid, test = splitting(df)

    # monitor(train, "monitor/monitor_before_feat_eng.html")

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

    # monitor(train, "monitor/monitor_afeterfeat_eng.html")

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

    train, valid, test, best_hyperparams, new = model(
        train=train,
        valid=valid,
        test=test,
        obs=test,
        y_col=y_col,
        x_col=x_col,
    )

    path = "data/"
    filename = path + "{}_predict_new.csv".format(start_time)

    df_to_csv(df=new, filename=filename)

    opt_thr = threshold(y_true=train["actual"], y_score=train["prob_0"])

    binary_map = {
        0: 1,
        1: 0,
        2: 0,
    }

    performance(
        y_true=binarize(binary_map=binary_map, series=train["actual"]),
        y_score=train["prob_0"],
        best_hyperparams=best_hyperparams,
        path="monitor/",
        opt_thr=opt_thr,
        suffix="train",
    )

    performance(
        y_true=binarize(binary_map=binary_map, series=test["actual"]),
        y_score=test["prob_0"],
        best_hyperparams=best_hyperparams,
        path="monitor/",
        opt_thr=opt_thr,
        suffix="test",
    )

    performance(
        y_true=binarize(binary_map=binary_map, series=valid["actual"]),
        y_score=valid["prob_0"],
        best_hyperparams=best_hyperparams,
        path="monitor/",
        opt_thr=opt_thr,
        suffix="valid",
    )

if __name__ == "__main__":

    flow.run()

    flow.visualize(filename="flow/prefect_flow")

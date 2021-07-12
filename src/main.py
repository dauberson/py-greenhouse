from os import name
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

    use_sourcing_example = True 
    path = "/usr/app/examples/df_example.csv" #path where is the data if you don't use the example change it.
    sep = ";" #whitch separate your columns, to defaut is comma(,)
    header = 0 #index of dataset header if it doesn't exists use 0
    names = ["colum_name_a", "colum_name_b"] #columns names it will override the headers existing or not

    if use_sourcing_example == True:
        return data_sourcing.get_example()

    return data_sourcing.get(path = path, sep = sep, header = header, names = names)

@task
def tokenizing(df, text_col, toke_type):

    df["tokens"] = data_preprocessing.tokenizing(df, text_col, toke_type)

    return df


@task
def stop_words(df, token_col):
    
    df["tokens_wosw"] = data_preprocessing.stop_words(df, token_col)

    return df

@task
def cleansing(df, text_col):

    df["clean_text"] = data_preprocessing.clean(df, text_col)

    return df


@task
def normalizing(df):

    return data_preprocessing.normalize(df)


@task(nout=3)
def splitting(df, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):

    return data_splitting.split(df, train_ratio, valid_ratio, test_ratio)

@task
def label(df, y_col):

    df["num_cat"] = data_preprocessing.label_encoding(df, y_col)

    return df


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


@task(nout=4)
def model(df, train, valid, test, y_col, x_col, vec_type):

    model = modeling.model()

    model.vectorizing(df, train, valid, test, x_col, vec_type)

    model.fit(train, y_col)

    lst = model.transform_sets(train=train, valid=valid, test=test)

    # lst.append(mo.transform_new(obs=obs))

    return lst


@task
def threshold(y_true, y_score):

    return performance_monitoring.optimal_threshold(y_true, y_score)


@task
def performance(y_true, y_pred, best_hyperparams, path, suffix):

    return performance_monitoring.report_performance(
        y_true=y_true,
        y_pred=y_pred,
        best_hyperparams=best_hyperparams,
        path=path,
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

@task
def predict(instance):

    return modeling.transform_new(instance)


with Flow("greenhouse") as flow:

    df = sourcing()
    df = cleansing(df, text_col = "text")
    df = tokenizing(df, text_col = "clean_text", toke_type = "multi_word")
    df = stop_words(df, token_col = "tokens")
    df = label(df, y_col = "cats")

    train, valid, test = splitting(df, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)

    monitor(train, "monitor/monitor.html")

    train, valid, test, best_hyperparams = model(
        df = df,
        train=train,
        valid=valid,
        test=test,
        y_col="num_cat",
        x_col="tokens_wosw",
        vec_type="count"
    )

    path = "data/"
    filename = path + "{}_predict_new.csv".format(start_time)

    df_to_csv(df=test, filename=filename)

    performance(
        y_true=train["actual"],
        y_pred=train["pred"],
        best_hyperparams=best_hyperparams,
        path="monitor/",
        suffix="train",
    )

    performance(
        y_true=test["actual"],
        y_pred=test["pred"],
        best_hyperparams=best_hyperparams,
        path="monitor/",
        suffix="test",
    )

    performance(
        y_true=valid["actual"],
        y_pred=valid["pred"],
        best_hyperparams=best_hyperparams,
        path="monitor/",
        suffix="valid",
    )


if __name__ == "__main__":

    flow.run()

    flow.visualize(filename="flow/prefect_flow")

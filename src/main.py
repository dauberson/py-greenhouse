from os import name
import data_sourcing
import data_splitting
import data_preprocessing
import feature_engineering
import monitoring
import modeling
import performance_monitoring
import feature_extraction
from prefect import Flow, task, context
import pandas as pd
import time

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# model_id = time.strftime("%Y%m%d%H%M%S")
model_id = "123"


@task
def sourcing():

    use_sourcing_example = False
    path = "/usr/app/examples/data_example.csv"  # path where is the data if you don't use the example change it.
    sep = ";"  # whitch separate your columns, to defaut is comma(,)
    header = 0  # index of dataset header if it doesn't exists use 0
    names = [
        "text",
        "cats",
    ]  # columns names will override the headers existing or not

    if use_sourcing_example:
        return data_sourcing.get_example()

    return data_sourcing.get(path=path, sep=sep, header=header, names=names)


@task
def tokenizing(tokenizer_type, text_col, df):

    df["tokens"] = df[text_col].apply(feature_extraction.tokenizer(tokenizer_type))

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
def model(
    df, train, valid, test, y_col, x_col, vectorizer_type, tokenizer_type, model_id
):

    model = modeling.model(model_id=model_id)

    model.vectorizing(vectorizer_type, tokenizer_type)

    model.fit(train, y_col, x_col)

    lst = model.metrics(train=train, valid=valid, test=test)

    # lst.append(mo.transform_new(obs=obs))

    return lst


@task
def threshold(y_true, y_score):

    return performance_monitoring.optimal_threshold(y_true, y_score)


@task
def performance(y_true, y_pred, best_hyperparams, path, suffix, model_id):

    return performance_monitoring.report_performance(
        y_true=y_true,
        y_pred=y_pred,
        best_hyperparams=best_hyperparams,
        path=path,
        suffix=suffix,
        model_id=model_id,
    )


@task(log_stdout=True)
def task_print(x):

    print(x)

    pass


@task
def df_to_csv(df, path, suffix):

    filename = "{0}/{1}_metadata_{2}.csv".format(path, model_id, suffix)

    df.to_csv(filename)

    pass


@task
def predict(instance):

    return modeling.transform_new(instance)


with Flow("greenhouse") as flow:

    df = sourcing()
    df = cleansing(df, text_col="text")
    # df = tokenizing(tokenizer_type="split", text_col="clean_text", df=df)
    df = stop_words(df, token_col="tokens")
    # df = label(df, y_col="cats")

    task_print(df)

    train, valid, test = splitting(df, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)

    # monitor(train, "monitor/monitor.html")

    train, valid, test, best_hyperparams = model(
        df=df,
        train=train,
        valid=valid,
        test=test,
        y_col="cats",
        x_col="clean_text",
        vectorizer_type="count",
        tokenizer_type="split",
        model_id=model_id,
    )

    df_to_csv(df=train, path="data/", suffix="train")
    df_to_csv(df=test, path="data/", suffix="test")
    df_to_csv(df=valid, path="data/", suffix="valid")

    performance(
        y_true=train["actual"],
        y_pred=train["pred"],
        best_hyperparams=best_hyperparams,
        path="monitor/",
        suffix="train",
        model_id=model_id,
    )

    performance(
        y_true=test["actual"],
        y_pred=test["pred"],
        best_hyperparams=best_hyperparams,
        path="monitor/",
        suffix="test",
        model_id=model_id,
    )

    performance(
        y_true=valid["actual"],
        y_pred=valid["pred"],
        best_hyperparams=best_hyperparams,
        path="monitor/",
        suffix="valid",
        model_id=model_id,
    )


if __name__ == "__main__":

    flow.run()

    flow.visualize(filename="flow/prefect_flow")

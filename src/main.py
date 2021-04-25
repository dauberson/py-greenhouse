import data_sourcing
import data_splitting
import data_preprocessing
import feature_engineering
from prefect import Flow, task

# from prefect import Task


@task()
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

    return feature_engineering.one_hot_encoding(train, valid, test, cols)


with Flow("greenhouse") as flow:

    df = sourcing()
    df = cleansing(df)
    df = normalizing(df)
    s = splitting(df)

    train = s["train"]
    valid = s["valid"]
    test = s["test"]

    cols = [
        "sex",
    ]

    o = one_hot(train, valid, test, cols)

    train = o["train"]
    valid = o["valid"]
    test = o["test"]


if __name__ == "__main__":

    flow.run()

    flow.visualize(filename="flow/prefect_flow")

import data_sourcing
import data_splitting
import data_preprocessing
from prefect import Flow, task


@task()
def sourcing():
    return data_sourcing.get()


@task(nout=3)
def splitting(df):
    return data_splitting.split(df)


@task
def cleansing(df):
    return data_preprocessing.clean(df)


@task
def normalizing(df):
    return data_preprocessing.normalize(df)


with Flow("greenhouse") as flow:
    df = sourcing()
    df = cleansing(df)
    df = normalizing(df)
    s = splitting(df)

    train = s["train"]
    valid = s["valid"]
    test = s["test"]

if __name__ == "__main__":

    flow.run()

    flow.visualize(filename="flow/prefect_flow")

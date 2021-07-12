from os import name
import pandas as pd

def get(path, sep, header, names):

    df = pd.read_csv(filepath_or_buffer = path, sep = sep, header = header, names = names)

    return df

def get_example():

    df = pd.read_csv("/usr/app/examples/data_example.csv", sep=";", header=1, names=["text","cats"])

    return df.sample(50, random_state=1)

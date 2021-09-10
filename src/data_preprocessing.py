import nltk
from resources import stopwords
import re

nltk.download("punkt")

from nltk.tokenize import (
    TreebankWordTokenizer,
    word_tokenize,
    wordpunct_tokenize,
    TweetTokenizer,
    MWETokenizer,
)

from sklearn.preprocessing import LabelEncoder


def clean(df, text_col):

    df = df[text_col]

    df = df.apply(
        lambda row: re.sub(r"\W", " ", row)
    )  # Remove all the special characters
    df = df.apply(
        lambda row: re.sub(r"\s+[a-zA-Z]\s+", " ", row)
    )  # Remove all single characters
    df = df.apply(
        lambda row: re.sub(r"\^[a-zA-Z]\s+", " ", row)
    )  # Remove single characters from the start
    df = df.apply(
        lambda row: re.sub(r"\s+", " ", row, flags=re.I)
    )  # Substituting multiple spaces with single space
    df = df.apply(lambda row: row.lower())  # Converting to Lowercase

    return df


def normalize(df):

    return df


def stop_words(df, token_col):
    stop = stopwords.stop_words_directory(language="portuguese")
    return df[token_col].apply(lambda row: [item for item in row if item not in stop])


def label_encoding(df, y_col):

    return LabelEncoder().fit_transform(df[y_col])


def pos_tagger():

    return

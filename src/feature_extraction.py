import nltk
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer,
)
from nltk.tokenize import (
    TreebankWordTokenizer,
    word_tokenize,
    wordpunct_tokenize,
    TweetTokenizer,
    MWETokenizer,
)


def vectorizer(vectorizer_type, tokenizer_type):

    # params = {input:'content',
    #         lowercase:False,
    #         preprocessor = lambda x: x,
    #         tokenizer = lambda x: x}

    return {
        "count": CountVectorizer(
            ngram_range=(1, 1),
            # tokenizer=tokenizer(tokenizer_type)
        ),
        "tfidf": TfidfVectorizer(
            ngram_range=(1, 1), tokenizer=tokenizer(tokenizer_type)
        ),
        "hashing": HashingVectorizer(
            ngram_range=(1, 1), tokenizer=tokenizer(tokenizer_type)
        ),
    }.get(
        vectorizer_type,
        CountVectorizer(ngram_range=(1, 1), tokenizer=tokenizer(tokenizer_type)),
    )


def tokenizer(tokenizer_type):

    return {
        "split": lambda x: x.split(),
        "wordpunct": lambda x: nltk.wordpunct_tokenize(x),
        "multi_word": lambda x: MWETokenizer([("escolas", "estaduais")]).tokenize(
            word_tokenize(x)
        ),
        "tweet": lambda x: nltk.TweetTokenizer().tokenize(x),
        "treebank": lambda x: nltk.TreebankWordTokenizer().tokenize(x),
    }.get(tokenizer_type, lambda x: x.split())

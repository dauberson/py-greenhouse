from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

def vectorizer(vec_type):

    # params = {input:'content',
    #         lowercase:False,
    #         preprocessor = lambda x: x,
    #         tokenizer = lambda x: x}

    return {
        "count": CountVectorizer(
            input='content',
            lowercase=False,
            preprocessor = lambda x: x,
            tokenizer = lambda x: x),
        "tfidf": TfidfVectorizer(
            input='content',
            lowercase=False,
            preprocessor = lambda x: x,
            tokenizer = lambda x: x),
        "hashing": HashingVectorizer(
            input='content',
            lowercase=False,
            preprocessor = lambda x: x,
            tokenizer = lambda x: x)
        }.get(vec_type,  CountVectorizer(
            input='content',
            lowercase=False,
            preprocessor = lambda x: x,
            tokenizer = lambda x: x))   
             

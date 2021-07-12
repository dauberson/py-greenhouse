import nltk
nltk.download('stopwords')

def stop_words_directory(language):
    return set(nltk.corpus.stopwords.words(language))
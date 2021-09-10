import data_preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import feature_extraction
import pickle
import time
from eli5 import show_weights
from eli5.lime import TextExplainer
import joblib
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


class model:
    def __init__(self, model_id):

        self.model_id = model_id

        pass

    def vectorizing(self, vectorizer_type, tokenizer_type):

        self.vec = feature_extraction.vectorizer(vectorizer_type, tokenizer_type)

    def fit(self, train, y_col, x_col, n_jobs=1, seed=1):

        self.x_col = x_col
        self.y_col = y_col

        model = RandomForestClassifier(random_state=seed)

        # self.clf = MultinomialNB()
        # self.clf = SVC(C=150, gamma=2e-2, probability=True)

        pipe = Pipeline(steps=[("vec", self.vec), ("model", model)])

        param_grid = {
            "model": [model],
            "model__max_depth": np.linspace(1, 32, 32),
            # "model__n_estimators": np.arange(100, 1000, 100),
            # "model__criterion": ["gini", "entropy"],
            # "model__max_leaf_nodes": [16, 64, 128, 256],
            # "model__oob_score": [True],
        }

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=2,
            cv=None,
            n_iter=10,
        )

        self.search = search.fit(train[x_col], train[y_col])
        self.pipe = pipe.fit(train[x_col], train[y_col])

        # path = "models/"
        joblib.dump(self.pipe, "/usr/app/models/" + self.model_id + ".joblib")
        # joblib.dump(self.pipe, path + "20210717003713.joblib")

        return "Model " + str(self.model_id) + " is trained!"

    def eli5(self):

        return show_weights(
            self.pipe.named_steps["model"],
            feature_names=self.pipe.named_steps["vec"].get_feature_names(),
        )

    def metrics(self, train, valid, test):

        np.set_printoptions(precision=2)

        train_out = pd.DataFrame(
            {
                "actual": train[self.y_col],
                "pred": (self.search).predict(train[self.x_col]),
            }
        )
        for i in range(0, len((self.search).predict_proba(train[self.x_col])[0])):
            train_out["proba_" + str(i)] = self.search.predict_proba(train[self.x_col])[
                :, i
            ].round(3)

        valid_out = pd.DataFrame(
            {
                "actual": valid[self.y_col],
                "pred": (self.search).predict(valid[self.x_col]),
            }
        )
        for i in range(0, len((self.search).predict_proba(valid[self.x_col])[0])):
            valid_out["proba_" + str(i)] = self.search.predict_proba(valid[self.x_col])[
                :, i
            ].round(3)

        test_out = pd.DataFrame(
            {
                "actual": test[self.y_col],
                "pred": (self.search).predict(test[self.x_col]),
            }
        )
        for i in range(0, len((self.search).predict_proba(test[self.x_col])[0])):
            test_out["proba_" + str(i)] = self.search.predict_proba(test[self.x_col])[
                :, i
            ].round(3)

        return train_out, valid_out, test_out, (self.search).best_params_

    def predict(self, obs):

        te = TextExplainer(random_state=42)

        new = pd.DataFrame({"text": obs}, index=[0])
        new["clean_text"] = data_preprocessing.clean(new, "text")

        # Predict
        obs_out = (self.search).predict(new["clean_text"])

        # Explain
        te.fit(obs, self.pipe.predict_proba)

        return obs_out, te.show_prediction(
            target_names=self.pipe.named_steps["vec"].get_feature_names()
        )

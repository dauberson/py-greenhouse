import data_preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
import numpy as np
import pandas as pd
import feature_extraction
import pickle
import time

start_time = time.strftime("%Y%m%d%H%M%S")

class model:
    def __init__(self):

        pass

    def vectorizing(self, df, train, valid, test, x_col, vec_type):

        self.x_col = x_col
        self.vec_type = vec_type

        self.vec = feature_extraction.vectorizer(vec_type)
        self.vec.fit(df[x_col]) #get all vocabulary

        self.train_vec = self.vec.transform(train[x_col])
        self.test_vec = self.vec.transform(test[x_col])
        self.valid_vec =  self.vec.transform(valid[x_col])


    def fit(self, train, y_col, n_jobs=1, seed=1):

        self.y_col = y_col

        grid = {}

        grid["max_features"] = [2, 3]
        grid["max_depth"] = [2, 3]
        grid["n_estimators"] = [100]

        clf = RandomForestClassifier(random_state=seed)

        self.clf_random = RandomizedSearchCV(
            estimator=clf,
            param_distributions=grid,
            n_iter=10,
            cv=None,
            verbose=2,
            random_state=seed,
            n_jobs=n_jobs,
        )

        (self.clf_random).fit(self.train_vec, train[y_col])

        path = "models/"
        pickle.dump(self.clf_random, open(path + "{}_classifier.pkl".format(start_time), "wb"))

        loaded_model = pickle.load(open("{}_classifier.pkl".format(start_time), "rb"))

        return loaded_model

    def transform_sets(self, train, valid, test):

        train_out = pd.DataFrame(
            {"actual": train["num_cat"],
            "pred": (self.clf_random).predict(self.train_vec),
            "prob_0": (self.clf_random).predict_proba(self.train_vec)[:,0],
            "prob_1":(self.clf_random).predict_proba(self.train_vec)[:,1],
            "prob_2":(self.clf_random).predict_proba(self.train_vec)[:,2]}
        )

        valid_out = pd.DataFrame(
            {"actual": valid["num_cat"],
            "pred": (self.clf_random).predict(self.valid_vec),
            "prob_0": (self.clf_random).predict_proba(self.valid_vec)[:,0],
            "prob_1":(self.clf_random).predict_proba(self.valid_vec)[:,1],
            "prob_2":(self.clf_random).predict_proba(self.valid_vec)[:,2]}
        )

        test_out = pd.DataFrame(
            {"actual": test["num_cat"],
            "pred": (self.clf_random).predict(self.test_vec),
            "prob_0": (self.clf_random).predict_proba(self.test_vec)[:,0],
            "prob_1":(self.clf_random).predict_proba(self.test_vec)[:,1],
            "prob_2":(self.clf_random).predict_proba(self.test_vec)[:,2]}
        )

        return train_out, valid_out, test_out, (self.clf_random).best_params_

    def transform_new(self, obs):

        new = pd.DataFrame({'text': obs}, index=[0])
        new["clean_text"] = data_preprocessing.clean(new, "text")
        new["tokens"] = data_preprocessing.tokenizing(new, "clean_text", self.vec_type)
        new["tokens_wosw"] = data_preprocessing.stop_words(new, "tokens")

        # Predict
        obs_out = (self.clf_random).predict(self.vec.transform(new["tokens_wosw"]))

        return obs_out
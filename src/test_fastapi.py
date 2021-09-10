from re import S
from fastapi import FastAPI, Query
import joblib
from eli5 import show_weights
import pandas as pd
from eli5.lime import TextExplainer
from fastapi.responses import HTMLResponse
import json
import os
from typing import List, Optional
import time

os.chdir("/usr/app/src")

app = FastAPI()


@app.get("/{model_id}")
async def main(model_id: str):

    model = joblib.load("/usr/app/models/" + model_id + ".joblib")
    # raise Exception('x should not exceed 5. The value of x was: {}'.format(model))

    model_pipeline = {}
    for idx in range(0, len(model)):
        model_pipeline["step_" + str(idx)] = str(model[idx])

    return "Model Pipeline " + str(model_id) + ":", model_pipeline


@app.get("/{model_id}/predict/{text}")
async def main(model_id: str, text: str):

    model = joblib.load("/usr/app/models/" + model_id + ".joblib")

    predict_dict = {"text": text, "predict_class": model.predict([text])[0]}
    for idx in range(0, len(model.classes_)):
        predict_dict[model.classes_[idx]] = model.predict_proba([text])[0][idx]

    # te = TextExplainer(random_state=42)
    # te.fit(text,model.predict_proba)
    # show_prediction = te.show_prediction(target_names=model.named_steps["vec"].get_feature_names())

    return predict_dict


@app.get("/{model_id}/metrics/{metadata}")
async def main(model_id: str, metadata: str):

    f = open(
        "/usr/app/monitor/" + model_id + "_metadata_" + metadata + ".json",
    )
    data = json.load(f)

    return data


@app.post("/train/{dataset_name}/")
async def main(
    dataset_name: str,
    sep: str,
    header: int,
    names: List[str] = Query(None, description="Dataset columns names", min_length=2),
    model_id: Optional[str] = None,
):
    import data_sourcing
    import data_preprocessing
    import feature_extraction
    import data_splitting
    import modeling
    import performance_monitoring

    df = data_sourcing.get(
        "/usr/app/examples/" + dataset_name + ".csv", sep, header, names
    )
    df["clean_text"] = data_preprocessing.clean(df, "text")
    df["tokens"] = df["clean_text"].apply(feature_extraction.tokenizer("split"))
    df["tokens_wosw"] = data_preprocessing.stop_words(df, "tokens")

    train, valid, test = data_splitting.split(
        df, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1
    )

    model = modeling.model(model_id if model_id else time.strftime("%Y%m%d%H%M%S"))

    model.vectorizing(vectorizer_type="count", tokenizer_type="split")

    id = model.fit(train, "cats", "clean_text")

    train, valid, test, best_hyperparams = model.metrics(train=train, valid=valid, test=test)

    train.to_csv("{0}/{1}_metadata_{2}.csv".format("/usr/app/data", model_id, "train"))
    test.to_csv("{0}/{1}_metadata_{2}.csv".format("/usr/app/data", model_id, "test"))
    valid.to_csv("{0}/{1}_metadata_{2}.csv".format("/usr/app/data", model_id, "valid"))

    performance_monitoring.report_performance(
        y_true=train["actual"],
        y_pred=train["pred"],
        best_hyperparams=best_hyperparams,
        path="/usr/app/monitor/",
        suffix="train",
        model_id=model_id,
    )

    performance_monitoring.report_performance(
        y_true=test["actual"],
        y_pred=test["pred"],
        best_hyperparams=best_hyperparams,
        path="/usr/app/monitor/",
        suffix="test",
        model_id=model_id,
    )

    performance_monitoring.report_performance(
        y_true=valid["actual"],
        y_pred=valid["pred"],
        best_hyperparams=best_hyperparams,
        path="/usr/app/monitor/",
        suffix="valid",
        model_id=model_id,
    )

    return id

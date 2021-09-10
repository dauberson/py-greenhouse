import time
import sklearn
from sklearn import metrics
import pandas as pd
import json

meta = {}


def report_performance(model_id, y_true, y_pred, best_hyperparams, path, suffix="_"):

    for key, value in best_hyperparams.items():
        best_hyperparams[key] = str(value)

    meta["optimal_hyperparams"] = best_hyperparams

    meta["classification_report"] = metrics.classification_report(
        y_true=y_true, y_pred=y_pred, output_dict=True
    )

    filename = "{0}/{1}_metadata_{2}.json".format(path, model_id, suffix)

    # Export to JSON
    with open(filename, "w") as fp:
        json.dump(meta, fp, indent=4)

    pass

import time
import sklearn
from sklearn import metrics
import pandas as pd
import json

meta = {}

meta["timestr"] = time.strftime("%Y%m%d%H%M%S")

def report_performance(
    y_true, y_pred, best_hyperparams, path, opt_thr=0.5, suffix="_"
):

    meta["optimal_hyperparams"] = best_hyperparams

    meta["classification_report"] = metrics.classification_report(
        y_true=y_true, y_pred=y_pred, output_dict=True
    )

    filename = "{0}/{1}_metadata_{2}.json".format(path, meta["timestr"], suffix)

    # Export to JSON
    with open(filename, "w") as fp:
        json.dump(meta, fp, indent=4)

    pass

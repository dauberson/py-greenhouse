import time
import sklearn
from sklearn import metrics
import numpy as np
import json

meta = {}

meta["timestr"] = time.strftime("%Y%m%d%H%M%S")


def optimal_threshold(y_true, y_score):

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        y_true=y_true, y_score=y_score, pos_label=1, drop_intermediate=False
    )

    diff = np.abs(tpr - fpr)

    diff_idx = np.argmax(diff)

    return thresholds[diff_idx]


def report_performance(
    y_true, y_score, best_hyperparams, path, opt_thr=0.5, suffix="_"
):

    meta["optimal_hyperparams"] = best_hyperparams

    meta["optimal_threshold"] = opt_thr

    fpr, tpr, thr = sklearn.metrics.roc_curve(
        y_true=y_true, y_score=y_score, pos_label=1, drop_intermediate=False
    )

    meta["AUC"] = metrics.auc(fpr, tpr)

    diff = np.abs(tpr - fpr)

    meta["max_diff_FPR_TPR"] = np.max(diff)

    diff_idx = np.argmax(diff)

    meta["threshold_from_max_diff"] = thr[diff_idx]

    y_pred = [int(k >= opt_thr) for k in y_score]

    meta["classification_report"] = metrics.classification_report(
        y_true=y_true, y_pred=y_pred, output_dict=True
    )

    filename = "{0}/{1}_metadata_{2}.json".format(path, meta["timestr"], suffix)

    # Export to JSON
    with open(filename, "w") as fp:
        json.dump(meta, fp, indent=4)

    pass

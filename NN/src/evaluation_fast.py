import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import rankdata
from joblib import Parallel, delayed


@njit
def _auc(actual, pred_ranks):
    actual = np.asarray(actual)
    pred_ranks = np.asarray(pred_ranks)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auc(actual, predicted):
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)


def uAUC(y_true, y_pred, userids, target):
    num_labels = y_pred.shape[1]

    def uAUC_infunc(i):
        uauc_df = pd.DataFrame()
        uauc_df['userid'] = userids
        uauc_df['y_true'] = y_true[:, i]
        uauc_df['y_pred'] = y_pred[:, i]

        label_nunique = uauc_df.groupby(by='userid')['y_true'].transform('nunique')
        uauc_df = uauc_df[label_nunique == 2]

        aucs = uauc_df.groupby(by='userid').apply(
            lambda x: auc(x['y_true'].values, x['y_pred'].values))
        return round(np.mean(aucs), 6)

    uauc = Parallel(n_jobs=14)(delayed(uAUC_infunc)(i) for i in range(num_labels))
    eval_dict = {}
    for i in range(num_labels):
        eval_dict[target[i]] = uauc[i]
    weight_uauc = round(np.average(uauc, weights=[4, 3, 2, 1, 1, 1, 1]), 6)
    print(eval_dict)
    print(f"Weighted uAUC: {weight_uauc:.6f}")
    return weight_uauc, uauc
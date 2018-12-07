"""
Metrics used on the challenge
"""
import numpy as np

"""
This metrics are only for train set, they require 4 weeks of data
"""
def week_std_metric(df, series_id, column='consumption'):
    values = df[column].values[df.series_id.values == series_id]
    values = np.reshape(values, (4, 24*7))
    metric = np.std(values, axis=0)/np.mean(values)
    return np.mean(metric)

def day_std_metric(df, series_id, column='consumption'):
    values = df[column].values[df.series_id.values == series_id]
    values = np.reshape(values, (-1, 24))
    metric = np.std(values, axis=0)/np.mean(values)
    return np.mean(metric)

"""
Definition of the metric of the challenge
"""
def weighted_normalized_mean_abs_error(y_trues, y_preds):
    """
    Parameters
    ----------
    y_trues : list of np.array
        A list with arrays of true compsuntions. It's a list because the elements
        may have different lenght (24, 7, 2)
    y_preds : list of np.array
        A list with arrays of predicted compsuntions. It's a list because the elements
        may have different lenght (24, 7, 2)
    """
    errors = [normalized_mean_abs_error(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]
    weights = get_window_size_weights(y_trues)
    weighted_errors = [error*weight for error, weight in zip(errors, weights)]
    weighted_errors = np.concatenate(weighted_errors)
    return np.mean(weighted_errors)

def normalized_mean_abs_error(y_true, y_pred):
    error = np.abs(y_true - y_pred)/np.mean(y_true)
    return error

def get_window_size_weights(y_trues):
    """
    Given a list with the true predictions or similar computes weights for
    each sample for giving same weight to hourly, daily and weekly predictions.
    """
    lens = [len(y_true) for y_true in y_trues]
    # values, counts = np.unique(lens, return_counts=True)
    # value_to_weight = {}
    # for value, count in zip(values, counts):
    #     value_to_weight[value] = count*1./np.sum(counts)
    value_to_weight = {
        24: 1,
        7: 24/7,
        2: 12,
    }
    weights = [value_to_weight[value] for value in lens]
    return weights

import os
import re
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
import hashlib


def ensure_dir(f):
    # from http://stackoverflow.com/questions/273192/python-best-way-to-create-directory-if-it-doesnt-exist-for-file-write
    d = os.path.dirname(f)
    if d != '' and not os.path.exists(d):
        os.makedirs(d)


def log_to_file(log_file):
    ensure_dir(log_file)
    h = logging.FileHandler(log_file)
    h.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    logging.getLogger().addHandler(h)


def tolist(x):
    return x if isinstance(x, list) else [x]


def sigmoid(x):
    return 1/(1+np.exp(-x))


def logit(x):
    return np.log(x / (1 - x))


def hash_of_numpy_array(x):
    """Works only if there are no objects"""
    h = hashlib.sha224(x.tobytes()).hexdigest()
    return h


def hash_of_pandas_df(x):
    s = pd.util.hash_pandas_object(x)
    assert len(s) == len(x)
    return hash_of_numpy_array(s.values)



def compute_nmae(y_pred, y_true, ci):
    assert all(y_pred.index == y_true.index)
    assert all(y_pred.index == ci.index)
    err = (y_pred - y_true).abs()
    err = err.multiply(ci, axis=0)
    return err.sum().sum() / (len(err) * len(err.columns))


def prediction_params(prediction_window):
    """returns details of prediction:
prediction_hours - what is the time frame for prediction
prediction_days  - how many days are covered by prediction
prediction_count - how many value are in target
prediction_agg   - how many hours need to be aggregated to form single target value
wi               - w_i value from the cost function"""
    res = {
        'hourly': {'prediction_hours': 24, 'prediction_agg': 1, 'wi': 24/24, 'offs': 0},
        'daily': {'prediction_hours': 24*7, 'prediction_agg': 24, 'wi': 24/7, 'offs': 0},
        'daily1': {'prediction_hours': 24, 'prediction_agg': 24, 'wi': 24/7, 'offs': 0},
        'weekly': {'prediction_hours': 24*7*2, 'prediction_agg': 24*7, 'wi': 24/2, 'offs': 0},
        'weekly1': {'prediction_hours': 24*7, 'prediction_agg': 24*7, 'wi': 24/2, 'offs': 0},
        'weekly2': {'prediction_hours': 24*7, 'prediction_agg': 24*7, 'wi': 24/2, 'offs': 24*7}
    }[prediction_window]
    res['prediction_days'] = (res['prediction_hours'] + res['offs'])//24
    res['prediction_count'] = res['prediction_hours'] // res['prediction_agg']
    return res


def filter_columns(df, prediction_window='hourly', cold_start_days=1, force_is_day_off=False):
    params = prediction_params(prediction_window)
    prediction_unit = prediction_window[0]
    target_days = params['prediction_days']
    res = pd.DataFrame(index=df.index)
    res['prediction_window'] = prediction_window
    res['cold_start_days'] = cold_start_days
    features = defaultdict(list)
    targets = []
    clusters_columns = ['hourly_same_day', 'hourly_working_days', 'hourly_days_off',
       'daily_same_day', 'daily_working_days', 'daily_days_off']
    for c in sorted(df.columns):
        m = re.match('^(.*)_(f|lag)_(d|h|w)_(\d{3})$', c)
        m2 = re.match('^(consumption_[dh]_mean|is_shutdown).*_last_(\d+)d$', c)
        if m:
            feat, flag, unit, num = m.group(1), m.group(2), m.group(3), int(m.group(4), 10)
            num_hours = {'h': 1, 'd': 24, 'w': 24 * 7}.get(unit) * num

            assert feat in ['consumption', 'target',
                'temperature',
                'is_day_off',
                'is_eq_target_day_off',
                'is_holiday_us', 'is_holiday_fra', 'is_holiday_custom'
            ]

            if flag == 'f' and num_hours >= target_days * 24:
                continue
            elif flag == 'lag' and num_hours > cold_start_days * 24:
                if feat in ['is_day_off', 'is_eq_target_day_off'] and force_is_day_off:
                    if num_hours > 7 * 24:
                        continue
                else:
                    continue

            if feat in ['target'] and unit != prediction_unit:
                continue

            if feat in ['consumption'] and prediction_unit == 'h' and unit != 'h':
                continue
            if feat in ['consumption'] and prediction_unit in ['d', 'w'] and unit != 'd':
                continue

            res[c] = df[c]
            if feat == 'target':
                targets.append(c)
            else:
                features[flag].append(c)
        elif m2:
            unit = m2.group(1).split("_")[1]
            last = int(m2.group(2))
            if unit == 'h' and prediction_unit != 'h':
                continue
            elif unit == 'd' and prediction_unit == 'h':
                continue
            if min(7, cold_start_days) != last:
                continue
            res[c] = df[c]
            features['other'].append(c)
        elif re.match('^leaking', c):
            if (prediction_unit == 'h' and 'h_mean' in c) or (prediction_unit != 'h' and 'd_mean' in c):
                res[c] = df[c]
                features['other'].append(c)
        elif c in ['working_days', 'is_shutdown', 'target_mean_change_d_000', 
            'consumption_min_d', 'consumption_max_d'] or \
            c in clusters_columns or \
                re.match('^is_dayofweek_\d+', c):
            res[c] = df[c]
            features['other'].append(c)
        elif c in [
            'cold_start_days', 'target_days', 'entry_type', 'date', 'timestamp',
            'submission_timestamp', 'series_id', 'prediction_window', 'k'
        ]:
            res[c] = df[c]
        else:
            raise Exception(f"Unknown column: {c}")
    final_features = list(reversed(features['lag'])) + features['other'] + features['f']
    return res, final_features, targets


def prepare_values_for_nn(df, features, targets, prediction_window, cold_start_days=None, scale_min_adj=0.9, scale_max_adj=4.0,
    false_is_negative=True, flags_vs_day0=True):
    params = prediction_params(prediction_window)

    res = df.copy()
    bool_columns = res[features].filter(regex='^is_').columns
    c_columns = res[features].filter(regex='^(leaking_)?consumption_(lag|h_mean|d_mean|min|max)').columns
    t_columns = res[features].filter(regex='^temperature').columns

    res['scale_min'] = res[c_columns].min(axis=1) * scale_min_adj
    res['scale_max'] = res[c_columns].max(axis=1) * scale_max_adj
    res['scale_delta'] = res['scale_max'] - res['scale_min']
    res['scale_delta'] = pd.concat([res['scale_delta'], res['scale_min']], axis=1)\
        .max(axis=1)
    f = 1 if prediction_window in ['hourly', 'daily', 'daily1'] else 7
    res['scale_y_mult'] = f

    res['ci'] = params['wi'] / res[targets].mean(axis=1)
    res['wi'] = params['wi']
    res['sample_weight'] = res['scale_delta'] * res['ci'] * res['scale_y_mult'] * res['k'] / params['wi']

    def scale_down(sel, cols, d=1):
        sel.loc[:, cols] = sel.loc[:, cols]\
            .divide(d)\
            .subtract(sel['scale_min'], axis=0)\
            .divide(sel['scale_delta'], axis=0)

    def scale_down_bool(sel, cols):
        sel.loc[:, cols] = sel.loc[:, cols].applymap(float)
        if false_is_negative:
            logging.info("scaling to negative!")
            sel.loc[:, cols] = sel.loc[:, cols].multiply(2).subtract(1)

    def scale_up(y_output, x_input):
        return y_output\
            .multiply(x_input['scale_delta'], axis=0)\
            .add(x_input['scale_min'], axis=0)\
            .multiply(x_input['scale_y_mult'], axis=0)\
            .clip(0, None)

    temp_delta = 20
    base_temp = 17.5
    res.loc[:, t_columns] = res.loc[:, t_columns].subtract(base_temp)\
        .abs().clip(0, temp_delta).divide(temp_delta)

    scale_down(res, c_columns, d=1)
    scale_down(res, targets, d=f)
    if prediction_window == 'hourly' and flags_vs_day0:
        logging.info("scaling vs day0")

        for c in res.filter(regex='^(is_day_off|is_holiday_(custom|fra|us))_lag_d').columns:
            cc = re.sub('lag_d_\d+', 'f_d_000', c)
            logging.info("- scaling %s vs %s", c, cc)
            res[c] = res[c] == res[cc]
    scale_down_bool(res, bool_columns)
    return res, scale_up

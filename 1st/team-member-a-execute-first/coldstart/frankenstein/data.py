"""
Functions for preparing data for the model
"""
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler

from coldstart.definitions import WINDOW_TO_PRED_DAYS, SURFACE_OHE, BASE_TEMPERATURE_OHE
from coldstart.utils import group_sum, _get_next_date, _get_next_weekday, _is_day_off, _is_holiday, group_mean
from coldstart.clusters import get_cluster_ohe, get_cluster_features_v2
from coldstart.validation import split_series_id, stratified_cv_series_id
from coldstart.utils import load_data

def prepare_data_for_train(df, input_days, window, verbose=True):
    """
    Returns
    --------

    ::

        x = {
            'past_consumption': past_consumption,
            'cluster_features_v2': cluster_features_v2,
            'past_weekday': past_weekday,
            'future_weekday': future_weekday,
            'past_day_off': past_day_off,
            'future_day_off': future_day_off,
        }
        return x, future_consumption
    """
    pred_days = WINDOW_TO_PRED_DAYS[window]
    past_consumption, future_consumption = [], []
    past_day_off, future_day_off = [], []
    past_weekday, future_weekday = [], []
    cluster_features_v2 = []
    if verbose:
        iterator = tqdm_notebook(df.series_id.unique(), desc='Preparing data')
    else:
        iterator = df.series_id.unique()
    for series_id in iterator:
        sub_df = df[df.series_id == series_id]
        consumption = sub_df.consumption.values
        days_off = sub_df.is_holiday.values
        weekdays = sub_df.weekday.values

        if window != 'hourly':
            consumption = group_sum(consumption, 24)
            days_off = days_off[::24]
            weekdays = weekdays[::24]
            step = 1
            past_samples = input_days
            future_samples = pred_days
        else:
            step = 24
            past_samples = input_days*24
            future_samples = 24
        for start_idx in range(0, len(consumption)-future_samples-past_samples+step, step):
            past_idx = start_idx + past_samples
            past_day_off.append(days_off[start_idx:past_idx])
            past_weekday.append(weekdays[start_idx:past_idx])

            future_idx = past_idx + future_samples
            if window == 'weekly':
                future_day_off.append(np.reshape(days_off[past_idx:future_idx], (2, -1)))
                future_weekday.append(weekdays[past_idx:future_idx:7])
            else:
                future_day_off.append(days_off[past_idx:future_idx])
                future_weekday.append(weekdays[past_idx:future_idx])

            past_consumption_values = consumption[start_idx:past_idx]
            mean_value = np.mean(past_consumption_values)
            mean_value *= normalization_factor(past_day_off[-1], future_day_off[-1])
            if window == 'weekly':
                mean_value *= 7
            past_consumption.append(past_consumption_values/mean_value)
            if window == 'weekly':
                future_consumption.append(group_sum(consumption[past_idx:future_idx], 7)/mean_value)
            else:
                future_consumption.append(consumption[past_idx:future_idx]/mean_value)
            cluster_features_v2.append(get_cluster_features_v2(series_id))
            # TODO: I should do refinement on weekly predictions

    past_consumption = np.array(past_consumption, dtype=np.float32)
    past_consumption = np.expand_dims(past_consumption, 2)
    future_consumption = np.array(future_consumption, dtype=np.float32)
    future_consumption = np.expand_dims(future_consumption, 2)

    past_day_off = np.array(past_day_off, dtype=np.float32)
    past_day_off = np.expand_dims(past_day_off, 2)
    future_day_off = np.array(future_day_off, dtype=np.float32)
    if window != 'weekly':
        future_day_off = np.expand_dims(future_day_off, 2)

    past_weekday = [[_weekday_ohe(weekday) for weekday in week] for week in past_weekday]
    past_weekday = np.array(past_weekday, dtype=np.float32)
    future_weekday = [[_weekday_ohe(weekday) for weekday in week] for week in future_weekday]
    future_weekday = np.array(future_weekday, dtype=np.float32)

    cluster_features_v2 = np.array(cluster_features_v2, dtype=np.float32)

    x = {
        'past_consumption': past_consumption,
        'cluster_features_v2': cluster_features_v2,
        'past_weekday': past_weekday,
        'future_weekday': future_weekday,
        'past_day_off': past_day_off,
        'future_day_off': future_day_off,
    }
    return x, future_consumption

def _weekday_ohe(weekday):
    ohe = np.zeros(7)
    ohe[weekday] = 1
    return ohe

def normalization_factor(input_days, output_days, ratio=0.71):
    """
    The inputs are binary values where 1 represents being a day off and
    0 represents being a working day
    """
    onput_factors = np.ones_like(output_days, dtype=np.float)
    onput_factors[output_days == 1] = ratio
    input_factors = np.ones_like(input_days, dtype=np.float)
    input_factors[input_days == 1] = ratio
    factor = np.mean(onput_factors)/np.mean(input_factors)
    return factor

def load_and_arrange_data(conf, verbose=False):
    """
    Prepares the data for training using cross-validation
    """
    train, test, _, metadata = load_data()

    if 'random_seed' in conf:
        print('Using random seed')
        train_ids, val_ids = stratified_cv_series_id(
            train.series_id.unique(), fold_idx=conf['fold_idx'], random_seed=conf['random_seed'])
    else:
        train_ids, val_ids = split_series_id(train.series_id.unique(), fold_idx=conf['fold_idx'])
    val = train[train.series_id.isin(val_ids)]
    val.reset_index(inplace=True, drop=True)
    train = train[train.series_id.isin(train_ids)]
    train.reset_index(inplace=True, drop=True)

    train = pd.concat([train, test])
    train.reset_index(inplace=True, drop=True)

    _train_x, train_y = prepare_data_for_train(
        train, conf['input_days'], conf['window'], verbose=verbose)
    _val_x, val_y = prepare_data_for_train(
        val, conf['input_days'], conf['window'], verbose=verbose)

    train_x, val_x = {}, {}
    train_x['past_features'] = np.concatenate([_train_x['past_%s' % key] \
        for key in conf['past_features']], axis=2)
    val_x['past_features'] = np.concatenate([_val_x['past_%s' % key] \
        for key in conf['past_features']], axis=2)

    train_x['future_features'] = np.concatenate([_train_x['future_%s' % key] \
        for key in conf['future_features']], axis=2)
    val_x['future_features'] = np.concatenate([_val_x['future_%s' % key] \
        for key in conf['future_features']], axis=2)

    train_x['cluster_features'] = np.concatenate([_train_x[key] \
        for key in conf['cluster_features']], axis=1)
    val_x['cluster_features'] = np.concatenate([_val_x[key] \
        for key in conf['cluster_features']], axis=1)

    return train_x, train_y, val_x, val_y

def prepare_x(window, df, metadata, series_id):
    x = {}
    x['past_consumption'], mean_consumption = _prepare_past_consumption(window, df)
    x['past_day_off'] = _prepare_past_day_off(window, df)
    x['past_weekday'] = _prepare_past_weekday(window, df)
    x['cluster_features_v2'] = _prepare_cluster_features_v2(series_id)

    x['future_day_off'] = _prepare_future_day_off(window, df, metadata, series_id)
    x['future_weekday'] = _prepare_future_weekday(window, df)

    factor = normalization_factor(x['past_day_off'], x['future_day_off'])
    mean_consumption *= factor
    x['past_consumption'] /= factor


    X = {}
    X['past_features'] = np.concatenate(
        [x[key] for key in ['past_consumption', 'past_day_off', 'past_weekday']],
        axis=2)
    X['future_features'] = np.concatenate(
        [x[key] for key in ['future_day_off', 'future_weekday']], axis=2)
    X['cluster_features'] = x['cluster_features_v2']
    return X, mean_consumption

def _prepare_past_consumption(window, df):
    past_consumption = df.consumption.values.copy()
    if window != 'hourly':
        past_consumption = group_sum(past_consumption, 24)
    mean_consumption = np.mean(past_consumption)
    if window == 'weekly':
        mean_consumption *= 7
    past_consumption /= mean_consumption
    past_consumption = np.expand_dims(past_consumption, axis=1)
    past_consumption = np.array(past_consumption, dtype=np.float32)
    past_consumption = np.expand_dims(past_consumption, axis=0)
    return past_consumption, mean_consumption

def _prepare_past_day_off(window, df):
    is_day_off = df.is_holiday.values[::24].tolist()
    is_day_off = np.array(is_day_off, dtype=np.float32)
    if window == 'hourly':
        is_day_off = np.repeat(is_day_off, 24, axis=0)
    is_day_off = np.expand_dims(is_day_off, axis=0)
    is_day_off = np.expand_dims(is_day_off, axis=2)
    return is_day_off

def _prepare_past_weekday(window, df):
    weekday = df.weekday.values
    if window != 'hourly':
        weekday = weekday[::24]
    weekday = [_weekday_ohe(day) for day in weekday]
    weekday = np.array(weekday, dtype=np.float32)
    weekday = np.expand_dims(weekday, axis=0)
    return weekday

def _prepare_future_day_off(window, df, metadata, series_id):
    is_day_off = []
    current_date = df.timestamp.values[-1]
    current_weekday = df.weekday.values[-1]
    for _ in range(WINDOW_TO_PRED_DAYS[window]):
        current_date = _get_next_date(current_date)
        current_weekday = _get_next_weekday(current_weekday)
        current_day_is_off = _is_day_off(series_id, current_weekday, metadata)
        current_day_is_off = current_day_is_off or _is_holiday(current_date)
        is_day_off.append(current_day_is_off)
    is_day_off = np.array(is_day_off, dtype=np.float32)
    if window == 'hourly':
        is_day_off = np.repeat(is_day_off, 24, axis=0)
    if window == 'weekly':
        is_day_off = np.reshape(is_day_off, (2, -1))
    else:
        is_day_off = np.expand_dims(is_day_off, axis=1)
    is_day_off = np.expand_dims(is_day_off, axis=0)
    return is_day_off

def _prepare_future_weekday(window, df):
    weekday = []
    current_weekday = df.weekday.values[-1]
    for _ in range(WINDOW_TO_PRED_DAYS[window]):
        current_weekday = _get_next_weekday(current_weekday)
        weekday.append(current_weekday)
    weekday = [_weekday_ohe(day) for day in weekday]
    weekday = np.array(weekday, dtype=np.float32)

    if window == 'hourly':
        weekday = np.repeat(weekday, 24, axis=0)
    elif window == 'weekly':
        weekday = weekday[::7]

    weekday = np.expand_dims(weekday, axis=0)
    return weekday

def _prepare_cluster_features_v2(series_id):
    cluster_ohe = get_cluster_features_v2(series_id)
    cluster_ohe = np.array(cluster_ohe, dtype=np.float32)
    cluster_ohe = np.expand_dims(cluster_ohe, axis=0)
    return cluster_ohe
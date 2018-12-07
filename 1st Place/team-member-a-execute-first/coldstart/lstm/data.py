"""
Functions for preparing data for the model
"""
import numpy as np
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler

from coldstart.definitions import WINDOW_TO_PRED_DAYS, SURFACE_OHE, BASE_TEMPERATURE_OHE
from coldstart.utils import group_sum, _get_next_date, _get_next_weekday, _is_day_off, _is_holiday, group_mean
from coldstart.clusters import get_cluster_ohe, get_cluster_features_v2


def prepare_data_for_train(df, metadata, input_days, window, verbose=True):
    pred_days = WINDOW_TO_PRED_DAYS[window]
    past_consumption, future_consumption = [], []
    is_day_off, cluster_features_v2 = [], []
    clock = []
    if verbose:
        iterator = tqdm_notebook(df.series_id.unique(), desc='Preparing data')
    else:
        iterator = df.series_id.unique()
    for series_id in iterator:
        sub_df = df[df.series_id == series_id]
        consumption = sub_df.consumption.values
        days_off = sub_df.is_holiday.values
        if window != 'hourly':
            consumption = group_sum(consumption, 24)
            days_off = days_off[::24]
            step = input_days
        else:
            clock_values = np.concatenate([np.linspace(0, 1, 24)]*(len(sub_df)//24), axis=0)
            step = input_days*24
        for start_idx in range(len(consumption)-step-1):
            is_day_off.append(days_off[start_idx:start_idx + step])
            past_consumption_values = consumption[start_idx:start_idx + step]
            mean_value, std_value = np.mean(past_consumption_values), np.std(past_consumption_values)
            past_consumption.append(past_consumption_values/mean_value)
            future_consumption.append(consumption[start_idx + step]/mean_value)
            cluster_features_v2.append(get_cluster_features_v2(series_id))
            clock.append(clock_values[start_idx:start_idx + step])

    past_consumption = np.array(past_consumption, dtype=np.float32)
    past_consumption = np.expand_dims(past_consumption, 2)
    future_consumption = np.array(future_consumption, dtype=np.float32)
    is_day_off = np.array(is_day_off, dtype=np.float32)
    is_day_off = np.expand_dims(is_day_off, 2)
    cluster_features_v2 = np.array(cluster_features_v2, dtype=np.float32)
    cluster_features_v2 = np.expand_dims(cluster_features_v2, 1)
    cluster_features_v2 = np.repeat(cluster_features_v2, is_day_off.shape[1], axis=1)
    clock = np.array(clock, dtype=np.float32)
    clock = np.expand_dims(clock, 2)

    x = {
        'past_consumption': past_consumption,
        'is_day_off': is_day_off,
        'cluster_features_v2': cluster_features_v2,
        'clock': clock,
    }
    return x, future_consumption

def prepare_x(window, df, metadata, series_id):
    x = {}
    x['past_consumption'], mean_consumption = _prepare_past_consumption(window, df)
    x['is_day_off'] = _prepare_is_day_off(window, df, metadata, series_id)
    ret = _prepare_cluster_features_v2(series_id)
    ret = np.expand_dims(ret, axis=0)
    x['cluster_features_v2'] = np.repeat(ret, x['past_consumption'].shape[1], axis=1)
    x['clock'] = _prepare_clock(x['is_day_off'].shape[1]//24)

    sorted_keys = ['past_consumption', 'is_day_off', 'cluster_features_v2', 'clock']
    x = np.concatenate([x[key] for key in sorted_keys], axis=2)
    return x, mean_consumption

def _prepare_clock(n_days):
    clock = np.concatenate([np.linspace(0, 1, 24)]*(n_days), axis=0)
    clock = np.array(clock, dtype=np.float32)
    clock = np.expand_dims(clock, 0)
    clock = np.expand_dims(clock, 2)
    return clock

def _prepare_past_consumption(window, df):
    past_consumption = df.consumption.values.copy()
    if window != 'hourly':
        past_consumption = group_sum(past_consumption, 24)
    mean_consumption = np.mean(past_consumption)
    past_consumption /= mean_consumption
    pred_size = WINDOW_TO_PRED_DAYS[window]
    if window == 'hourly':
        pred_size *= 24
    pred = np.zeros(pred_size)
    past_consumption = np.concatenate([past_consumption, pred])
    past_consumption = np.expand_dims(past_consumption, axis=1)
    past_consumption = np.array(past_consumption, dtype=np.float32)
    past_consumption = np.expand_dims(past_consumption, axis=0)
    return past_consumption, mean_consumption

def _prepare_is_day_off(window, df, metadata, series_id):
    is_day_off = df.is_holiday.values[::24].tolist()
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
    is_day_off = np.expand_dims(is_day_off, axis=0)
    is_day_off = np.expand_dims(is_day_off, axis=2)
    return is_day_off

def _prepare_cluster_features_v2(series_id):
    cluster_ohe = get_cluster_features_v2(series_id)
    cluster_ohe = np.array(cluster_ohe, dtype=np.float32)
    cluster_ohe = np.expand_dims(cluster_ohe, axis=0)
    return cluster_ohe

"""
Functions for preparing data for the model
"""
import numpy as np
from tqdm import tqdm_notebook

from coldstart.definitions import WINDOW_TO_PRED_DAYS, SURFACE_OHE, BASE_TEMPERATURE_OHE
from coldstart.utils import group_sum, _get_next_date, _get_next_weekday, _is_day_off, _is_holiday, group_mean
from coldstart.clusters import get_cluster_ohe, get_cluster_features_v2


def prepare_data_for_train(df, metadata, input_days, window, verbose=True):
    pred_days = WINDOW_TO_PRED_DAYS[window]
    past_consumption, future_consumption = [], []
    is_day_off, data_trend, metadata_ohe = [], [], []
    metadata_days_off  = []
    cluster_id_ohe, cluster_features_v2 = [], []
    if verbose:
        iterator = tqdm_notebook(df.series_id.unique(), desc='Preparing data')
    else:
        iterator = df.series_id.unique()
    for series_id in iterator:
        sub_df = df[df.series_id == series_id]
        consumption = sub_df.consumption.values
        if window != 'hourly':
            consumption = group_sum(consumption, 24)
        series_is_day_off = [int(value) for value in sub_df.is_holiday.values[::24]]
        for start_idx in range(len(series_is_day_off)-input_days - pred_days + 1):
            is_day_off.append(series_is_day_off[start_idx:start_idx + input_days + pred_days])
            val_idx = start_idx + input_days
            if window == 'hourly':
                x = np.reshape(consumption[start_idx*24: val_idx*24], newshape=(-1, 24))
                y = consumption[val_idx*24:(val_idx+pred_days)*24]
            else:
                x = consumption[start_idx: val_idx]
                x = np.expand_dims(x, axis=1)
                y = consumption[val_idx: val_idx+pred_days]
                if window == 'weekly':
                    x = np.repeat(x, 2, axis=1)
                    y = group_sum(y, 7)
                else:
                    x = np.repeat(x, 7, axis=1)
            y_mean = np.mean(y)
            past_consumption.append(x/y_mean)
            future_consumption.append(y/y_mean)
            # Data trend
            if window == 'hourly':
                _data_trend = group_sum(consumption[start_idx*24: val_idx*24], 24)
            else:
                _data_trend = consumption[start_idx: val_idx].copy()
            _data_trend /= np.mean(_data_trend)
            data_trend.append(_data_trend)
            metadata_ohe.append(_get_metadata_ohe(metadata, series_id))
            metadata_days_off.append(_get_metadata_days_off(metadata, series_id))
            cluster_id_ohe.append(get_cluster_ohe(series_id))
            cluster_features_v2.append(get_cluster_features_v2(series_id))

    past_consumption = np.array(past_consumption, dtype=np.float32)
    past_consumption = np.transpose(past_consumption, axes=(0, 2, 1))
    future_consumption = np.array(future_consumption, dtype=np.float32)
    is_day_off = np.array(is_day_off, dtype=np.float32)
    is_day_off[is_day_off == 0] = -1
    data_trend = np.array(data_trend, dtype=np.float32)
    metadata_ohe = np.array(metadata_ohe, dtype=np.float32)
    metadata_days_off = np.array(metadata_days_off, dtype=np.float32)
    cluster_id_ohe = np.array(cluster_id_ohe, dtype=np.float32)
    cluster_features_v2 = np.array(cluster_features_v2, dtype=np.float32)
    x = {
        'past_consumption': past_consumption,
        'is_day_off': is_day_off,
        'data_trend': data_trend,
        'metadata_ohe': metadata_ohe,
        'metadata_days_off': metadata_days_off,
        'cluster_id_ohe': cluster_id_ohe,
        'cluster_features_v2': cluster_features_v2,
    }
    return x, future_consumption

def prepare_x(window, df, metadata, series_id):
    x = {}
    x['past_consumption'] = _prepare_past_consumption(window, df)
    x['is_day_off'] = _prepare_is_day_off(window, df, metadata, series_id)
    x['metadata_ohe'] = _prepare_metadata_ohe(metadata, series_id)
    x['metadata_days_off'] = _prepare_metadata_days_off(metadata, series_id)
    x['data_trend'] = _prepare_data_trend(window, df)
    x['cluster_id_ohe'] = _prepare_cluster_id_ohe(series_id)
    x['cluster_features_v2'] = _prepare_cluster_features_v2(series_id)
    return x

def _prepare_past_consumption(window, df):
    consumption = df.consumption.values
    if window == 'hourly':
        past_consumption = np.reshape(consumption, newshape=(-1, 24))
    else:
        past_consumption = group_sum(consumption, 24)
        past_consumption = np.expand_dims(past_consumption, axis=1)
        if window == 'weekly':
            past_consumption = np.repeat(past_consumption, 2, axis=1)
        else:
            past_consumption = np.repeat(past_consumption, 7, axis=1)
    past_consumption = np.array(past_consumption, dtype=np.float32)
    past_consumption = np.expand_dims(past_consumption, axis=0)
    past_consumption = np.transpose(past_consumption, axes=(0, 2, 1))
    return past_consumption

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
    is_day_off[is_day_off == 0] = -1
    is_day_off = np.expand_dims(is_day_off, axis=0)
    return is_day_off

def _get_metadata_ohe(metadata, series_id):
    row = metadata.loc[series_id]
    surface = SURFACE_OHE[row['surface']]
    base_temperature = BASE_TEMPERATURE_OHE[row['base_temperature']]
    return surface + base_temperature

def _get_metadata_days_off(metadata, series_id):
    row = metadata.loc[series_id]
    days_off = row.values[-7:].astype(np.int)
    days_off[days_off == 0] = -1
    return days_off

def _prepare_metadata_ohe(metadata, series_id):
    metadata_ohe = _get_metadata_ohe(metadata, series_id)
    metadata_ohe = np.array(metadata_ohe, dtype=np.float32)
    metadata_ohe = np.expand_dims(metadata_ohe, axis=0)
    return metadata_ohe

def _prepare_metadata_days_off(metadata, series_id):
    metadata_days_off = _get_metadata_days_off(metadata, series_id)
    metadata_days_off = np.array(metadata_days_off, dtype=np.float32)
    metadata_days_off = np.expand_dims(metadata_days_off, axis=0)
    return metadata_days_off

def _prepare_cluster_id_ohe(series_id):
    cluster_ohe = get_cluster_ohe(series_id)
    cluster_ohe = np.array(cluster_ohe, dtype=np.float32)
    cluster_ohe = np.expand_dims(cluster_ohe, axis=0)
    return cluster_ohe

def _prepare_cluster_features_v2(series_id):
    cluster_ohe = get_cluster_features_v2(series_id)
    cluster_ohe = np.array(cluster_ohe, dtype=np.float32)
    cluster_ohe = np.expand_dims(cluster_ohe, axis=0)
    return cluster_ohe

def _prepare_data_trend(window, df):
    consumption = df.consumption.values
    data_trend = group_sum(consumption, 24)
    data_trend /= np.mean(data_trend)
    data_trend = np.array(data_trend, dtype=np.float32)
    data_trend = np.expand_dims(data_trend, axis=0)
    return data_trend

def _replace_missing_values_in_temperature(temperatures):
    """
    If all values are missing returns the mean temperature (15.17)
    Otherwise replaces the missing values by the mean temperature
    """
    mean_temperature = np.nanmean(temperatures)
    if np.isnan(mean_temperature):
        temperatures = np.ones_like(temperatures)*15.17
    else:
        temperatures = temperatures.copy()
        temperatures[np.isnan(temperatures)] = mean_temperature
    return temperatures

def _normalize_temperature(temperatures):
    return (temperatures-15.17)/9

# coding: utf-8

import os

# math and data manipulation
import numpy as np
import pandas as pd

from tqdm import tqdm
import lightgbm as lgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from datetime import timedelta, datetime

root_path = '../input/'

consumption_train = pd.read_csv(root_path + 'consumption_train.csv', index_col=0, parse_dates=['timestamp'])
meta = pd.read_csv(root_path + 'meta.csv')
cold_start_test = pd.read_csv(root_path + 'cold_start_test.csv', index_col=0, parse_dates=['timestamp'])
submission_format = pd.read_csv(root_path + 'submission_format.csv', parse_dates=['timestamp'])
df = pd.concat([consumption_train, cold_start_test], axis=0).reset_index(drop=True)

dict_hour_minmax_cons = {}
for id_ in tqdm(meta.series_id.unique()):
    dict_hour_minmax_cons[id_] = [df[df.series_id == id_].consumption.min(),
                                  df[df.series_id == id_].consumption.max(),
                                  df[df.series_id == id_].consumption.mean()]
lag_day = 7 * 2
lag_hour = 24 * 7 * 2
lag_week = 2


def add_basic_features(df_1, meta, mode='hourly'):
    df_1['month'] = df_1.timestamp.dt.month
    df_1['day'] = df_1.timestamp.dt.day
    df_1['day_of_week'] = df_1.timestamp.dt.dayofweek
    if mode == 'hourly':
        df_1['hour'] = df_1.timestamp.dt.hour
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    meta_data = meta.values
    dict_off_days = {}
    for i, id_ in enumerate(list(meta.series_id.unique())):
        dict_off_days[id_] = {}
        for idx in range(7):
            dict_off_days[id_][days[idx]] = meta_data[i, 3 + idx]
    df_1 = pd.merge(df_1, meta, how='left', on='series_id')
    df_1['week_day'] = df_1.timestamp.dt.day_name()
    cols = df_1.columns.tolist()
    idx_id = [i for i in range(len(cols)) if cols[i] == 'series_id'][0]
    idx_wd = [i for i in range(len(cols)) if cols[i] == 'week_day'][0]

    def is_off_day(x):
        id_ = x[idx_id]
        week_day = x[idx_wd]
        return dict_off_days[id_][week_day]
    df_1['is_off_day'] = df_1.apply(is_off_day, axis=1)
    dict_surface = {
        'large': 0, 'medium': 1, 'small': 2, 'x-large': 3, 'x-small': 4, 'xx-large': 5, 'xx-small': 6
    }
    dict_base_temp = {
        'low': 1, 'high': 0
    }
    df_1.surface = df_1.surface.apply(lambda x: dict_surface[x])
    df_1.base_temperature = df_1.base_temperature.apply(lambda x: dict_base_temp[x])
    df_1.drop(['week_day'], axis=1, inplace=True)
    if mode == 'hourly':
        one_hot_cols = ['month', 'hour', 'day_of_week', 'surface', 'base_temperature', 'is_off_day']
    else:
        one_hot_cols = ['month', 'day_of_week', 'surface', 'base_temperature', 'is_off_day']

    def apply_ohe(df, one_hot_cols):

        df_tmp = pd.get_dummies(data=df, columns=one_hot_cols, drop_first=True)
        df = pd.concat([df, df_tmp], axis=1)

        return df_tmp
    df_1 = apply_ohe(df_1, one_hot_cols=one_hot_cols)
    dropped_cols = [
        'monday_is_day_off', 'tuesday_is_day_off', 'wednesday_is_day_off',
        'thursday_is_day_off', 'friday_is_day_off', 'saturday_is_day_off', 'sunday_is_day_off']
    df_1.drop(dropped_cols, axis=1, inplace=True)
    return df_1


def add_stat_consumptions(df):
    df['con_hour_min'] = df.series_id.apply(lambda x: dict_hour_minmax_cons[x][0])
    df['con_hour_max'] = df.series_id.apply(lambda x: dict_hour_minmax_cons[x][1])
    df['con_hour_mean'] = df.series_id.apply(lambda x: dict_hour_minmax_cons[x][2])
    return df


def prepare_training_data_week(df_1):
    df = df_1.copy()
    data_all = []
    series_ids = df.series_id.unique()
    dict_df_ids = {}
    for id_ in tqdm(series_ids):
        dict_df_ids[id_] = df[df.series_id == id_]
    for i in tqdm(range(df.shape[0])):
        data_tmp = []
        id_ = df.get_value(i, 'series_id')
        dt = df.get_value(i, 'timestamp')
        data_tmp += [id_, dt]
        df_id = dict_df_ids[id_]
        df_id_after = df_id[df_id.timestamp > dt]
        if df_id_after.shape[0] < 7:
            continue
        consumption_after_week = np.sum(df_id_after.consumption.tolist()[:7])
        temp = np.nanmean(df_id_after.temperature.tolist()[:7])
        data_tmp += [consumption_after_week, temp]
        df_id_before = df_id[df_id.timestamp <= dt].reset_index(drop=True)
        len_ = df_id_before.shape[0]
        if len_ >= 21:
            df_id_before_21 = df_id_before[-21:]
            consumption_before_21 = df_id_before_21.consumption.tolist()[::-1]
            data_tmp += [np.sum(consumption_before_21[:7]), np.sum(consumption_before_21[7:14]),
                         np.sum(consumption_before_21[14:])]
        else:
            consumption_before = df_id_before.consumption.tolist()[::-1]
            if len_ >= 14:
                data_tmp += [np.sum(consumption_before[:7]), np.sum(consumption_before[7:14]), np.NaN]
            elif len_ >= 7:
                data_tmp += [np.sum(consumption_before[:7]), np.NaN, np.NaN]
            else:
                data_tmp += [np.NaN, np.NaN, np.NaN]
            consumption_before_21 = df_id_before.consumption.tolist()[::-1] + [np.NaN] * (21 - len_)
        data_tmp += consumption_before_21
        data_all.append(data_tmp)
        temperature_before = df_id_before.temperature.tolist()
        len_b = len(temperature_before)
        if len_b >= 21:
            temperature_before_21 = temperature_before[-21:][::-1]
            data_tmp += [np.nanmean(temperature_before_21[:7]), np.nanmean(temperature_before_21[7:14]),
                         np.nanmean(temperature_before_21[14:])]
        else:
            temperature_before_21 = temperature_before[::-1] + [np.NaN] * (21 - len(temperature_before))
            if len_b >= 14:
                data_tmp += [np.nanmean(temperature_before_21[:7]),
                             np.nanmean(temperature_before_21[7:14]), np.NaN]
            elif len_b >= 7:
                data_tmp += [np.nanmean(temperature_before_21[:7]), np.NaN, np.NaN]
            else:
                data_tmp += [np.NaN, np.NaN, np.NaN]
    cols = ['series_id', 'timestamp', 'consumption', 'temperature',
            'consumption_prev_week_1', 'consumption_prev_week_2', 'consumption_prev_week_3'] + \
        ['consumption_prev_day_' + str(i) for i in range(1, 22)] + \
        ['temperature_prev_week_1', 'temperature_prev_week_2', 'temperature_prev_week_3']
    df_res = pd.DataFrame(data=data_all, columns=cols)
    return df_res


def create_lagged_features(df, lag=7, mode='hourly'):
    if not type(df) == pd.DataFrame:
        df = pd.DataFrame(df, columns=['consumption'])

    def _rename_lag(ser, j):
        if mode == 'hourly':
            ser.name = ser.name + f'_prev_hour_{j}'
        elif mode == 'daily':
            ser.name = ser.name + f'_prev_day_{j}'
        elif mode == 'weekly':
            ser.name = ser.name + f'_prev_week_{j}'
        return ser
    for i in range(1, lag + 1):
        df = df.join(df.consumption.shift(i).pipe(_rename_lag, i))

    return df


def prepare_training_data(df, lag, use_scaler=False, mode='hourly'):
    dfs = []
    for i, id_ in tqdm(list(enumerate(df.series_id.unique()))):
        consumption_vals = df[df.series_id == id_]['consumption'].values
        if use_scaler:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            consumption_vals = scaler.fit_transform(consumption_vals.reshape(-1, 1))
        consumption_lagged = create_lagged_features(consumption_vals, lag, mode)
        dfs.append(consumption_lagged)

    df_processed = pd.concat(dfs, axis=0)
    df.drop(['consumption'], axis=1, inplace=True)
    df = pd.concat([df.reset_index(drop=True), df_processed.reset_index(drop=True)], axis=1)
    return df


def prepare_test_data(df, mode='hourly'):
    df_1 = df[df.prediction_window == 'hourly']
    df_2 = df[df.prediction_window == 'daily']
    df_2.timestamp = df_2.timestamp + timedelta(seconds=3600 * 23)
    df_3 = df[df.prediction_window == 'weekly']
    df_3.timestamp = df_3.timestamp + timedelta(seconds=3600*23)
    if mode == 'hourly':
        dfs = [df_1]
        for i in tqdm(range(24)):
            dfs.append(df_2.copy())  # copy() is needed
            df_2.timestamp = df_2.timestamp - timedelta(seconds=3600)
        dfs2 = []
        for i in tqdm(range(24)):
            dfs2.append(df_3.copy())
            df_3.timestamp = df_3.timestamp - timedelta(seconds=3600)
        df_4 = pd.concat(dfs2, axis=0)
        for i in tqdm(range(7)):
            dfs.append(df_4.copy())
            df_4.timestamp = df_4.timestamp - timedelta(seconds=3600*24)
    elif mode == 'daily':
        dfs = [df_2]
        for i in tqdm(range(7)):
            dfs.append(df_3.copy())
            df_3.timestamp = df_3.timestamp - timedelta(seconds=3600*24)

    df_1 = pd.concat(dfs, axis=0)
    df_1 = df_1.sort_values(by=['series_id', 'timestamp']).reset_index(drop=True)
    return df_1


def generate_data(df, mode='hourly', use_scaler=False):
    df_train = df.copy()
    if mode == 'hourly':
        lag = lag_hour
        consumption_cols = ['consumption_prev_hour_' + str(i) for i in range(1, lag + 1)]
    elif mode == 'daily':
        lag = lag_day
        consumption_cols = ['consumption_prev_day_' + str(i) for i in range(1, lag + 1)]
    print('Prepare ' + mode + ' data...')
    df_train = prepare_training_data(df_train, lag, use_scaler, mode)
    print('Prepare test data...')
    df_test = prepare_test_data(submission_format, mode)
    exclude_cols = ['prediction_window', 'pred_id']
    df_new_train = df_train[[col for col in df_test.columns.tolist() if col not in exclude_cols]]
    df_all = pd.concat([df_new_train, df_test.drop(exclude_cols,
                                                   axis=1)], axis=0)[df_new_train.columns.tolist()].reset_index(drop=True)
    print('Add basic features...')
    df_all = add_basic_features(df_all, meta, mode)
    df_all = add_stat_consumptions(df_all)

    df_new_train = pd.concat([df_all[:df_train.shape[0]], df_train[consumption_cols]], axis=1)
    df_new_test = pd.concat([df_all[df_train.shape[0]:].reset_index(drop=True),
                             df_test[exclude_cols]], axis=1)

    print('Write to disk...')

    df_new_train.to_csv(root_path + 'cold_start_df_train_' + mode + '_no_scaler_ohe.csv', index=False)
    df_new_test.to_csv(root_path + 'cold_start_df_test_' + mode + '_no_scaler_ohe.csv', index=False)
    return df_new_train, df_new_test


def preprocess_test_day(df_train_hourly):
    df = df_train_hourly.copy()
    df['date'] = df.timestamp.dt.date.apply(lambda x: str(x))
    day_consumption = df.groupby(['series_id', 'date']).consumption.sum().reset_index()
    day_temperature = df.groupby(['series_id', 'date']).temperature.apply(lambda x: np.nanmean(x)).reset_index()
    df = pd.merge(day_consumption, day_temperature, how='left', on=['series_id', 'date'])
    df.columns = [col if col != 'date' else 'timestamp' for col in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df


# Genrate hourly data for hourly model
df_train_hour, df_test_hour = generate_data(df, mode='hourly')

# Genrate daily data for daily model
df_preprocessed = preprocess_test_day(df)
df_train_day, df_test_day = generate_data(df_preprocessed, mode='daily')


# Generate weekly data for weekly model
# df_train_day = pd.read_csv(root_path + 'cold_start_df_train_daily_no_scaler_ohe.csv', parse_dates=['timestamp'])
# df_train_week = prepare_training_data_week(df_train_day)
# df_train_week = add_basic_features(df_train_week, meta, mode='weekly')
# df_train_week = add_stat_consumptions(df_train_week)
# df_train_week.to_csv(root_path + 'cold_start_df_train_weekly_no_scaler_ohe.csv', index=False)

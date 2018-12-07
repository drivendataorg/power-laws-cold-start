# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os

# math and data manipulation
import numpy as np
import pandas as pd

from tqdm import tqdm
import lightgbm as lgb
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import keras

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *

from datetime import timedelta
from sklearn.externals import joblib


# Read data

print('Reading data...')

root_path = '../input/'

consumption_train = pd.read_csv('../input/consumption_train.csv', index_col=0, parse_dates=['timestamp'])
meta = pd.read_csv('../input/meta.csv')
cold_start_test = pd.read_csv('../input/cold_start_test.csv', index_col=0, parse_dates=['timestamp'])
submission_format = pd.read_csv('../input/submission_format.csv', parse_dates=['timestamp'])
# df = pd.concat([consumption_train, cold_start_test], axis=0).reset_index(drop=True)

df_train_hour = pd.read_csv(root_path + "cold_start_df_train_hourly_no_scaler_ohe.csv", parse_dates=['timestamp'])
df_test_hour = pd.read_csv(root_path + 'cold_start_df_test_hourly_no_scaler_ohe.csv', parse_dates=['timestamp'])

df_train_day = pd.read_csv(root_path + "cold_start_df_train_daily_no_scaler_ohe.csv", parse_dates=['timestamp'])
df_test_day = pd.read_csv(root_path + 'cold_start_df_test_daily_no_scaler_ohe.csv', parse_dates=['timestamp'])


# Train features
consumption_cols_day = ['consumption_prev_day_' + str(i) for i in range(1, 7 * 2 + 1)]
consumption_cols_hour = ['consumption_prev_hour_' + str(i) for i in range(1, 24 * 7 * 2 + 1)]
one_hot_cols_daily = [
    'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 
    'month_10', 'month_11', 'month_12', 
    'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 
    'surface_1', 'surface_2', 'surface_3', 'surface_4', 'surface_5', 'surface_6', 
    'is_off_day_True', 'base_temperature_1'
]
train_features = one_hot_cols_daily + consumption_cols_day + consumption_cols_hour[:168] + ['temperature', 'day']
# train_features += ['day', 'tsp_int']
print('Number of lag days: ', len(consumption_cols_day), ' Number of lag hours: ', len(consumption_cols_hour[:168]), 
    'Number of all train features: ', len(train_features))


def day_model_with_hour(hidden_size_lstm, hidden_size_ohe, final_layer_size, num_ohe, lag):
    consumption_input_day = Input(shape=(1, 14), name='input_lstm_day')
    x_day = LSTM(units=hidden_size_lstm, input_shape=(1, 14), 
             return_sequences=True)(consumption_input_day)
#     x = Dropout(0.25)(x)
#     x = LSTM(units=hidden_size_lstm, return_sequences=True)(x)
    x_day = LSTM(units=hidden_size_lstm)(x_day)
#     x = Dropout(0.25)(x)

    consumption_input_hour = Input(shape=(1, 168), name='input_lstm_hour')
    x_hour = LSTM(units=hidden_size_lstm, input_shape=(1, lag), 
             return_sequences=True)(consumption_input_hour)
    x_hour = Dropout(0.5)(x_hour)
#     x = LSTM(units=hidden_size_lstm, return_sequences=True)(x)
    x_hour = LSTM(units=hidden_size_lstm)(x_hour)
    
    ohe_input = Input(shape=(num_ohe,), name='input_ohe')
    y = Dropout(0.5)(ohe_input)
    y = Dense(hidden_size_ohe, activation='elu')(ohe_input)
#     y = Dropout(0.25)(y)
#     y = Dense(hidden_size_ohe, activation='elu')(y)
    
    x = concatenate([x_day, x_hour, y], axis = -1)
    x = Dropout(0.25)(x)
    x = Dense(final_layer_size, activation='elu')(x)
    x = Dropout(0.25)(x)
    
    out = Dense(1)(x)
    model = Model(inputs=[consumption_input_day, consumption_input_hour, ohe_input], outputs=out)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    
    return model

def day_model_without_hour(hidden_size_lstm, hidden_size_ohe, final_layer_size, num_ohe, lag):
    consumption_input = Input(shape=(1, lag), name='input_lstm')
    x = LSTM(units=hidden_size_lstm, input_shape=(1, lag), 
             return_sequences=True)(consumption_input)
#     x = Dropout(0.25)(x)
    x = LSTM(units=hidden_size_lstm)(x)
#     x = Dropout(0.25)(x)
    
    ohe_input = Input(shape=(num_ohe,), name='input_ohe')
    y = Dropout(0.5)(ohe_input)
    y = Dense(hidden_size_ohe, activation='elu')(ohe_input)
    
    x = concatenate([x, y], axis = -1)
    x = Dropout(0.25)(x)
    x = Dense(final_layer_size, activation='elu')(x)
#     x = Dropout(0.25)(x)
#     x = Dense(100, activation='elu')(x)
    x = Dropout(0.25)(x)
    
    out = Dense(1)(x)
    model = Model(inputs=[consumption_input, ohe_input], outputs=out)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    
    return model

print('Loading daily models...')
hidden_size_lstm = 300
hidden_size_ohe = 300
final_layer_size = 100
num_ohe = len(one_hot_cols_daily)
lag = 14

nn_model_day_hour = day_model_with_hour(hidden_size_lstm, hidden_size_ohe, final_layer_size, num_ohe, lag)
filepath_use_hour=root_path + "day_nn_with_scaler_14_day_168_hours_25_ohe_300_300_100_batch_16_36_epoch_sub.hdf5"
nn_model_day_hour.load_weights(filepath_use_hour)

nn_model_day_no_hour = day_model_without_hour(hidden_size_lstm, hidden_size_ohe, final_layer_size, num_ohe, lag)
filepath_no_hour = root_path + "day_14_day_25_ohe_300_300_100_batch_16_30_iter_sub.hdf5"
nn_model_day_no_hour.load_weights(filepath_no_hour)

lgb_model_day_hour = lgb.Booster(model_file=root_path + "day_lgb_11k_iter_209_features_sub.txt")
lgb_model_day_no_hour = lgb.Booster(model_file=root_path + "day_lgb_41_features_4500_iter_sub.txt")

xgb_model_day_hour = joblib.load(root_path + "day_xgb_209_features_1200_iter_sub.joblib.dat")
xgb_model_day_no_hour = joblib.load(root_path + "day_xgb_41_features_no_hour_1400_iter_sub.joblib.dat")


def normalize_consumptions(df, mode='hourly'):
    if mode == 'hourly':
        df['consumption'] = (df['consumption'] - df['con_hour_min'] + 1e-5) / (df['con_hour_max'] - df['con_hour_min'] + 1e-5) * 2 - 1
    elif mode == 'daily':
        df['consumption'] = (df['consumption'] - df['con_hour_min'] * 24 + 1e-2) / (df['con_hour_max'] * 24 - df['con_hour_min'] * 24 + 1e-2) * 2 - 1
    elif mode == 'weekly':
        df['consumption'] = (df['consumption'] - df['con_hour_min'] * 24 * 7 + 1e-2) / (df['con_hour_max'] * 24 * 7 - df['con_hour_min'] * 24 * 7 + 1e-2) * 2 - 1
    else:
        print('Wrong mode...')
        return
    for col in tqdm(df.columns):
        if 'consumption_prev_hour_' in col:
            df[col] = (df[col] - df['con_hour_min'] + 1e-5) / (df['con_hour_max'] - df['con_hour_min'] + 1e-5) * 2 - 1
        elif 'consumption_prev_day_' in col:
            df[col] = (df[col] - df['con_hour_min'] * 24 + 1e-2) / (df['con_hour_max'] * 24 - df['con_hour_min'] * 24 + 1e-2) * 2 - 1
        elif 'consumption_prev_week_' in col:
            df[col] = (df[col] - df['con_hour_min'] * 24 * 7 + 1e-3) / (df['con_hour_max'] * 24 * 7 - df['con_hour_min'] * 24 * 7 + 1e-3) * 2 - 1

    return df

def find_prev_consumption(df_train, lag_num = 14, mode='day'):
    ser_ids = df_train.series_id.unique()
    list_consumptions = []
    for id_ in ser_ids:
        df_tmp = df_train[df_train.series_id==id_]
        if df_tmp.shape[0] >= lag_num:
            con_vs = df_tmp[-lag_num:].consumption.tolist()[::-1]
        else:
            con_vs =  df_tmp.consumption.tolist()[::-1] + [np.NaN] * (lag_num - df_tmp.shape[0])
        list_consumptions.append([id_] + con_vs)
    if mode=='day':
        cols = ['series_id'] + ['consumption_prev_day_' + str(i + 1) for i in range(lag_num)]
    elif mode=='hour':
        cols = ['series_id'] + ['consumption_prev_hour_' + str(i + 1) for i in range(lag_num)]
    df_res = pd.DataFrame(data=list_consumptions, columns=cols)
    return df_res

def lgb_predict(df, model, train_features):
#     df.fillna(0, inplace=True)
    preds = model.predict(df[train_features])
    preds = np.clip(preds, -1, 1).ravel()
    return preds

def xgb_predict(df, model, train_features):
    #     df.fillna(0, inplace=True)
    d_data = xgb.DMatrix(df[train_features])
    preds = model.predict(d_data)
    preds = np.clip(preds, -1, 1).ravel()
    return preds

def nn_predict(df, model, use_hour = True):
    df.fillna(0, inplace=True)
    if use_hour:
        pred = np.clip(model.predict([df[consumption_cols_day].values.reshape(-1, 1, 14),
                                      df[consumption_cols_hour[:168]].values.reshape(-1, 1, 168), 
                                      df[one_hot_cols_daily].values]).ravel(), -1, 1)
    else:
        pred = np.clip(model.predict([df[consumption_cols_day].values.reshape(-1, 1, 14), 
                                      df[one_hot_cols_daily].values]).ravel(), -1, 1)
    return pred


def generate_predict_day(test, ids, initial_df_day, initial_df_hour,
                         prediction_window='daily'):
    pred_idxes = []
    pred_values = []
    if prediction_window == 'daily':
        num_iteration = 7
    else:
        num_iteration = 14

    for i in tqdm(range(num_iteration)):
        if i == 0:
            df_1 = test[test.series_id.isin(ids)].drop_duplicates(subset=['series_id'])
            df_1.sort_values(by=['series_id'], inplace=True)
            hist_days = initial_df_day[initial_df_day.series_id.isin(ids)]
            hist_days.sort_values(by=['series_id'], inplace=True)
            hist_hours = initial_df_hour[initial_df_hour.series_id.isin(ids)]
            hist_hours.sort_values(by=['series_id'], inplace=True)
            indexes = df_1.index.tolist()

            df_1 = pd.merge(df_1, hist_days, how='left', on=['series_id'])
            df_1 = pd.merge(df_1, hist_hours, how='left', on=['series_id'])
            
        else:
            data_new_day = np.concatenate([hist_days.values[:, 0:1],  np.array(pred_1).reshape(len(pred_1), 1), 
                                       hist_days.values[:, 1:-1]],axis=1)
            hist_days = pd.DataFrame(data=data_new_day, columns=hist_days.columns.tolist())
            
            indexes = [idx + 1 for idx in indexes]
            df_1 = pd.merge(test.loc[indexes], hist_days, how='left', on=['series_id'])
                
        pred_idxes += indexes
        if i == 0:
            train_features = one_hot_cols_daily + consumption_cols_day + consumption_cols_hour[:168] + ['temperature', 'day']

            pred_lgb = lgb_predict(df_1, lgb_model_day_hour, train_features=train_features)
            pred_xgb = xgb_predict(df_1, xgb_model_day_hour, train_features=train_features)
            pred_nn = nn_predict(df_1, nn_model_day_hour, use_hour=True)
            pred_1 = pred_lgb * 0.3 + pred_xgb * 0.3 + pred_nn * 0.4
            pred_1 = pred_1.tolist()
        else:
            train_features = one_hot_cols_daily + consumption_cols_day + ['temperature', 'day']
            pred_lgb = lgb_predict(df_1, lgb_model_day_no_hour, train_features=train_features)
            pred_xgb = xgb_predict(df_1, xgb_model_day_no_hour, train_features=train_features)
            pred_nn = nn_predict(df_1, nn_model_day_no_hour, use_hour=False)
            pred_1 = pred_lgb * 0.5 + pred_xgb * 0.3 + pred_nn * 0.2
            pred_1 = pred_1.tolist()
        pred_values += pred_1
    return pred_idxes, pred_values

def generate_predicts_day(df_train_d, test, windows=['daily', 'weekly']):
    pred_idxes = []
    pred_values = []
    df_hist_hours = find_prev_consumption(df_train_hour, lag_num=24 * 7 * 2, mode='hour')
    df_hist_days = find_prev_consumption(df_train_d, lag_num=14, mode='day')
    for window_ in windows:
        ids_ = test[test.prediction_window==window_].series_id.unique()
        preds = generate_predict_day(test, ids_, df_hist_days, df_hist_hours, window_)
        pred_idxes += preds[0]
        pred_values += preds[1]

    return pred_idxes, pred_values

df_train_hour = normalize_consumptions(df_train_hour, mode='hourly')

df_test_day.sort_values(by=['series_id', 'timestamp'], inplace=True)
df_test_day = df_test_day.reset_index(drop=True)
df_train_day = normalize_consumptions(df_train_day, mode='daily')

pred_idxes_day_combine, pred_values_day_combine = generate_predicts_day(df_train_day, df_test_day)


df_test_day.at[pred_idxes_day_combine, 'consumption'] = pred_values_day_combine
df_test_day['consumption'] = (df_test_day.consumption + 1) / 2 * (df_test_day.con_hour_max * 24 - df_test_day.con_hour_min * 24) + df_test_day.con_hour_min * 24
df_test_sub = df_test_day[submission_format.columns.tolist()]

df_test_con = df_test_sub.groupby(['pred_id']).consumption.sum().reset_index()
df_sub_days = pd.merge(submission_format.drop(['consumption'], axis=1), df_test_con, how='left', on=['pred_id'])
df_sub_days = df_sub_days[submission_format.columns.tolist()]


df_sub_hour = pd.read_csv('pred_from_hourly_model.csv')

df_sub_final = pd.concat([df_sub_hour[df_sub_hour.prediction_window=='hourly'],
                          df_sub_days[df_sub_days.prediction_window!='hourly']], axis=0)
df_sub_final.sort_values(by=['pred_id'], inplace=True)
df_sub_final.set_index(['pred_id'], inplace=True)

df_sub_final.to_csv('final_sub.csv')

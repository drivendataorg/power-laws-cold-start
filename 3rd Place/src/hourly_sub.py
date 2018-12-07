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
from keras import backend as K

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *

from datetime import timedelta
from sklearn.externals import joblib

root_path = '../input/'

# %%time
print('Reading data...')
consumption_train = pd.read_csv(root_path + 'consumption_train.csv', index_col=0, parse_dates=['timestamp'])
meta = pd.read_csv(root_path + 'meta.csv')
cold_start_test = pd.read_csv(root_path + 'cold_start_test.csv', index_col=0, parse_dates=['timestamp'])
submission_format = pd.read_csv(root_path + 'submission_format.csv', parse_dates=['timestamp'])
# df = pd.concat([consumption_train, cold_start_test], axis=0).reset_index(drop=True)

df_train_hour = pd.read_csv(root_path + "cold_start_df_train_hourly_no_scaler_ohe.csv", parse_dates=['timestamp'])
df_test_hour = pd.read_csv(root_path + 'cold_start_df_test_hourly_no_scaler_ohe.csv', parse_dates=['timestamp'])

df_train_day = pd.read_csv(root_path + "cold_start_df_train_daily_no_scaler_ohe.csv", parse_dates=['timestamp'])
df_test_day = pd.read_csv(root_path + 'cold_start_df_test_daily_no_scaler_ohe.csv', parse_dates=['timestamp'])


consumption_cols_day = ['consumption_prev_day_' + str(i) for i in range(1, 7 * 2 + 1)]
consumption_cols_hour = ['consumption_prev_hour_' + str(i) for i in range(1, 24 * 7 * 2 + 1)]
ohe_months = ['month_' + str(i) for i in range(2, 13)]
# ohe_days = ['day_' + str(i) for i in range(2, 32)]
ohe_hours = ['hour_' + str(i) for i in range(1, 24)]
ohe_dow = ['day_of_week_' + str(i) for i in range(1, 7)]
ohe_surface = ['surface_' + str(i) for i in range(1, 7)]
one_hot_cols_hour = [ 'is_off_day_True', 'base_temperature_1']
one_hot_cols_hour += ohe_months + ohe_dow + ohe_hours + ohe_surface

train_features = consumption_cols_hour + one_hot_cols_hour + ['day']
print(len(consumption_cols_hour), len(train_features))


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

def lstm_model_hour(hidden_size_lstm, hidden_size_ohe, final_layer_size, num_ohe, lag):

    consumption_input_hour = Input(shape=(1, lag), name='input_lstm_hour')
    x_hour = LSTM(units=hidden_size_lstm, input_shape=(1, lag),
                  return_sequences=True)(consumption_input_hour)
#     x_hour = Dropout(0.25)(x_hour)
#     x = LSTM(units=hidden_size_lstm, return_sequences=True)(x)
    x_hour = LSTM(units=hidden_size_lstm)(x_hour)

    ohe_input = Input(shape=(num_ohe,), name='input_ohe')
    y = Dropout(0.5)(ohe_input)
    y = Dense(hidden_size_ohe, activation='elu')(ohe_input)
#     y = Dropout(0.25)(y)
#     y = Dense(hidden_size_ohe, activation='elu')(y)

    x = concatenate([x_hour, y], axis=-1)
    x = Dropout(0.25)(x)
    x = Dense(final_layer_size, activation='elu')(x)
    x = Dropout(0.25)(x)

    out = Dense(1)(x)
    model = Model(inputs=[consumption_input_hour, ohe_input], outputs=out)
    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model

print('Loading hourly models...')
num_ohe = len(one_hot_cols_hour)

hidden_size_lstm = 600
hidden_size_ohe = 600
final_layer_size = 300
lag = 168

nn_model_hour = lstm_model_hour(hidden_size_lstm, hidden_size_ohe, final_layer_size, num_ohe, lag)

filepath="../input/nn_hour_168_hour_48_ohe_600_600_300_batch_128_16_iter_sub.hdf5"
nn_model_hour.load_weights(filepath)

lgb_model_hour = lgb.Booster(model_file='../input/hour_lgb_385_featues_13k_iter_sub.txt')
xgb_model_hour = joblib.load('../input/hour_xgb_385_features_400_iter_sub.joblib.dat')

df_train_hour = normalize_consumptions(df_train_hour, mode='hourly')


def find_prev_consumption(df_train, lag_num=14, mode='day'):
    ser_ids = df_train.series_id.unique()
    list_consumptions = []
    for id_ in ser_ids:
        df_tmp = df_train[df_train.series_id == id_]
        if df_tmp.shape[0] >= lag_num:
            con_vs = df_tmp[-lag_num:].consumption.tolist()[::-1]
        else:
            con_vs = df_tmp.consumption.tolist()[::-1] + [np.NaN] * (lag_num - df_tmp.shape[0])
        list_consumptions.append([id_] + con_vs)
    if mode == 'day':
        cols = ['series_id'] + ['consumption_prev_day_' + str(i + 1) for i in range(lag_num)]
    elif mode == 'hour':
        cols = ['series_id'] + ['consumption_prev_hour_' + str(i + 1) for i in range(lag_num)]
    df_res = pd.DataFrame(data=list_consumptions, columns=cols)
    return df_res

def lgb_predict(df, model):
    #     df.fillna(0, inplace=True)
    preds = model.predict(df[train_features])
    preds = np.clip(preds, -1, 1).ravel().tolist()
    return preds

def xgb_predict(df, model):
    #     df.fillna(0, inplace=True)
    d_data = xgb.DMatrix(df[train_features])
    preds = model.predict(d_data)
    preds = np.clip(preds, -1, 1).ravel().tolist()
    return preds

def nn_predict(df, model):
    df.fillna(0, inplace=True)
    pred = np.clip(model.predict(
        [df[consumption_cols_hour[:168]].values.reshape(-1, 1, len(consumption_cols_hour[:168])),
         df[one_hot_cols_hour].values]).ravel(), -1, 1).tolist()
    return pred

def generate_predict_hour(models, test, ids, initial_df, mode, prediction_window='hourly'):
#     train_features = consumption_cols_hour + one_hot_cols_hour
    pred_idxes = []
    pred_values = []
    if prediction_window == 'hourly':
        num_iteration = 24
    elif prediction_window == 'daily':
        num_iteration = 24 * 7
    else:
        num_iteration = 24 * 14

    for i in tqdm(range(num_iteration)):
        if i == 0:
            df_1 = test[test.series_id.isin(ids)].drop_duplicates(subset=['series_id'])
            df_1.sort_values(by=['series_id'], inplace=True)
            hist_df = initial_df[initial_df.series_id.isin(ids)]
            hist_df.sort_values(by=['series_id'], inplace=True)
            indexes = df_1.index.tolist()

            df_1 = pd.merge(df_1, hist_df, how='left', on=['series_id'])
        else:
            data_new = np.concatenate([hist_df.values[:, 0:1],  np.array(pred_1).reshape(len(pred_1), 1),
                                       hist_df.values[:, 1:-1]], axis=1)
            hist_df = pd.DataFrame(data=data_new, columns=initial_df.columns.tolist())
            indexes = [idx + 1 for idx in indexes]
            df_1 = pd.merge(test.loc[indexes], hist_df, how='left', on=['series_id'])
        pred_idxes += indexes
        if mode == 'nn':
            if len(models) > 1:
                return
            pred_1 = nn_predict(df_1, models[0])
        elif mode == 'lgb':
            if len(models) > 1:
                return
            pred_1 = lgb_predict(df_1, models[0])
        elif mode == 'xgb':
            pred_1 = xgb_predict(df_1, models[0])
        pred_values += pred_1
    return pred_idxes, pred_values


def generate_predicts_hour(df_train_hour, models, test, mode='nn',
                           windows=['hourly', 'daily', 'weekly'], lag_hour=24*7*2):
    pred_idxes = []
    pred_values = []
    df_hist_hours = find_prev_consumption(df_train_hour, lag_num=lag_hour, mode='hour')
    for window_ in windows:
        ids_ = test[test.prediction_window == window_].series_id.unique()
        preds = generate_predict_hour(models, test, ids_, df_hist_hours, mode, window_)
        pred_idxes += preds[0]
        pred_values += preds[1]

    return pred_idxes, pred_values

print('Predict using lgb model...')
pred_idxes_lgb_hour, pred_values_lgb_hour = generate_predicts_hour(df_train_hour, [lgb_model_hour], df_test_hour, mode='lgb')
print('Predict using xgb model...')
pred_idxes_xgb_hour, pred_values_xgb_hour = generate_predicts_hour(df_train_hour, [xgb_model_hour], df_test_hour, mode='xgb')
print('Predict using nn model...')
pred_idxes_nn_hour, pred_values_nn_hour = generate_predicts_hour(df_train_hour, [nn_model_hour], df_test_hour, mode='nn')

pred_comb = [pred_values_xgb_hour[i] * 0.1 + pred_values_lgb_hour[i] * 0.5 + pred_values_nn_hour[i] * 0.4 for i in range(len(pred_idxes_lgb_hour))]
df_test_hour.at[pred_idxes_lgb_hour, 'consumption'] = pred_comb


df_test_hour['consumption'] = (df_test_hour['consumption'] + 1) / 2 * (df_test_hour.con_hour_max - df_test_hour.con_hour_min) + df_test_hour.con_hour_min


df_sub_hour = df_test_hour.groupby(['pred_id']).consumption.sum().reset_index()
df_sub_hour = pd.merge(submission_format.drop(['consumption'], axis=1), df_sub_hour, how='left', on=['pred_id'])
df_sub_hour = df_sub_hour[submission_format.columns.tolist()]
df_sub_hour.set_index(['pred_id'], inplace=True)

df_sub_hour.to_csv('pred_from_hourly_model.csv')


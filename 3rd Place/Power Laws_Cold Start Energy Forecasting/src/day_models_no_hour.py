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

root_path = '../input/'

DEBUG_MODE = False
NUM_LGB_ITER = 4500
NUM_XGB_ITER = 1400


print('Reading data...')
consumption_train = pd.read_csv('../input/consumption_train.csv', index_col=0, parse_dates=['timestamp'])
meta = pd.read_csv('../input/meta.csv')
cold_start_test = pd.read_csv('../input/cold_start_test.csv', index_col=0, parse_dates=['timestamp'])
submission_format = pd.read_csv('../input/submission_format.csv', parse_dates=['timestamp'])
# df = pd.concat([consumption_train, cold_start_test], axis=0).reset_index(drop=True)

# df_train_hour = pd.read_csv(root_path + "cold_start_df_train_hourly_no_scaler_ohe.csv", parse_dates=['timestamp'])
# df_test_hour = pd.read_csv(root_path + 'cold_start_df_test_hourly_no_scaler_ohe.csv', parse_dates=['timestamp'])

df_train_day = pd.read_csv(root_path + "cold_start_df_train_daily_no_scaler_ohe.csv", parse_dates=['timestamp'])
df_test_day = pd.read_csv(root_path + 'cold_start_df_test_daily_no_scaler_ohe.csv', parse_dates=['timestamp'])

def split_data(df, mode='hourly', model='lgb', id_start=758, id_end=1400):
    tr_val_idx = []
    sample_weights = []
    ids_ = df.series_id.unique().tolist()[id_start:id_end]
    df_tmp = df[df.series_id.isin(ids_)].copy()
    for i, id_ in enumerate(ids_):
        df_tmp2 = df_tmp[df_tmp.series_id == id_]
        nr = df_tmp2.shape[0]
        if id_ in df.series_id.unique().tolist()[:758]:
            nr_val = 0
        else:
            if mode == 'hourly':
                nr_val = min(24, np.power(2, nr // 24 - 1))
            elif mode == 'daily':
                nr_val = min(3, nr // 3)
            elif mode == 'weekly':
                nr_val = max(0, nr - 4)
        tr_val_idx += ['tr'] * (nr - nr_val) + ['val'] * nr_val
    df_tmp['train_val'] = tr_val_idx
    return df_tmp


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

consumption_cols_day = ['consumption_prev_day_' + str(i) for i in range(1, 7 * 2 + 1)]
one_hot_cols_daily = [
    'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 
    'month_10', 'month_11', 'month_12', 
    'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 
    'surface_1', 'surface_2', 'surface_3', 'surface_4', 'surface_5', 'surface_6', 
    'is_off_day_True', 'base_temperature_1'
]
train_features = one_hot_cols_daily + consumption_cols_day + ['temperature', 'day']
# train_features += ['day', 'tsp_int']
print('Number of lag days: ', len(consumption_cols_day), ' Number of used features: ', len(train_features))
# print(train_features)


# LGB day model

df_lgb_daily = split_data(df_train_day, mode='daily', id_start=0)
df_lgb_daily = normalize_consumptions(df_lgb_daily, mode='daily')
print(df_lgb_daily.train_val.value_counts())

def nmae(preds, train_data):
    ys = train_data.get_label()
    inv_ys = (ys + 1) / 2 * (val_maxs - val_mins) + val_mins
    inv_preds = (preds + 1) / 2 * (val_maxs - val_mins) + val_mins
    nmae = np.mean(abs(inv_preds - inv_ys) / val_means)
    return 'nmae', nmae, False

def nmae_sub(preds, train_data):
    ys = train_data.get_label()
    inv_ys = (ys + 1) / 2 * (sub_maxs - sub_mins) + sub_mins
    inv_preds = (preds + 1) / 2 * (sub_maxs - sub_mins) + sub_mins
    nmae = np.mean(abs(inv_preds - inv_ys) / sub_means)
    return 'nmae', nmae, False

def nmae_obj(y_hat, dtrain):
    y = dtrain.get_label()
    weights = dtrain.get_weight()
    grad = (y_hat - y) / abs(y - y_hat)# * weights
#     grad = 2 * (y_hat - y) * weights
    hess = np.zeros_like(grad)
#     hess = np.ones_like(grad) * 2 * weights
    return grad, hess

lgb_params = {
    'task' : 'train',
    'boosting_type' : 'gbdt',
    'objective' : 'regression_l1',
#     'metric' : {'mae'},
    'max_depth': 12,
    'max_bin': 255,
#     'num_leaves' : 128,
    'learning_rate' : 0.01,
    'feature_fraction' : 0.75,
    'bagging_fraction': 0.75,
    'bagging_seed': 2018,
    'bagging_freq': 100,
    'min_data_in_leaf': 30,
    'min_sum_hessian_in_leaf': 0.3,
    'silent': True
}

if DEBUG_MODE:
    x_tr = df_lgb_daily[df_lgb_daily.train_val=='tr'][train_features].values
    x_val = df_lgb_daily[df_lgb_daily.train_val=='val'][train_features].values

    y_tr = df_lgb_daily[df_lgb_daily.train_val=='tr'].consumption.values
    y_val = df_lgb_daily[df_lgb_daily.train_val=='val'].consumption.values

    val_maxs = df_lgb_daily[df_lgb_daily.train_val=='val'].con_hour_max.values
    val_mins = df_lgb_daily[df_lgb_daily.train_val=='val'].con_hour_min.values
    val_means = df_lgb_daily[df_lgb_daily.train_val=='val'].con_hour_mean.values

    # train_features += ['pred_nn']
    lgb_train = lgb.Dataset(x_tr, y_tr, feature_name=train_features)
    lgb_val = lgb.Dataset(x_val, y_val, feature_name=train_features, reference=lgb_train)
        
    print(x_tr.shape, x_val.shape, y_tr.shape, y_val.shape)
    print(df_lgb_daily.consumption.describe())
    print(df_lgb_daily.consumption_prev_day_1.describe())

    print(len(train_features), x_tr.shape, x_val.shape)
    lgb_model_day = lgb.train(lgb_params, lgb_train, num_boost_round=1000000, valid_sets=[lgb_val],
                             early_stopping_rounds=500, verbose_eval=500, feval=nmae)


x_tr = df_lgb_daily[train_features].values
y_tr = df_lgb_daily.consumption.values

sub_maxs = df_lgb_daily.con_hour_max.values
sub_mins = df_lgb_daily.con_hour_min.values
sub_means = df_lgb_daily.con_hour_mean.values

lgb_train = lgb.Dataset(x_tr, y_tr, feature_name=train_features)

print(x_tr.shape, y_tr.shape, ' Number of training iteration: ', NUM_LGB_ITER)

lgb_model_day_sub = lgb.train(lgb_params, lgb_train, num_boost_round=NUM_LGB_ITER, 
                          valid_sets=[lgb_train], verbose_eval=500, feval=nmae_sub)

lgb_model_day_sub.save_model('../input/day_lgb_41_features_4500_iter_sub.txt')


# XGB model

xgb_params = {
    'eta': 0.01,
    'max_depth': 12,
#     'subsample': 0.75,
    'colsample_bytree': 0.75,
    'objective': 'reg:linear',
    'min_child_weight': 32,                            
    'eval_metric': 'mae',
    'seed': 123,
#     'min_child_weight': 0.3,
}

if DEBUG_MODE:
    x_tr = df_lgb_daily[df_lgb_daily.train_val=='tr'][train_features].values
    y_tr = df_lgb_daily[df_lgb_daily.train_val=='tr'].consumption.values

    x_val = df_lgb_daily[df_lgb_daily.train_val=='val'][train_features].values
    y_val = df_lgb_daily[df_lgb_daily.train_val=='val'].consumption.values

    print(len(train_features), x_tr.shape, x_val.shape)

    dtrain = xgb.DMatrix(x_tr, y_tr, feature_names=train_features)
    dval = xgb.DMatrix(x_val, y_val, feature_names=train_features)

    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=50000, evals=[(dtrain, 'train'), (dval, 'val')],
                               early_stopping_rounds=100, verbose_eval=100)


x_tr = df_lgb_daily[train_features].values
y_tr = df_lgb_daily.consumption.values

dtrain = xgb.DMatrix(x_tr, y_tr, feature_names=train_features)
print(x_tr.shape, y_tr.shape, ' Number of used iterations: ', NUM_XGB_ITER)

xgb_model_sub = xgb.train(xgb_params, dtrain, num_boost_round=NUM_XGB_ITER, evals=[(dtrain, 'train')],
                           verbose_eval=100)
joblib.dump(xgb_model_sub, "../input/day_xgb_41_features_no_hour_1400_iter_sub.joblib.dat")


# NN model

def step_decay(epoch):
    return 0.001 * np.power(0.5, epoch // 5)

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

df_nn_day = split_data(df_train_day, mode='daily', id_start=0)
df_nn_day = normalize_consumptions(df_nn_day, mode='daily')
df_nn_day.fillna(0, inplace=True)
print(df_nn_day.train_val.value_counts())
print(df_nn_day.consumption.describe())
print(df_nn_day.consumption_prev_day_1.describe())

hidden_size_lstm = 300
hidden_size_ohe = 300
final_layer_size = 100
num_ohe = len(one_hot_cols_daily)
batch_size = 16
lag = 14

if DEBUG_MODE:
    # df_lstm = df_split.copy()
    # used_hours = 168
    X_lstm_train = df_nn_day[df_nn_day.train_val=='tr'][
        consumption_cols_day].values.reshape(-1, 1, len(consumption_cols_day))
    y_lstm_train = df_nn_day[df_nn_day.train_val=='tr']['consumption']
    X_lstm_val = df_nn_day[df_nn_day.train_val=='val'][
        consumption_cols_day].values.reshape(-1, 1, len(consumption_cols_day))
    y_lstm_val = df_nn_day[df_nn_day.train_val=='val']['consumption']

    X_ohe_train = df_nn_day[df_nn_day.train_val=='tr'][one_hot_cols_daily].values
    X_ohe_val = df_nn_day[df_nn_day.train_val=='val'][one_hot_cols_daily].values
    print(X_lstm_train.shape, y_lstm_train.shape, X_lstm_val.shape, y_lstm_val.shape, X_ohe_train.shape)

    nn_model_day_without = day_model_without_hour(hidden_size_lstm, hidden_size_ohe, final_layer_size, num_ohe, lag)

    filepath="../input/day_14_day_25_ohe_300_300_100_batch_16_debug.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    lrate=LearningRateScheduler(step_decay)
    callbacks_list = [EarlyStopping(monitor='val_loss', patience=10), checkpoint, lrate]

    h1 = nn_model_day_without.fit([X_lstm_train, X_ohe_train], y_lstm_train, epochs=100, 
                        batch_size=batch_size, verbose=1, shuffle=True, 
                        validation_data=([X_lstm_val, X_ohe_val], y_lstm_val), 
                        callbacks=callbacks_list
    )

X_lstm_train = df_nn_day[consumption_cols_day].values.reshape(-1, 1, len(consumption_cols_day))
y_lstm_train = df_nn_day['consumption']

X_ohe_train = df_nn_day[one_hot_cols_daily].values
print(X_lstm_train.shape, y_lstm_train.shape, X_ohe_train.shape)

nn_model_day_sub = day_model_without_hour(hidden_size_lstm, hidden_size_ohe, final_layer_size, num_ohe, lag)

filepath="../input/day_14_day_25_ohe_300_300_100_batch_16_30_iter_sub.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)
lrate=LearningRateScheduler(step_decay)
callbacks_list = [checkpoint, lrate]

h1 = nn_model_day_sub.fit([X_lstm_train, X_ohe_train], y_lstm_train, epochs=30, 
                    batch_size=batch_size, verbose=1, shuffle=True, 
                    callbacks=callbacks_list
)

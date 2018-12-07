# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os

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

from sklearn.externals import joblib

# from utils import *

DEBUG_MODE = False
NUM_LGB_ITER = 13000
NUM_XGB_ITER = 400
# Read data

root_path = '../input/'

consumption_train = pd.read_csv(root_path + 'consumption_train.csv', index_col=0, parse_dates=['timestamp'])
meta = pd.read_csv(root_path + 'meta.csv')
cold_start_test = pd.read_csv(root_path + 'cold_start_test.csv', index_col=0, parse_dates=['timestamp'])
submission_format = pd.read_csv(root_path + 'submission_format.csv', parse_dates=['timestamp'])
# df = pd.concat([consumption_train, cold_start_test], axis=0).reset_index(drop=True)

# use_scaler = False
df_train_hour = pd.read_csv(root_path + "cold_start_df_train_hourly_no_scaler_ohe.csv", parse_dates=['timestamp'])
df_test_hour = pd.read_csv(root_path + 'cold_start_df_test_hourly_no_scaler_ohe.csv', parse_dates=['timestamp'])

df_train_day = pd.read_csv(root_path + "cold_start_df_train_daily_no_scaler_ohe.csv", parse_dates=['timestamp'])
df_test_day = pd.read_csv(root_path + 'cold_start_df_test_daily_no_scaler_ohe.csv', parse_dates=['timestamp'])

# Used features

# consumption_cols_day = ['consumption_prev_day_' + str(i) for i in range(1, 7 * 2 + 1)]
consumption_cols_hour = ['consumption_prev_hour_' + str(i) for i in range(1, 24 * 7 * 2 + 1)]
ohe_months = ['month_' + str(i) for i in range(2, 13)]
ohe_hours = ['hour_' + str(i) for i in range(1, 24)]
ohe_dow = ['day_of_week_' + str(i) for i in range(1, 7)]
ohe_surface = ['surface_' + str(i) for i in range(1, 7)]
one_hot_cols_hour = [ 'is_off_day_True', 'base_temperature_1']
one_hot_cols_hour += ohe_months + ohe_dow + ohe_hours + ohe_surface

train_features = consumption_cols_hour + one_hot_cols_hour + ['day']
print('Number of used hourly lag features: ', len(consumption_cols_hour),' Number of all used features: ',  len(train_features))


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

# LGB hour model

df_lgb_hourly = split_data(df_train_hour, mode='hourly', model='lgb', id_start=0)
print(df_lgb_hourly.train_val.value_counts())
df_lgb_hourly = normalize_consumptions(df_lgb_hourly, mode='hourly')

def nmae_sub(preds, train_data):
    ys = train_data.get_label()
    inv_ys = (ys + 1) / 2 * (sub_maxs - sub_mins) + sub_mins
    inv_preds = (preds + 1) / 2 * (sub_maxs - sub_mins) + sub_mins
    nmae = np.mean(abs(inv_preds - inv_ys) / sub_means)
    return 'nmae', nmae, False

lgb_params = {
    'task' : 'train',
    'boosting_type' : 'gbdt',
    'objective' : 'regression_l1',
    'metric' : {'mae'},
    'max_depth': 12,
    'max_bin': 255,
    'num_leaves' : 64,
    'learning_rate' : 0.01,
    'feature_fraction' : 0.75,
    'bagging_fraction': 0.75,
    'bagging_seed': 123,
    'bagging_freq': 100,
    'min_data_in_leaf': 30,
    'min_sum_hessian_in_leaf': 0.3,
}

def nmae(preds, train_data):
    ys = train_data.get_label()
    inv_ys = (ys + 1) / 2 * (val_maxs - val_mins) + val_mins
    inv_preds = (preds + 1) / 2 * (val_maxs - val_mins) + val_mins
    nmae = np.mean(abs(inv_preds - inv_ys) / val_means)
    return 'nmae', nmae, False

if DEBUG_MODE:
    x_tr = df_lgb_hourly[df_lgb_hourly.train_val=='tr'][train_features].values
    x_val = df_lgb_hourly[df_lgb_hourly.train_val=='val'][train_features].values

    y_tr = df_lgb_hourly[df_lgb_hourly.train_val=='tr'].consumption.values
    y_val = df_lgb_hourly[df_lgb_hourly.train_val=='val'].consumption.values

    val_maxs = df_lgb_hourly[df_lgb_hourly.train_val=='val'].con_hour_max.values
    val_mins = df_lgb_hourly[df_lgb_hourly.train_val=='val'].con_hour_min.values
    val_means = df_lgb_hourly[df_lgb_hourly.train_val=='val'].con_hour_mean.values



    lgb_train = lgb.Dataset(x_tr, y_tr, feature_name=train_features)
    lgb_val = lgb.Dataset(x_val, y_val, feature_name=train_features, reference=lgb_train)
    
    print(x_tr.shape, x_val.shape, y_tr.shape, y_val.shape)
    print(df_lgb_hourly.consumption.describe())


    lgb_model_hour = lgb.train(params, lgb_train, num_boost_round=100000, valid_sets=[lgb_val],
                             early_stopping_rounds=500, verbose_eval=500, feval=nmae
                         )

    lgb_model_hour.save_model(root_path + 'hour_lgb_385_featues_debug.txt')


print('Lightgbm hourly model...')
x_tr = df_lgb_hourly[train_features].values
y_tr = df_lgb_hourly.consumption.values

sub_maxs = df_lgb_hourly.con_hour_max.values
sub_mins = df_lgb_hourly.con_hour_min.values
sub_means = df_lgb_hourly.con_hour_mean.values

lgb_train = lgb.Dataset(x_tr, y_tr, feature_name=train_features)
print(len(train_features), x_tr.shape, y_tr.shape, ' Number of training iteration: ', NUM_LGB_ITER)
lgb_model_hour_sub = lgb.train(lgb_params, lgb_train, num_boost_round=NUM_LGB_ITER, 
                           valid_sets=[lgb_train], verbose_eval=100, feval=nmae_sub)

lgb_model_hour_sub.save_model(root_path + 'hour_lgb_385_featues_13k_iter_sub.txt')

xgb_params = {
    'eta': 0.03,
    'max_depth': 12,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'objective': 'reg:linear',
    'min_child_weight': 32,                            
    'eval_metric': 'mae',
    'seed': 123,
#     'min_child_weight': 0.3,
}

if DEBUG_MODE:
    x_tr = df_lgb_hourly[df_lgb_hourly.train_val=='tr'][train_features].values
    x_val = df_lgb_hourly[df_lgb_hourly.train_val=='val'][train_features].values

    y_tr = df_lgb_hourly[df_lgb_hourly.train_val=='tr'].consumption.values
    y_val = df_lgb_hourly[df_lgb_hourly.train_val=='val'].consumption.values
    
    dtrain = xgb.DMatrix(x_tr, y_tr, feature_names=train_features)
    dval = xgb.DMatrix(x_val, y_val, feature_names=train_features)
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=50000, evals=[(dtrain, 'train'), (dval, 'val')],
                               early_stopping_rounds=100, verbose_eval=50)

    joblib.dump(xgb_model, root_path + "hour_xgb_385_features_debug.joblib.dat")


print('XGBoost hourly model...')
x_tr = df_lgb_hourly[train_features].values
y_tr = df_lgb_hourly.consumption.values
print(len(train_features), x_tr.shape, y_tr.shape, ' Number of training iteration: ', NUM_XGB_ITER)
dtrain = xgb.DMatrix(x_tr, y_tr, feature_names=train_features)


xgb_model_sub = xgb.train(xgb_params, dtrain, num_boost_round=NUM_XGB_ITER, evals=[(dtrain, 'train')],
                           verbose_eval=10)

joblib.dump(xgb_model_sub, root_path + "hour_xgb_385_features_400_iter_sub.joblib.dat")


# NN model

def step_decay(epoch):
    return 0.001 * np.power(0.5, epoch // 3)

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

df_nn_hour = split_data(df_train_hour, mode='hourly', id_start=0)
df_nn_hour = normalize_consumptions(df_nn_hour, mode='hourly')
df_nn_hour.fillna(0, inplace=True)
print(df_nn_hour.train_val.value_counts())
print(df_nn_hour.consumption.describe())

used_hours = 168
hidden_size_lstm = 600
hidden_size_ohe = 600
final_layer_size = 300
batch_size = 128
num_ohe = len(one_hot_cols_hour)
lag = used_hours

if DEBUG_MODE:
    X_lstm_train = df_nn_hour[df_nn_hour.train_val=='tr'][
        consumption_cols_hour[:used_hours]].values.reshape(-1, 1, len(consumption_cols_hour[:used_hours]))
    y_lstm_train = df_nn_hour[df_nn_hour.train_val=='tr']['consumption']
    X_lstm_val = df_nn_hour[df_nn_hour.train_val=='val'][
        consumption_cols_hour[:used_hours]].values.reshape(-1, 1, len(consumption_cols_hour[:used_hours]))
    y_lstm_val = df_nn_hour[df_nn_hour.train_val=='val']['consumption']


    X_ohe_train = df_nn_hour[df_nn_hour.train_val=='tr'][one_hot_cols_hour].values
    X_ohe_val = df_nn_hour[df_nn_hour.train_val=='val'][one_hot_cols_hour].values
    print(X_lstm_train.shape, y_lstm_train.shape, X_lstm_val.shape, y_lstm_val.shape, X_ohe_train.shape)

    nn_model_hour_one = lstm_model_hour(hidden_size_lstm, hidden_size_ohe, final_layer_size, num_ohe, lag)
    print(nn_model_hour.summary())

    filepath = root_path + "nn_hour_168_hour_48_ohe_600_600_300_batch_128_all.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [EarlyStopping(monitor='val_loss', patience=5), checkpoint, lrate]

    h1 = nn_model_hour_one.fit([X_lstm_train, X_ohe_train], y_lstm_train, epochs=100, 
                        batch_size=batch_size, verbose=1, shuffle=True, 
                        validation_data=([X_lstm_val, X_ohe_val], y_lstm_val), 
                        callbacks=callbacks_list
    )


X_lstm_train = df_nn_hour[
    consumption_cols_hour[:used_hours]].values.reshape(-1, 1, len(consumption_cols_hour[:used_hours]))
y_lstm_train = df_nn_hour['consumption']

X_ohe_train = df_nn_hour[one_hot_cols_hour].values
print(X_lstm_train.shape, y_lstm_train.shape, X_ohe_train.shape)

nn_model_hour_one = lstm_model_hour(hidden_size_lstm, hidden_size_ohe, final_layer_size, num_ohe, lag)

filepath = root_path + "nn_hour_168_hour_48_ohe_600_600_300_batch_128_16_iter_sub.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1)
lrate = LearningRateScheduler(step_decay)
callbacks_list = [checkpoint, lrate]

h1 = nn_model_hour_one.fit([X_lstm_train, X_ohe_train], y_lstm_train, epochs=16, 
                    batch_size=batch_size, verbose=1, shuffle=True,  
                    callbacks=callbacks_list
)

print('Finish...')
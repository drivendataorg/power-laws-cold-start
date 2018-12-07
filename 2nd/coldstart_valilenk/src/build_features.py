import re
import os
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.cluster import KMeans


def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, suffixes=("", suffix))

def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

def add_time_features(df):
    add_datepart(df, "timestamp", drop=False)
    df.loc[:, 'hour'] = df.timestamp.map(lambda x: x.hour)
    return df

def add_data_len(df):
    for _id in df.series_id.unique(): df.loc[df.series_id==_id, 'len'] = df.loc[df.series_id==_id, :].shape[0]
    return df

def generate_test(df, sub, mode='hourly'):
    ids = sub.loc[sub.prediction_window==mode, 'series_id'].unique()
    df = df.loc[df.series_id.isin(ids), :].copy()
    for i, _id in enumerate(ids):
        add_df = df.loc[df.series_id==_id, :].iloc[-24:]
        add_df.timestamp = add_df.timestamp.map(lambda x: x + timedelta(days=1))
        if mode=='daily':    r = 6
        elif mode=='weekly': r = 13
        if mode!='hourly':
            for x in range(r):
                add_df_temp = add_df.iloc[-24:].copy()
                add_df_temp.timestamp = add_df_temp.timestamp.map(lambda x: x + timedelta(days=1))
                add_df = add_df.append(add_df_temp, sort=False)
        add_df.consumption = -1
        add_df.temperature = np.nan
        df = df.append(add_df, sort=False)
        
    df = df.loc[df.consumption==-1, :].reset_index(drop=True)
    return df

def fill_temp(df):
    df.loc[:, 'temperature_nan'] = 0
    df.loc[df.temperature.isnull(), 'temperature_nan'] = 1
    filler = df.loc[:, ['timestamp', 'temperature']].groupby('timestamp').agg('median').reset_index()
    filler.columns = ['timestamp', 't_filler']
    df = df.merge(filler, how='left', on='timestamp')
    df.loc[df.temperature.isnull(), 'temperature'] = df.loc[df.temperature.isnull(), 't_filler']
    for hour in range(24):
        for day in range(0, 367):
            median_temp = df.loc[(df.hour==hour)&(df.timestampDayofyear==day), 'temperature'].median()
            df.loc[(df.hour==hour)&(df.timestampDayofyear==day)&(df.temperature.isnull()), 'temperature'] = median_temp
    df.drop('t_filler', axis=1, inplace=True)
    return df

def fill_temp_in_test(df, X):
    _len = df.shape[0]
    temp = X.append(df, sort=False)
    df = fill_temp(temp)
    df = df.iloc[-_len:,:]
    return df

def fill_dw_temp_and_rolling(df):
    day_temp = df[['series_id', 'timestampDayofyear', 'temperature']].groupby(['series_id', 'timestampDayofyear']).agg({'temperature': ['mean']})
    day_temp.reset_index(inplace=True)
    day_temp.columns = ['series_id', 'timestampDayofyear', 'temperature_d_mean']
    df = join_df(df, day_temp, ['series_id', 'timestampDayofyear'])
    df.temperature_d.fillna(df.temperature_d_mean, inplace=True)
    df.drop('temperature_d_mean', axis=1, inplace=True)
    df.temperature_w.fillna(df.temperature_d, inplace=True)
    
    for _id in df.series_id.unique():
        mask = df.series_id==_id
        df.loc[mask, 'temp_rolling_3'] = df.loc[mask, 'temperature'].rolling(min_periods=0,window=3).mean()
        df.loc[mask, 'temp_rolling_6'] = df.loc[mask, 'temperature'].rolling(min_periods=0,window=6).mean()
        df.loc[mask, 'temp_rolling_12'] = df.loc[mask, 'temperature'].rolling(min_periods=0,window=12).mean()
        df.loc[mask, 'temp_rolling_24'] = df.loc[mask, 'temperature'].rolling(min_periods=0,window=24).mean()
    return df

def create_working_day(df):
    df.loc[df.timestampDayofweek==0, 'working_day'] = 1 - df.iloc[:, 6]
    df.loc[df.timestampDayofweek==1, 'working_day'] = 1 - df.iloc[:, 7]
    df.loc[df.timestampDayofweek==2, 'working_day'] = 1 - df.iloc[:, 8]
    df.loc[df.timestampDayofweek==3, 'working_day'] = 1 - df.iloc[:, 9]
    df.loc[df.timestampDayofweek==4, 'working_day'] = 1 - df.iloc[:, 10]
    df.loc[df.timestampDayofweek==5, 'working_day'] = 1 - df.iloc[:, 11]
    df.loc[df.timestampDayofweek==6, 'working_day'] = 1 - df.iloc[:, 12]
    df.working_day = df.working_day.astype(np.int32)
    return df

def get_presence_of_daytypes(df, name='X', X=None):
    ids = df.series_id.unique()
    if name!='X':
        X_temp = X.append(df).reset_index(drop=True)
    else:
        X_temp = df.copy()
    X_temp = X_temp.loc[:, ['series_id', 'timestamp', 'working_day', 'timestampElapsed', 'consumption']]

    def create_lagged_features(id_num, X_temp=X_temp, ids=ids, name=name):
        _id = ids[id_num]
        X_id = X_temp.loc[X_temp.series_id==_id, :].copy()
        X_id.loc[:, 'has_day_off'] = 0
        X_id.loc[:, 'has_working_day'] = 0
        for i in X_id.index:
            time_cap = X_id.loc[i, 'timestampElapsed'] // 86400 * 86400
            day_types = X_id.loc[(X_id.timestampElapsed<time_cap)&(X_id.timestampElapsed>=time_cap-1209600), 'working_day'].unique()
            if 1 in day_types: X_id.loc[i, 'has_working_day'] = 1
            if 0 in day_types: X_id.loc[i, 'has_day_off'] = 1
        if name!='X': X_id = X_id.loc[X_id.consumption==-1, :]
        X_id.drop(['working_day', 'timestampElapsed', 'consumption'], axis=1, inplace=True)
        return X_id

    p = mp.Pool(mp.cpu_count())
    results = p.map(lambda x: create_lagged_features(x), range(len(ids)))
    shifted_df = pd.concat(results)
    df = df.merge(shifted_df, on=['series_id', 'timestamp'], how='left')
    return df


print('Loading data')
data_path = Path(os.path.dirname(os.getcwd())) / 'data'
train = pd.read_csv(data_path/'raw/consumption_train.csv', index_col=0, parse_dates=['timestamp'])
test = pd.read_csv(data_path/'raw/cold_start_test.csv', index_col=0, parse_dates=['timestamp'])
submit = pd.read_csv(data_path/'raw/submission_format.csv', index_col='pred_id', parse_dates=['timestamp'])
meta = pd.read_csv(data_path/'raw/meta.csv')

print('Generating 3 separate test dataframes')
h_test = generate_test(test, submit, mode='hourly')
d_test = generate_test(test, submit, mode='daily')
w_test = generate_test(test, submit, mode='weekly')

len_train = len(train)
X = train.append(test)

print('Joining meta data')
X = join_df(X, meta, "series_id")
h_test = join_df(h_test, meta, "series_id")
d_test = join_df(d_test, meta, "series_id")
w_test = join_df(w_test, meta, "series_id")

print('Adding time features')
X = add_time_features(X)
h_test = add_time_features(h_test)
d_test = add_time_features(d_test)
w_test = add_time_features(w_test)

print('Filling hourly temperature from submit file')
h_test.drop('temperature', axis=1, inplace=True)
h_submit = submit.loc[submit.prediction_window=='hourly', ['series_id', 'timestamp', 'temperature']]
h_test = join_df(h_test, h_submit, ['series_id', 'timestamp'])
h_test.loc[:, 'temperature_d'] = np.nan
h_test.loc[:, 'temperature_w'] = np.nan

print('Filling daily mean temperature from submit file')
d_submit = submit.loc[submit.prediction_window=='daily', ['series_id', 'timestamp', 'temperature']]
d_submit = add_time_features(d_submit)
d_submit = d_submit.loc[:, ['series_id', 'timestampDayofyear', 'temperature']]
d_submit.columns = ['series_id', 'timestampDayofyear', 'temperature_d']
d_test = join_df(d_test, d_submit, ['series_id', 'timestampDayofyear'])
d_test.loc[:, 'temperature_w'] = np.nan

print('Filling weekly mean temperature from submit file')
w_submit = submit.loc[submit.prediction_window=='weekly', ['series_id', 'temperature']]
w_submit.loc[:, 'temperature_w2'] = w_submit.temperature.shift(-1)
w_submit.columns = ['series_id', 'temperature_w', 'temperature_w2']
w_submit.drop_duplicates(subset='series_id', keep='first', inplace=True)
w_test = join_df(w_test, w_submit, 'series_id')
ids = w_test.series_id.unique()
for _id in w_test.series_id.unique():
    mask = w_test.series_id==_id
    index = w_test.loc[mask, :].index[0]+168
    w_test.loc[mask&(w_test.index>=index), 'temperature_w'] = w_test.temperature_w2
w_test.drop('temperature_w2', axis=1, inplace=True)
w_test.loc[:, 'temperature_d'] = np.nan
X.loc[:, 'temperature_d'] = np.nan
X.loc[:, 'temperature_w'] = np.nan

print('Filling temperature with medians')
h_test = fill_temp_in_test(h_test, X)
d_test = fill_temp_in_test(d_test, X)
w_test = fill_temp_in_test(w_test, X)
X = fill_temp(X)

print('Filling temperatures for days and weeks')
X = fill_dw_temp_and_rolling(X)
h_test = fill_dw_temp_and_rolling(h_test)
d_test = fill_dw_temp_and_rolling(d_test)
w_test = fill_dw_temp_and_rolling(w_test)

print('Adding history depth feature')
X = add_data_len(X)
h_test = add_data_len(h_test)
d_test = add_data_len(d_test)
w_test = add_data_len(w_test)

print('Adding feature of the current day number')
min_days = X.groupby('series_id').timestamp.min().apply(lambda t: (t - datetime(1970,1,1)).days).reset_index()
min_days.columns = ['series_id', 'first_day']

def get_current_day(df, min_days=min_days):
    df = df.merge(min_days, on='series_id', how='left')
    df.loc[:, 'current_day'] = df.timestamp.apply(lambda t: (t - datetime(1970,1,1)).days) - df.first_day
    df.drop('first_day', axis=1, inplace=True)
    return df

X = get_current_day(X)
h_test = get_current_day(h_test)
d_test = get_current_day(d_test)
w_test = get_current_day(w_test)

print('Adding feature of day type (working or not)')
X = create_working_day(X)
h_test = create_working_day(h_test)
d_test = create_working_day(d_test)
w_test = create_working_day(w_test)

def get_yesterday_and_tomorrow(df, kmeans):
    df.loc[df.timestampDayofweek==0, 'worked_yesterday'] = 1 - df.iloc[:, 12]
    df.loc[df.timestampDayofweek==1, 'worked_yesterday'] = 1 - df.iloc[:, 6]
    df.loc[df.timestampDayofweek==2, 'worked_yesterday'] = 1 - df.iloc[:, 7]
    df.loc[df.timestampDayofweek==3, 'worked_yesterday'] = 1 - df.iloc[:, 8]
    df.loc[df.timestampDayofweek==4, 'worked_yesterday'] = 1 - df.iloc[:, 9]
    df.loc[df.timestampDayofweek==5, 'worked_yesterday'] = 1 - df.iloc[:, 10]
    df.loc[df.timestampDayofweek==6, 'worked_yesterday'] = 1 - df.iloc[:, 11]
    df.loc[df.timestampDayofweek==0, 'works_tomorrow'] = 1 - df.iloc[:, 7]
    df.loc[df.timestampDayofweek==1, 'works_tomorrow'] = 1 - df.iloc[:, 8]
    df.loc[df.timestampDayofweek==2, 'works_tomorrow'] = 1 - df.iloc[:, 9]
    df.loc[df.timestampDayofweek==3, 'works_tomorrow'] = 1 - df.iloc[:, 10]
    df.loc[df.timestampDayofweek==4, 'works_tomorrow'] = 1 - df.iloc[:, 11]
    df.loc[df.timestampDayofweek==5, 'works_tomorrow'] = 1 - df.iloc[:, 12]
    df.loc[df.timestampDayofweek==6, 'works_tomorrow'] = 1 - df.iloc[:, 6]
    df.loc[:, 'work_schedule'] = kmeans.predict(df[day_off_cols])
    df.worked_yesterday = df.worked_yesterday.astype(np.int32)
    df.works_tomorrow = df.works_tomorrow.astype(np.int32)
    return df

print('Adding features of schedule and yesterdays/tomorrows day type')
day_off_cols = [c for c in X.columns if c.endswith('_is_day_off')]
kmeans = KMeans(n_clusters=7, random_state=0).fit(X[day_off_cols])
X = get_yesterday_and_tomorrow(X, kmeans)
h_test = get_yesterday_and_tomorrow(h_test, kmeans)
d_test = get_yesterday_and_tomorrow(d_test, kmeans)
w_test = get_yesterday_and_tomorrow(w_test, kmeans)

print('Adding circular time features')
for df in (X, h_test, d_test, w_test):  
    df['circ_wday_sin'] = np.sin(np.pi * df['timestampDayofweek'])
    df['circ_wday_cos'] = np.cos(np.pi * df['timestampDayofweek'])
    df['circ_mday_sin'] = np.sin(np.pi * df['timestampDay'])
    df['circ_mday_cos'] = np.cos(np.pi * df['timestampDay'])
    df['circ_yday_sin'] = np.sin(np.pi * df['timestampDayofyear'])
    df['circ_yday_cos'] = np.cos(np.pi * df['timestampDayofyear'])
    df['circ_week_sin'] = np.sin(np.pi * df['timestampWeek'])
    df['circ_week_cos'] = np.cos(np.pi * df['timestampWeek'])
    df['circ_month_sin'] = np.sin(np.pi * df['timestampMonth'])
    df['circ_month_cos'] = np.cos(np.pi * df['timestampMonth'])
    df['circ_time_sin'] = np.sin(np.pi * df['hour'])
    df['circ_time_cos'] = np.cos(np.pi * df['hour'])
    
print(f'Making scaling of consumption per each series_id, current max: {X.consumption.max()}')
for _id in X.series_id.unique():
    id_min = X.loc[X.series_id==_id, 'consumption'].min()
    X.loc[X.series_id==_id, 'id_min'] = id_min
    h_test.loc[h_test.series_id==_id, 'id_min'] = id_min
    d_test.loc[d_test.series_id==_id, 'id_min'] = id_min
    w_test.loc[w_test.series_id==_id, 'id_min'] = id_min
    
    id_max = X.loc[X.series_id==_id, 'consumption'].max()
    X.loc[X.series_id==_id, 'id_max'] = id_max
    h_test.loc[h_test.series_id==_id, 'id_max'] = id_max
    d_test.loc[d_test.series_id==_id, 'id_max'] = id_max
    w_test.loc[w_test.series_id==_id, 'id_max'] = id_max
    
    X.loc[X.series_id==_id, 'consumption'] = X.loc[X.series_id==_id, 'consumption'].map(lambda x: (x-id_min)/(id_max-id_min+1))
print(f'Min/max consumption values after scaling: {X.consumption.min()}, {X.consumption.max()}')

print('Checking, do we have each day type in history for each series_id')
h_test = get_presence_of_daytypes(h_test, name='h', X=X)
d_test = get_presence_of_daytypes(d_test, name='d', X=X)
w_test = get_presence_of_daytypes(w_test, name='w', X=X)
X = get_presence_of_daytypes(X, name='X')

print('Saving data')
h_test.to_csv(data_path/f'processed/h.csv')
d_test.to_csv(data_path/f'processed/d.csv')
w_test.to_csv(data_path/f'processed/w.csv')
X.to_csv(data_path/f'processed/X.csv')

print('Datasets are saved and ready')
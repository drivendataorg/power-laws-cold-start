import pandas as pd
import numpy as np
from tqdm import tqdm
import holidays


def _surface_id(x_series):
    d = {
        'xx-small': 1,
        'x-small': 2,
        'small': 3,
        'medium': 4,
        'large': 5,
        'x-large': 6,
        'xx-large': 7
    }
    res = x_series.map(pd.Series(d))
    assert all(res.notnull())
    return res


def _base_temperature_id(x_series):
    d = {
        'low': 0,
        'high': 1
    }
    res = x_series.map(pd.Series(d))
    assert all(res.notnull())
    return res


def calc_days_off(res, meta, shift=0, prefix=''):
    assert shift in [-1, 0, 1]
    col = {0: 'is_day_off', 1: 'is_next_day_off', -1: 'is_prev_day_off'}.get(shift)
    if prefix != '':
        col = prefix + "_" + col
    res[col] = None
    for i, d in enumerate(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
        sel = res[((res['dayofweek'] + shift) % 7) == i]
        res.loc[sel.index, col] = sel['series_id'].map(meta['{}_is_day_off'.format(d)])
    assert all(res[col].notnull())
    return res


def _holidays(min_date, max_date, countries=None):
    dates = pd.date_range(min_date, max_date, freq="D")
    d = {}
    if countries is None:
        countries = ['FRA', 'AU', 'US', 'DE', 'custom']
    for c in countries:
        if c != 'custom':
            h = holidays.CountryHoliday(c)
            col_name = "is_holiday_{}".format(c.lower())
            d[col_name] = dates.map(lambda x: x in h)
        else:
            custom_countries = [
                'US', "PL", "AU", "FRA", "DE", "CZ", "SK", "ES", "PT",
                "CH", "NZ", "UK", "IT", "AT", "BE", "ZA"
            ]
            h_set = [holidays.CountryHoliday(c) for c in custom_countries]
            col_name = "is_holiday_custom"
            d[col_name] = dates.map(lambda x: any([x in h for h in h_set]))
    return pd.DataFrame(d, index=dates)


def calc_holidays(res):
    min_date, max_date = res['date'].min(), res['date'].max()
    h = _holidays(min_date, max_date)
    res = res.join(h, on='date', how='left')
    return res


def calc_interim_features(train_test, meta, meta_org, mode='hourly'):
    res = train_test.copy()
    res['date'] = pd.to_datetime(res['timestamp'].dt.date)
    res['dayofweek'] = res['timestamp'].dt.dayofweek
    res['surface_id'] = _surface_id(res['series_id'].map(meta['surface']))
    res['base_temperature_id'] = _base_temperature_id(res['series_id'].map(meta['base_temperature']))

    res['series_start_timestamp'] = res.groupby('series_id')['timestamp'].transform(min)
    res['series_start_dayofweek'] = res['series_start_timestamp'].dt.dayofweek
    res['series_start_dayofyear'] = res['series_start_timestamp'].dt.dayofyear
    dt = (res['timestamp'] - res['series_start_timestamp'])
    res['h_num'] = (dt / pd.to_timedelta('1 hour')).map(int)
    res['d_num'] = (dt / pd.to_timedelta('1 day')).map(int)

    for shift in [0, 1, -1]:
        res = calc_days_off(res, meta, shift=shift)
        res = calc_days_off(res, meta_org, shift=shift, prefix='v1')
    res = calc_holidays(res)
    return res


################

def days_off_matrix(meta):
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    res = pd.concat([
        pd.DataFrame({'series_id': meta.index, 'dayofweek': i, 'is_day_off': meta[f'{day}_is_day_off']})
        for i, day in enumerate(days)
    ], axis=0, ignore_index=True)
    return res


def is_day_off_features(df, meta, lags, prefix=''):
    days_matrix = days_off_matrix(meta)

    def _is_day_off_feat(shift):
        tmp = pd.concat([
            df['series_id'],
            (df['date'].dt.dayofweek + shift).mod(7).rename("dayofweek")
        ], axis=1)
        tmp = tmp.merge(days_matrix, on=['series_id', 'dayofweek'], how='left')
        assert all(tmp['is_day_off'].notnull())
        p = 'lag' if shift < 0 else 'f'
        col_name = f'is_day_off_{p}_d_{abs(shift):03d}'
        if prefix != '':
            col_name = prefix + "_" + col_name
        return tmp['is_day_off'].rename(col_name)
                
    return pd.concat([
        _is_day_off_feat(lag) for lag in lags
    ], axis=1)


def add_is_day_off_features(df, meta, lags=None, prefix=''):
    if lags is None:
        lags = range(-14, 15)
    return pd.concat([df, is_day_off_features(df, meta, lags=lags, prefix=prefix)], axis=1)


def add_is_holiday_features(df, lags=None, date_column='date', countries=None):
    if lags is None:
        lags = range(-15, 16)
    if countries is None:
        countries = ['US', 'FRA', 'custom']
    return pd.concat([df, is_holiday_features(df, lags=lags, date_column=date_column, countries=countries)], axis=1)


def is_holiday_features(df, lags, date_column='date', countries=None):
    h = _holidays(
        df[date_column].min() + pd.DateOffset(days=min(lags)),
        df[date_column].max() + pd.DateOffset(days=max(lags)),
        countries=countries
    )
    def _is_holiday_feat(shift):
        tmp = (df[date_column] + pd.DateOffset(days=shift)).to_frame()
        tmp = tmp.join(h, on=[date_column], how='left')
        #assert all(tmp['is_holiday_us'].notnull())
        prefix = 'lag' if shift < 0 else 'f'
        col_suffix = f'{prefix}_d_{abs(shift):03d}'
        return tmp.filter(regex='^is_holiday').rename(columns=lambda x: x + '_' + col_suffix)

    return pd.concat([
        _is_holiday_feat(lag) for lag in lags
    ], axis=1)


################

def normalize_left(arr, n, fill_value=np.nan, is_nan_func=None):
    arr = np.pad(arr, (n - len(arr), 0), 'constant', constant_values=fill_value)
    if is_nan_func is None:
        is_nan_func = np.isnan
    isnan = is_nan_func(arr)
    if all(~isnan):
        return arr, len(arr)
    else:
        i = np.where(isnan==True)[0][-1]
        arr[:(i+1)] = np.nan
        return arr, len(arr) - (i + 1)


def normalize_right(arr, n, fill_value=np.nan):
    arr = np.pad(arr, (0, n - len(arr)), 'constant', constant_values=fill_value)
    isnan = np.isnan(arr)
    if all(~isnan):
        return arr, len(arr)
    else:
        i = np.where(isnan == True)[0][0]
        arr[i:] = np.nan
        return arr, i


def add_values(d, arr, prefix, start=1):
    for i, v in enumerate(arr, start=start):
        d[f'{prefix}_{i:03d}'] = v


def _calc_consumption_mean(max_history_days, lag_consumption, lag_is_day_off, lag_dates, working_days, all_holidays=None, data_variant='default'):
    assert len(lag_consumption) >= max_history_days * 24
    res = {}
    for last in range(1, max_history_days + 1):
        slen = last * 24
        lc, off, dt = lag_consumption[-slen:], lag_is_day_off[-slen:], lag_dates[-slen:]
        if data_variant == 'v2':
            # simulate bug in selecting on/off day
            c_on = lc[off==True]
            dt_on = dt[off==True]
            c_off = lc[off==False]
        else:
            c_on = lc[off==False]
            c_off = lc[off==True]
            dt_on = dt[off==False]
        c_mean = np.mean(lc)
        c_mean_min = np.percentile(lc, 100 * 1 / 24)
        c_mean_min24 = np.percentile(lc[-24:], 100 * 1 / 24)
        c_mean_max = np.percentile(lc, 100 * 23 / 24)
        mean_func = lambda x: \
            pd.Series(x.reshape(len(x)//24, 24).mean(axis=1)).ewm(1).mean().iloc[-1]
        if data_variant == 'v2':
            mean_func = np.mean
        c_mean_on = mean_func(c_on) if len(c_on) > 0 else c_mean
        c_mean_off = mean_func(c_off) if len(c_off) > 0 else c_mean
        if data_variant != 'v2':
            c_mean_off = min(c_mean_on, c_mean_off)
        c_mean_w = (working_days * c_mean_on + (7 - working_days) * c_mean_off) / 7
        c_mean_w2 = c_mean_w
        w3_on = np.percentile(
            c_on.reshape(-1, 24).mean(axis=1), q=99
        ) if len(c_on) > 0 else c_mean
        w3_off = c_off.reshape(-1, 24).mean(axis=1).mean() if len(c_off) > 0 else c_mean
        c_mean_w3 = (working_days * w3_on + (7 - working_days) * w3_off) / 7

        is_shutdown = False
        if len(c_on) >= 24 * 3:
            d_last24 = dt_on[-1]
            c_on_first24_mean = np.mean(c_on[:24])
            c_on_last24_mean = np.mean(c_on[-24:])
            if (all_holidays is None or d_last24 not in all_holidays.index) \
               and (c_on_first24_mean / max(1, c_on_last24_mean) >= 3):
                c_mean_w2 = c_on_last24_mean
                is_shutdown = True

        for suffix, value in [
            ('', c_mean),
            ('_is_day_on', c_mean_on), ('_is_day_off', c_mean_off),
            ('_min', c_mean_min), ('_min24', c_mean_min24),
            ('_max', c_mean_max),
            ('_w', c_mean_w), ('_w2', c_mean_w2), ('_w3', c_mean_w3)
        ]:
            res[f'consumption_h_mean{suffix}_last_{last:d}d'] = value
            res[f'consumption_d_mean{suffix}_last_{last:d}d'] = 24 * value

            # WARNING! those features create data leakage in train set
            # unfortunately I was using this during the contest
            # and this is the cause why my local validation scores are lower
            # than the one from public LB (for daily predictions)
            if last == max_history_days:
                res[f'leaking_consumption_h_mean{suffix}'] = value
                res[f'leaking_consumption_d_mean{suffix}'] = 24 * value
        res[f'is_shutdown_last_{last:d}d'] = is_shutdown
    return res


def calc_final_features(train_test, meta, meta_org, verbose=False, tqdm_function=None):
    all_holidays = _holidays(train_test['date'].min(), train_test['date'].max(), countries=['FRA', 'AU', 'US', 'DE']).applymap(int).sum(axis=1)
    all_holidays = all_holidays[all_holidays > 0]
    first_test_timestamp = train_test[train_test['entry_type']=='test']\
        .groupby('series_id')['timestamp'].min()

    sel = train_test[
        (train_test.entry_type.isin(['train', 'cold_start']))
        | (train_test['timestamp'] == train_test['series_id'].map(first_test_timestamp))
    ].copy()
    g = train_test.groupby(['series_id', 'date'])['consumption']
    sel['diff'] = g.transform(np.max) - g.transform(np.min)

    last_cold_start_date = sel[sel.entry_type.isin(['train', 'cold_start'])]\
        .groupby('series_id')['date'].max()

    eps = 1e-3;
    bad_consumption_sel = (sel['diff'] <= eps) \
        & (sel['series_id'].map(last_cold_start_date) != sel['date'])
    sel.loc[bad_consumption_sel, 'consumption'] = None

    groups = sel.groupby('series_id')
    if verbose:
        groups = tqdm(groups) if tqdm_function is None else tqdm_function(groups)
    res = []
    for series_id, g in groups:
        meta_row = meta.loc[series_id]
        meta_org_row = meta_org.loc[series_id]
        working_days = 7 - meta_row.filter(regex='.*is_day_off$').map(int).sum()
        v1_working_days = 7 - meta_org_row.filter(regex='.*is_day_off$').map(int).sum()
        # simulate bug v2 data variant!
        v2_working_days = meta_org_row.filter(regex='.*is_day_off$').map(int).sum()
        last_entry_type = g['entry_type'].iloc[-1]
        assert (last_entry_type == 'train' and len(g) % 24 == 0) \
            or (last_entry_type == 'test' and len(g) % 24 == 1)
        consumption = g['consumption'].values
        is_day_off = g['is_day_off']
        v1_is_day_off = g['v1_is_day_off']
        dates = g['date'].values
        s_timestamp = g['submission_timestamp'].values
        entry_type = g['entry_type'].values

        lag_window = 14 * 24
        pred_window = 14 * 24

        for offs in range(24, len(g), 24):
            lag_consumption = consumption[max(0, offs - lag_window): offs]
            lag_consumption, suffix_len = normalize_left(lag_consumption, lag_window)
            lag_is_day_off = is_day_off[max(0, offs - lag_window): offs]
            lag_is_day_off, _ = normalize_left(lag_is_day_off, lag_window)
            v1_lag_is_day_off = v1_is_day_off[max(0, offs - lag_window): offs]
            v1_lag_is_day_off, _ = normalize_left(v1_lag_is_day_off, lag_window)
            lag_dates = dates[max(0, offs - lag_window): offs]
            lag_dates, _ = normalize_left(lag_dates, lag_window, 
                fill_value=np.datetime64('NaT'), is_nan_func=np.isnat)
            assert len(lag_is_day_off) == len(lag_consumption)
            target = consumption[offs: offs + pred_window]
            target, prefix_len = normalize_right(target, pred_window)
            curr_entry_type = entry_type[offs]
            if (suffix_len == 0) or (prefix_len == 0 and curr_entry_type != 'test'):
                continue
            assert suffix_len % 24 == 0 and prefix_len % 24 == 0

            row = {
                'series_id': series_id,
                'timestamp': dates[offs],
                'submission_timestamp': s_timestamp[offs],
                'date': dates[offs],
                'entry_type': curr_entry_type,
                'cold_start_days': suffix_len // 24,
                'target_days': prefix_len // 24,
                'working_days': working_days / 7,
                'v1_working_days': v1_working_days / 7
            }
            for k, v in _calc_consumption_mean(
                min(7, suffix_len // 24),
                lag_consumption, lag_is_day_off, lag_dates, working_days, all_holidays
            ).items():
                row[k] = v

            for k, v in _calc_consumption_mean(
                min(7, suffix_len // 24),
                lag_consumption, v1_lag_is_day_off, lag_dates, v1_working_days, all_holidays
            ).items():
                row['v1_'+k] = v

            for k, v in _calc_consumption_mean(
                min(7, suffix_len // 24),
                lag_consumption, v1_lag_is_day_off, lag_dates, v2_working_days, all_holidays,
                data_variant='v2'
            ).items():
                row['v2_'+k] = v

            lag_d_consumption = lag_consumption.reshape(-1, 24).sum(axis=1)
            add_values(row, reversed(lag_consumption), 'consumption_lag_h', start=1)
            add_values(row, reversed(lag_d_consumption), 'consumption_lag_d', start=1)
            for prediction_window, prediction_hours, prediction_agg in [
                ('hourly', 24, 1),
                ('daily', 24 * 7, 24),
                ('weekly', 24 * 7 * 2, 24 * 7)
            ]:
                curr_target = target[:prediction_hours].reshape(-1, prediction_agg).sum(axis=1)
                add_values(row, curr_target, f'target_f_{prediction_window[0]}', start=0)
            res.append(row)

    res = pd.DataFrame(res)
    res = add_is_day_off_features(res, meta)
    res = add_is_day_off_features(res, meta_org, prefix='v1')
    res = add_is_holiday_features(res, lags=range(-14, 15), countries=['US', 'FRA', 'custom'])
    # simulate bug in previous build_features version
    for c in res.filter(regex='^is_holiday_custom').columns:
        res['v1_' + c] = False
    # simulate bug in wrong timestamp/date setting for weekly pred
    res['v2_timestamp'] = res['submission_timestamp']
    res['v2_date'] = pd.to_datetime(res['submission_timestamp'].dt.date)
    res = add_daily_temp_features(res, train_test=train_test, lags=range(-14, 15))
    return res


def calc_daily_temperatures(df):
    res = df.groupby(['series_id', 'date'], as_index=False)['temperature'].mean()
    res['temperature'] = res['temperature'].fillna(
        res.groupby('date')['temperature'].transform(np.mean)
    ).fillna(
        res.groupby(res.date.dt.dayofyear)['temperature'].transform(np.mean)
    )
    assert res['temperature'].isnull().sum() == 0
    return res


def add_daily_temp_features(df, train_test, lags):
    t = calc_daily_temperatures(train_test)
    res = [df]
    for lag in lags:
        tmp = df[['series_id', 'timestamp']].copy()
        tmp['date'] = tmp['timestamp'] + pd.DateOffset(days=lag)
        tmp = tmp.merge(t, on=['series_id', 'date'], how='left')
        prefix = "f" if lag >= 0 else "lag"
        tmp['temperature'] = tmp['temperature'].fillna(
            tmp.groupby('date')['temperature'].transform(np.mean)
        ).fillna(
            tmp['temperature'].mean()
        )
        res.append(tmp['temperature'].rename(f"temperature_{prefix}_d_{abs(lag):03d}"))
    return pd.concat(res, axis=1)
"""
Making predictions using just linear regression.
"""
import numpy as np
from tqdm import tqdm_notebook
from scipy.optimize import minimize

from coldstart.utils import _is_day_off, _get_next_weekday, group_sum, _is_holiday
from coldstart.utils import _get_next_date

class LinearRegression(object):
    """
    Encapsulation of everything needed for making predictions
    using multiple linear regression models.
    """
    def __init__(self, metadata, use_holidays=True, input_days=7):
        self.train_data = {
            'hourly': {i:{} for i in range(24)},
            'daily': {i:{} for i in range(7)},
            'weekly': {i:{} for i in range(2)},
        }
        self._metadata = metadata
        self._use_holidays = use_holidays
        self._input_days = input_days

    def prepare_data(self, df):
        self._prepare_hourly_data(df)
        self._prepare_daily_data(df)
        self._prepare_weekly_data(df)

    def _prepare_hourly_data(self, df):
        for series_id in tqdm_notebook(df.series_id.unique(), desc='Preparing data'):
            sub_df = df[df.series_id == series_id]
            consumption = sub_df.consumption.values
            is_day_off = self._get_is_day_off_from_df(sub_df)
            for n_days in range(2, 2 + self._input_days):
                for start_idx in range(len(is_day_off)-n_days):
                    key = ''.join([str(i) for i in is_day_off[start_idx:start_idx + n_days]])
                    x = np.zeros((24, n_days-1))
                    for i in range(n_days -1):
                        x[:, i] = consumption[(start_idx + i)*24:(start_idx + i + 1)*24]
                    val_idx = start_idx + n_days - 1
                    y = consumption[val_idx*24:(val_idx+1)*24]
                    y_mean = np.mean(y)
                    for offset in range(24):
                        self._add_train_data(x[offset:offset+1]/y_mean, y[offset:offset+1]/y_mean,
                                             'hourly', offset, key)

    def _prepare_daily_data(self, df):
        for series_id in tqdm_notebook(df.series_id.unique(), desc='Preparing data'):
            sub_df = df[df.series_id == series_id]
            consumption = sub_df.consumption.values
            consumption = group_sum(consumption, 24)
            is_day_off = self._get_is_day_off_from_df(sub_df)
            for input_days in range(1, 1 + self._input_days):
                for start_idx in range(len(is_day_off)-input_days-7):
                    key = ''.join([str(i) for i in is_day_off[start_idx:start_idx + input_days]])

                    x = consumption[start_idx: start_idx + input_days]
                    x = np.expand_dims(x, axis=0)
                    val_idx = start_idx + input_days
                    y = consumption[val_idx:val_idx + 7]
                    y_mean = np.mean(y)
                    for offset in range(7):
                        final_key = key + str(is_day_off[val_idx + offset])
                        self._add_train_data(x/y_mean, [y[offset]/y_mean], 'daily', offset, final_key)

    def _prepare_weekly_data(self, df):
        for series_id in tqdm_notebook(df.series_id.unique(), desc='Preparing data'):
            sub_df = df[df.series_id == series_id]
            consumption = sub_df.consumption.values
            consumption = group_sum(consumption, 24)
            is_day_off = self._get_is_day_off_from_df(sub_df)
            for input_days in range(1, 1 + self._input_days):
                for start_idx in range(len(is_day_off)-input_days-14):
                    key = ''.join([str(i) for i in is_day_off[start_idx:start_idx + input_days]])

                    x = consumption[start_idx: start_idx + input_days]
                    x = np.expand_dims(x, axis=0)
                    val_idx = start_idx + input_days
                    y = consumption[val_idx:val_idx + 14]
                    y = group_sum(y, 7)
                    y_mean = np.mean(y)
                    for offset in range(2):
                        final_key = key
                        self._add_train_data(x/y_mean, [y[offset]/y_mean], 'weekly', offset, final_key)

    def _add_train_data(self, x, y, window, offset, key):
        if key in self.train_data[window][offset]:
            self.train_data[window][offset][key] = {
                'n': self.train_data[window][offset][key]['n'] + 1,
                'x': np.concatenate([x, self.train_data[window][offset][key]['x']], axis=0),
                'y': np.concatenate([y, self.train_data[window][offset][key]['y']], axis=0),
            }
        else:
            self.train_data[window][offset][key] = {
                'n': 1,
                'x': x,
                'y': y,
            }

    def fit(self):
        for window in tqdm_notebook(self.train_data, desc='Fitting'):
            for offset in self.train_data[window]:
                iterator = tqdm_notebook(
                    self.train_data[window][offset],
                    leave=False,
                    desc='Fitting window: %s offset: %i' % (window, offset))
                for key in iterator:
                    x = self.train_data[window][offset][key]['x']
                    y = self.train_data[window][offset][key]['y']
                    n_parameters = x.shape[1]
                    if window == 'hourly':
                        x0 = np.ones(n_parameters)/n_parameters
                    elif window == 'daily':
                        x0 = np.ones(n_parameters)/n_parameters
                    elif window == 'weekly':
                        x0 = np.ones(n_parameters)/n_parameters*7
                    output = minimize(
                        _cost_function, x0,
                        args=(x, y),
                        bounds=[(0, np.inf)]*n_parameters)
                    weights = output.x
                    self.train_data[window][offset][key]['weights'] = weights
                    self.train_data[window][offset][key]['nmae'] = _cost_function(
                        weights, x, y
                    )

    def predict(self, window, series_id, consumption, weekdays, dates):
        if window == 'hourly':
            return self._hourly_predict(series_id, consumption, weekdays, dates)
        elif window == 'daily':
            return self._daily_predict(series_id, consumption, weekdays, dates)
        else:
            return self._weekly_predict(series_id, consumption, weekdays, dates)

    def _hourly_predict(self, series_id, consumption, weekdays, dates):
        is_day_off = self._get_is_day_off(weekdays, series_id, dates)
        is_day_off = is_day_off[-self._input_days:]
        is_day_off.append(self._is_day_off(_get_next_weekday(weekdays[-1]), series_id,
                                           _get_next_date(dates[-1])))
        key = ''.join([str(i) for i in is_day_off])
        # print(key, weekdays)
        while 1:
            if key in self.train_data['hourly'][0]:
                break
            else:
                # print(key, 'not found')
                key = key[1:]
            if not len(key):
                raise KeyError('Empty key')
        consumption = consumption[-(len(key)-1)*24:]

        x = np.zeros((24, len(key)-1))
        for i in range(len(key)-1):
            x[:, i] = consumption[i*24:(i + 1)*24]
        pred = []
        for offset in range(24):
            weights = self.train_data['hourly'][offset][key]['weights']
            pred.append(x[offset:offset+1].dot(weights)[0])
        return np.array(pred)

    def _daily_predict(self, series_id, consumption, weekdays, dates):
        is_day_off = self._get_is_day_off(weekdays, series_id, dates)
        is_day_off = is_day_off[-self._input_days:]
        org_key = ''.join([str(i) for i in is_day_off])
        pred = []
        for offset in range(7):
            weekday = weekdays[-1]
            date = dates[-1]
            for _ in range(offset+1):
                weekday = _get_next_weekday(weekday)
            date = _get_next_date(date, offset+1)
            key = org_key + str(self._is_day_off(weekday, series_id, date))
            while 1:
                if key in self.train_data['daily'][offset]:
                    break
                else:
                    # print(key, 'not found')
                    key = key[1:]
                if not len(key):
                    msg = 'Key not found: %s\tWindow: %s\tOffset: %s' % (org_key, 'daily', offset)
                    raise KeyError(msg)
            x = consumption[-(len(key)-1)*24:]
            x = group_sum(x, 24)
            x = np.expand_dims(x, axis=0)
            weights = self.train_data['daily'][offset][key]['weights']
            # print(consumption.shape, x.shape, weights.shape, key)
            pred.append(x.dot(weights)[0])
        return np.array(pred)

    def _weekly_predict(self, series_id, consumption, weekdays, dates):
        is_day_off = self._get_is_day_off(weekdays, series_id, dates)
        is_day_off = is_day_off[-self._input_days:]
        org_key = ''.join([str(i) for i in is_day_off])
        pred = []
        for offset in range(2):
            key = org_key[:]
            while 1:
                if key in self.train_data['weekly'][offset]:
                    break
                else:
                    # print(key, 'not found')
                    key = key[1:]
                if not len(key):
                    msg = 'Key not found: %s\tWindow: %s\tOffset: %s' % (org_key, 'weekly', offset)
                    raise KeyError(msg)
            x = consumption[-(len(key))*24:]
            x = group_sum(x, 24)
            x = np.expand_dims(x, axis=0)
            weights = self.train_data['weekly'][offset][key]['weights']
            # print(consumption.shape, x.shape, weights.shape, key)
            pred.append(x.dot(weights)[0])
        return np.array(pred)

    def _get_is_day_off(self, weekdays, series_id, dates):
        weekdays = weekdays[::24]
        dates = dates[::24]
        is_day_off = [self._is_day_off(weekday, series_id, date) \
                      for weekday, date in zip(weekdays, dates)]
        return is_day_off

    def _is_day_off(self, weekday, series_id, date):
        ret = _is_day_off(series_id, weekday, self._metadata)
        if self._use_holidays:
            ret = ret or _is_holiday(date)
        return int(ret)

    def _get_is_day_off_from_df(self, df):
        if self._use_holidays:
            is_day_off = df.is_holiday.values[::24]
        else:
            is_day_off = df.is_day_off.values[::24]
        is_day_off = [int(value) for value in is_day_off]
        return is_day_off

def _cost_function(params, X, y):
    return np.mean(np.abs(y - X.dot(params)))

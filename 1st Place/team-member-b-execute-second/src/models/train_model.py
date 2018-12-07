# -*- coding: utf-8 -*-
import os
import re
import itertools
import logging
import click
import pandas as pd
import numpy as np
import keras

from sklearn.model_selection import train_test_split


from src.keras_utils import keras_initialize_random_state, keras_set_random_state, \
    generate_simple_model, keras_hash_of_model
from src.common import ensure_dir, tolist, log_to_file, \
    prediction_params, compute_nmae, \
    filter_columns, prepare_values_for_nn, \
    hash_of_numpy_array, hash_of_pandas_df

from src.models import keras_models

DEBUG = False


class Configuration():

    def get(self, key):
        parts = key.split("/")
        return self._get(*parts)

    def _get(self, key, *args):
        if key == 'data_variant':
            if (args[0] == 'daily') or (args[0] == 'hourly' and int(args[1]) in [1, 2, 7]):
                return 'v1'
            if (args[0] == 'hourly' and int(args[1]) in [3, 4, 5, 6]):
                return 'v2'
            return 'default'
        if key == 'boost_weights':
            if args[0] == 'weekly' and int(args[1]) != 5:
                return True
            return False
        if key == 'epochs':
            if args[0] == 'hourly' and int(args[1]) in [3, 4, 5, 6, 7]:
                return 200
            return 256
        if key == 'batch_size':
            if args[0] == 'hourly' and int(args[1]) in [3, 4, 5, 6, 7]:
                return 500
            return 512
        if key == 'patience':
            if args[0] == 'hourly' and int(args[1]) in [3, 4, 5, 6, 7]:
                return 100
            return 50
        if key == 'layers_num':
            if args[0] == 'hourly' and int(args[1]) in [3, 4, 5]:
                return 1
            return 2
        if key == 'network_size':
            if args[0] == 'hourly' and int(args[1]) in [3, 4, 5]:
                return 512
            if args[0] == 'hourly':
                return 128
            return 64
        if key == 'scale_min_adj':
            if args[0] == 'hourly' and int(args[1]) == 1:
                return 0.8
            else:
                return 0.9
        if key == 'scale_max_adj':
            if args[0] == 'weekly':
                return 1.1
            else:
                return 4.0
        if key == 'train_validate_split':
            if args[0] == 'weekly':
                return 'only_test_dates'
            if args[0] == 'daily':
                return f'daily{args[1]}'
            if args[0] == 'hourly' and int(args[1]) in [1, 2]:
                return 'only_test_dates'
            if args[0] == 'hourly' and int(args[1]) in [3, 4, 5, 6]:
                return f'v2_hourly{args[1]}'
            return 'default'  # hourly7
        if key == 'f_regex':
            if args[0] == 'hourly' and int(args[1]) in [2]:
                return ("^("
                    r"consumption_h_mean_last_\d+d"
                    r"|consumption_h_mean_(w|min|max|is_day_on|is_day_off)_last_\d+d"
                    r"|consumption_lag_(d|h)_\d+"
                    r"|is_day_off_(f|lag)_(d|h)_\d+"
                    r"|is_holiday_custom_(f|lag)_(d|h)_\d+"
                    r"|working_days"
                ")$")
            elif args[0] == 'hourly' and int(args[1]) in [3, 4, 5, 6]:
                return ("^("
                    r"leaking_consumption_h_mean"
                    r"|leaking_consumption_h_mean_(w|is_day_on|is_day_off)"
                    r"|consumption_lag_(d|h)_\d+"
                    r"|is_day_off_(f|lag)_(d|h)_\d+"
                    r"|is_holiday_(us|fra)_f_(d|h)_\d+"
                    r"|is_eq_target_day_off_(f|lag)_d_\d+"  # only used in cs6
                ")$")
            elif args[0] == 'hourly' and int(args[1]) in [7]:
                return ("^("
                    r"consumption_h_mean_last_\d+d"
                    r"|consumption_h_mean_(w|is_day_on|is_day_off)_last_\d+d"
                    r"|consumption_lag_(d|h)_\d+"
                    r"|is_day_off_(f|lag)_(d|h)_\d+"
                    r"|is_holiday_(us|fra)_f_(d|h)_\d+"
                ")$")
            elif args[0] == 'hourly' and int(args[1]) in [1]:
                return ("^("
                    r"consumption_h_mean_last_\d+d"
                    r"|consumption_h_mean_(w|w2|min|max|is_day_on|is_day_off)_last_\d+d"
                    r"|consumption_lag_(d|h)_\d+"
                    r"|is_day_off_(f|lag)_(d|h)_\d+"
                    r"|is_holiday_custom_(f|lag)_(d|h)_\d+"
                    r"|is_shutdown_last_\d+d"
                    r"|working_days"
                ")$")
            elif args[0] == 'daily': 
                return ("^("
                    r"consumption_lag_(d|h)_\d+"
                    r"|is_day_off_(f|lag)_(d|h)_\d+"
                    r"|leaking_consumption_d_mean"
                    r"|leaking_consumption_d_mean_(is_day_off|is_day_on|min|max|w|w2)"
                    r"|working_days"
                ")$")
            elif args[0] == 'weekly':
                return ("^("
                    r"consumption_d_mean_last_\d+d"
                    r"|consumption_d_mean_(is_day_off|is_day_on|min|max|w)_last_\d+d"
                    r"|consumption_lag_(d|h)_\d+"
                    r"|is_day_off_(f|lag)_(d|h)_\d+"
                    r"|working_days"
                ")$")
            else:
                raise Exception("invalid")
        raise Exception(f"Unknown key={key}")


def prepare_data_variant(df, variant='default'):
    def copy_columns(prefix):
        for c in df.filter(regex='^'+prefix+'_').columns:
            cc = re.sub('^'+prefix+'_', '', c)
            assert cc in df.columns
            df[cc] = df[c]
    if variant == 'v1':
        df = df.copy()
        copy_columns('v1')
        df = df.filter(regex='^(?!(leaking_)?consumption_(d|h)_mean_(w3|min24))')
    elif variant == 'v2':
        df = df.copy()
        copy_columns('v1')
        copy_columns('v2')
        df = pd.concat([
            df[df.entry_type.isin(['train', 'cold_start'])].sort_values(['series_id', 'timestamp', 'cold_start_days']).reset_index(),
            df[df.entry_type.isin(['test'])].sort_values(['series_id', 'timestamp', 'cold_start_days']).reset_index()
        ], axis=0, sort=False, ignore_index=True)

        df = df.filter(regex='^(?!leaking_consumption_h_mean_(w2|w3|min24|min|max))')
        df = df.filter(regex='^(?!leaking_consumption_d_mean)')
        df = df.filter(regex='^(?!consumption_(d|h)_mean_)')
        df = df.filter(regex='^(?!(temperature|working_days|is_shutdown))')
        df = df.filter(regex='^(?!is_holiday_custom)')
        df = df.filter(regex='^(?!is_holiday_(us|fra)_lag)')
        df = df.filter(regex='^(?!is_holiday_(us|fra)_f_d_(00[1-9]|01))')
        df = df.filter(regex='^(?!is_day_off_lag_d_(01|00[7-9]))')
        df = df.filter(regex='^(?!is_day_off_f_d_(01|00[1-9]))')
    return df.filter(regex='^(?!(v[12]_|index))')


def order_very_old_features(features, reorder_lags=True):
    def sort_key(x):
        tmp = x.split("_")
        if re.match('^consumption', x):
            i = int(tmp[-1], 10)
            if reorder_lags:
                return ((i-1)//24, 24 - (i-1)%24, x)
            else:
                return ((i-1)//24, i, x)
        elif re.match('^leaking_consumption', x):
            return (50, 0, x)
        elif re.match('^is_day_off', x):
            return (60, -int(tmp[-1], 10), x)
        elif re.match('^is_holiday_us', x):
            return (70, 0, x)
        elif re.match('^is_holiday_fra', x):
            return (71, 0, x)
        elif re.match('^is_eq_target_day_off', x):
            return (72, int(tmp[-1], 10), x)
        return (99, 0, x)
    return sorted(features, key=sort_key)


def prepare_data_set(df, prediction_window, cold_start_days, features_regex=None,
                     for_training=True, scale_min_adj=None, scale_max_adj=None,
                     test_set=None, boost_weights=False, train_test_set=None, 
                     data_variant='default'):

    df = prepare_data_variant(df, data_variant)

    force_is_day_off = False
    old_style_feat = False
    if data_variant == 'v2':
        df['cs'] = df['cold_start_days'].clip(1, 7)
        df = df.sort_values(by=['cs', 'series_id', 'timestamp']).reset_index(drop=True)
        del df['cs']
        force_is_day_off = True
        old_style_feat = True
        if prediction_window == 'hourly' and cold_start_days >= 6:
            for c in df.filter(regex='^is_day_off').columns:
                cc = re.sub('lag_d_\d+', 'f_d_000', c)
                new_c = c.replace("is_day_off", "is_eq_target_day_off")
                logging.info("- adding column %s", new_c)
                df[new_c] = df[c] == df[cc]

    if scale_min_adj is None:
        scale_min_adj = 0.5 if prediction_window == 'weekly' else 0.9
    if scale_max_adj is None:
        scale_max_adj = 5.0 if prediction_window == 'weekly' else 4.0
    params = prediction_params(prediction_window)
    sel = df
    sel = sel[sel.cold_start_days >= cold_start_days]

    if for_training:
        sel = sel[sel.target_days >= params['prediction_days']]
    res, features, targets = filter_columns(sel,
        prediction_window=prediction_window, cold_start_days=cold_start_days,
        force_is_day_off=force_is_day_off
    )
    if old_style_feat:
        features = order_very_old_features(features, reorder_lags=cold_start_days >= 6)
    if features_regex is not None:
        logging.info("filtering features using regex=%s", features_regex)
        features = [f for f in features if re.match(features_regex, f)]
        logging.info("using features: %s", features)
    assert res[features].isnull().sum().sum() == 0
    assert for_training is False or res[targets].isnull().sum().sum() == 0
    y_res = res[targets]

    res['k'] = 1
    if boost_weights:
        sel_test_set = test_set[
            (test_set.prediction_window == prediction_window)
            & (
                (test_set.cold_start_days == cold_start_days)
                if cold_start_days < 7 else 
                (test_set.cold_start_days >= cold_start_days)
            )
        ]
        sel_test_dates = sel_test_set['date'].unique()
        a_len = res['date'].isin(sel_test_dates).sum()
        b_len = len(res) - a_len
        k = 8
        alpha = len(sel) * k / (k * a_len + b_len)
        beta = len(sel) / (k * a_len + b_len)
        logging.info("boosting weights of %d samples with k=%.1f alpha=%.4f beta=%.4f", a_len, k, alpha, beta)
        res['k'] = res['date'].isin(sel_test_dates).map({False: beta, True: alpha})

    logging.info(f"scale_min_adj={scale_min_adj:.2f} scale_max_adj={scale_max_adj:.2f}")
    false_is_negative = True
    if prediction_window=='hourly' and cold_start_days >= 3 and cold_start_days <= 6:
        false_is_negative = False
    flags_vs_day0 = prediction_window=='hourly' and cold_start_days >= 7
    logging.info(f"false_is_negative={false_is_negative} flags_vs_day0={flags_vs_day0}")
    res, scale_up = prepare_values_for_nn(res, features, targets, prediction_window=prediction_window, 
        cold_start_days=cold_start_days, 
        scale_min_adj=scale_min_adj, scale_max_adj=scale_max_adj,
        false_is_negative=false_is_negative, flags_vs_day0=flags_vs_day0
    )
    assert res[features].isnull().sum().sum() == 0
    assert for_training is False or res[targets].isnull().sum().sum() == 0
    return res, y_res, features, targets, scale_up


class PowerLawsModelV2(object):

    def __init__(self, input_filepath='data/processed/train_test.hdf5', log_dir=None, epochs=256):
        logging.info("loading data from %s", input_filepath)
        self.epochs = epochs

        self.test_set = pd.read_hdf(input_filepath, "test")
        # self.submission = pd.read_hdf(input_filepath, "submission")
        self.submission = pd.read_hdf('data/processed/train_test.hdf5', "submission")

        self.test_set = self.test_set.join(
            self.submission.groupby('series_id')['prediction_window'].first(),
            on='series_id', how='left'
        )

        self.train_validate_set = pd.concat([
            pd.read_hdf(input_filepath, "train"), 
            pd.read_hdf(input_filepath, "validate")
        ], axis=0, sort=False, ignore_index=True)
        
        self.train_test_set = pd.concat([
            self.train_validate_set,
            self.test_set
        ], axis=0, sort=False, ignore_index=True)

        self.train_validate_splits = self.initialize_train_validate_splits(self.train_validate_set, self.test_set, self.submission)

        self.conf = Configuration()

        self.log_dir = log_dir
        if self.log_dir:
            ensure_dir(self.log_dir)
        self.models = {}
        self.prediction_windows = ['hourly', 'daily', 'weekly']
        self.cold_start_days = range(1, 8)
        super(PowerLawsModelV2, self).__init__()

    def initialize_train_validate_splits(self, train_df, test_df, submission):
        res = {}
        def load_series(fn):
            t = pd.read_hdf(fn, "train_series")
            v = pd.read_hdf(fn, "validate_series")
            return t, v

        # recreated old split behaviour
        tmp = train_df.copy()
        tmp['cs'] = tmp['cold_start_days'].clip(1, 7)
        tmp = tmp.sort_values(by=['cs', 'series_id', 'timestamp']).reset_index(drop=True)

        for cs in [3, 4, 5, 6, 7]:
            train_series = tmp[tmp.cold_start_days >= cs].series_id.unique()
            t, v = train_test_split(train_series, random_state=0)
            ts = train_df[train_df.series_id.isin(t)]
            vs = train_df[train_df.series_id.isin(v)]
            if cs == 7:
                res[f'default'] = (ts, vs)
            else:
                res[f'v2_hourly{cs}'] = (ts, vs)

        vs_only_test_dates = vs[vs['date'].isin(test_df['date'])]
        res['only_test_dates'] = (ts, vs_only_test_dates)

        for cs in range(1, 8):
            train_series = train_df[train_df.cold_start_days >= cs].series_id.unique()
            t, v = train_test_split(train_series, random_state=0)
            ts = train_df[train_df.series_id.isin(t)]
            vs = train_df[train_df.series_id.isin(v)]
            res[f'daily{cs}'] = (ts, vs)

        for cs in range(1, 8):
            train_series = train_df[train_df.cold_start_days >= cs].series_id.unique()
            t, v = train_test_split(train_series, random_state=0)
            ts = train_df[train_df.series_id.isin(t)]
            vs = train_df[train_df.series_id.isin(v)]
            res[f'hourly{cs}'] = (ts, vs)
   
        return res

    def train_validate_split(self, prediction_window, cold_start_days):
        split_name = self.conf.get(f"train_validate_split/{prediction_window}/{cold_start_days}")
        (t_df, v_df) = self.train_validate_splits[split_name]
        h = hash_of_numpy_array(np.sort(t_df.series_id.unique()))
        logging.info(f"train_validate_split({prediction_window}, {cold_start_days})={split_name}, hash={h}")
        return (t_df, v_df)

    def _predict(self, model_name, df, scale_up=True):
        m_desc = self.models[model_name]
        model = m_desc['model']
        features, targets, scale_up_func = m_desc['features'], m_desc['targets'], m_desc['scale_up']
        y = model.predict(df[features])
        y = pd.DataFrame(y, columns=targets, index=df.index)
        if scale_up:
            return scale_up_func(y, df)
        else:
            return y

    def _unpack_prediction(self, y_pred, x_df, prediction_window):
        params = prediction_params(prediction_window)
        dt = pd.DateOffset(hours=params['prediction_agg'])
        merged = pd.concat([x_df[['series_id', 'submission_timestamp']], y_pred], axis=1)
        unpacked = []
        for _, row in merged.iterrows():
            series_id, t = row['series_id'], row['submission_timestamp']
            for i, c in enumerate(y_pred.columns):
                unpacked.append({
                    'series_id': series_id,
                    'timestamp': t + i * dt,
                    'consumption': row[c]
                })
        unpacked = pd.DataFrame(unpacked)
        return unpacked

    def predict(self, entry_type, prediction_window, cold_start_days, unpack=False):
        model_name = f"model_{prediction_window}_cs{cold_start_days:02d}"
        assert model_name in self.models.keys(), f"model not found: {model_name}"
        train_set, validate_set = self.train_validate_split(prediction_window, cold_start_days)
        df = {
            'train': train_set, 'validate': validate_set, 'test': self.test_set
        }.get(entry_type)
        if entry_type == 'test':
            p = {'weekly1': 'weekly', 'weekly2': 'weekly'}.get(prediction_window, prediction_window)
            sel_series = self.submission[
                self.submission.prediction_window == p
            ]['series_id'].unique()
            df = df[df.series_id.isin(sel_series)]
            df = df[df.cold_start_days >= cold_start_days]
            if cold_start_days < max(self.cold_start_days):
                df = df[df.cold_start_days <= cold_start_days]
        f_regex = self.conf.get(f"f_regex/{prediction_window}/{cold_start_days}")
        dv = self.conf.get(f"data_variant/{prediction_window}/{cold_start_days}")
        x_df, _, _, _, _ = prepare_data_set(
            df, prediction_window, cold_start_days, for_training=False, 
            features_regex=f_regex,
            scale_min_adj=self.conf.get(f'scale_min_adj/{prediction_window}/{cold_start_days}'),
            scale_max_adj=self.conf.get(f'scale_max_adj/{prediction_window}/{cold_start_days}'),
            train_test_set=self.train_test_set, data_variant=dv
        )
        y_pred = self._predict(model_name, x_df)

        if unpack:
            return self._unpack_prediction(y_pred, x_df, prediction_window)
        else:
            return y_pred

    def _convert_to_submission(self, df):
        sub = self.submission.copy()
        sub['pred_id'] = sub.index
        res = df.merge(
            sub[['series_id', 'timestamp', 'pred_id', 'temperature', 'prediction_window']],
            on=['series_id', 'timestamp'], how='left'
        )
        assert all(res['pred_id'].notnull())
        columns = [
            'pred_id', 'series_id', 'timestamp', 'temperature',
            'consumption', 'prediction_window'
        ]
        res = res[columns].sort_values(by='pred_id')
        return res

    def gen_submission(self, prediction_window=None, cold_start_days=None, fn=None):
        prediction_windows = tolist(prediction_window) if prediction_window is not None else self.prediction_windows
        cold_start_days = tolist(cold_start_days) if cold_start_days is not None else self.cold_start_days
        res = []
        for p in prediction_windows:
            for cs in cold_start_days:
                res.append(
                    self.predict("test", p, cs, unpack=True)
                )
        res = pd.concat(res, ignore_index=True, sort=False)
        res = self._convert_to_submission(res)
        t = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        if fn is None:
            fn = os.path.join(self.log_dir or "/tmp", f"submission-{t}.csv")
        logging.info("writing submission %s", fn)
        res.to_csv(fn, index=False)
        return res.set_index("pred_id")

    def train(self, random_state=0, verbose=True):
        for p, cs in itertools.product(
            self.prediction_windows, self.cold_start_days
        ):
            self._train(p, cs, random_state=random_state, verbose=verbose)

    def _train(self, prediction_window, cold_start_days, random_state=0, verbose=True, epochs=None,
               *args, **kwargs):
        if epochs is None:
            epochs = self.conf.get(f"epochs/{prediction_window}/{cold_start_days}")
        batch_size = self.conf.get(f"batch_size/{prediction_window}/{cold_start_days}")
        model_name = f"model_{prediction_window}_cs{cold_start_days:02d}"
        logging.info("training %s", model_name)

        f_regex = self.conf.get(f"f_regex/{prediction_window}/{cold_start_days}")
        train_set, validate_set = self.train_validate_split(prediction_window, cold_start_days)

        boost_weights = self.conf.get(f"boost_weights/{prediction_window}/{cold_start_days}")

        dv = self.conf.get(f"data_variant/{prediction_window}/{cold_start_days}")
        x_train, y_train, features, targets, scale_up = prepare_data_set(
            train_set, prediction_window, cold_start_days, features_regex=f_regex,
            scale_min_adj=self.conf.get(f'scale_min_adj/{prediction_window}/{cold_start_days}'),
            scale_max_adj=self.conf.get(f'scale_max_adj/{prediction_window}/{cold_start_days}'),
            test_set=self.test_set, boost_weights=boost_weights,
            train_test_set=self.train_test_set, data_variant=dv
        )

        x_validate, y_validate, _, _, _ = prepare_data_set(
            validate_set, prediction_window, cold_start_days, features_regex=f_regex,
            scale_min_adj=self.conf.get(f'scale_min_adj/{prediction_window}/{cold_start_days}'),
            scale_max_adj=self.conf.get(f'scale_max_adj/{prediction_window}/{cold_start_days}'),
            test_set=self.test_set, boost_weights=boost_weights,
            train_test_set=self.train_test_set, data_variant=dv
        )

        keras_set_random_state(random_state)
        if prediction_window == 'hourly' and cold_start_days >= 6:
            model = keras_models.original_gen_hourly_pred_model(features, cold_start_days=cold_start_days)
        elif prediction_window == 'hourly' and cold_start_days in [3, 4, 5]:
            model = keras_models.old_generate_model(512, len(features), len(targets))
        elif prediction_window == 'daily': # this can be used for perfect match with daily solutions
            model = generate_simple_model(len(features), len(targets), network_size=512, layers_num=1)
        else:
            model = generate_simple_model(len(features), len(targets), 
                layers_num=self.conf.get(f"layers_num/{prediction_window}/{cold_start_days}"),
                network_size=self.conf.get(f"network_size/{prediction_window}/{cold_start_days}"))
        if verbose:
            model.summary()


        h = keras_hash_of_model(model)
        logging.info("hash of {} before training {}".format(model_name, h))

        callbacks = []
        patience = self.conf.get(f"patience/{prediction_window}/{cold_start_days}")
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience))
        callbacks.append(keras.callbacks.ModelCheckpoint(filepath="/tmp/weights3.hdf5", verbose=0, save_best_only=True))

        logging.info(f"training epochs={epochs} batch_size={batch_size} patience={patience}")
        h = hash_of_pandas_df(x_train[features])
        h1 = hash_of_numpy_array(x_train[targets].values)
        h2 = hash_of_numpy_array(x_train['sample_weight'].values)
        logging.info(f"training x_train[feat].shape={x_train[features].shape} h(x_train[feat])={h} h(x_train[targets])={h1} h(w)={h2}")
        h = hash_of_pandas_df(x_validate[features])
        logging.info(f"training x_validate[feat].shape={x_validate[features].shape} h(x_validate[feat])={h}")
        if DEBUG:
            x_train.to_hdf("/tmp/new-train.hdf5", "x_train", mode="w")
            pd.Series(features).to_hdf("/tmp/new-train.hdf5", "features", mode="a")
        model.fit(
            x_train[features].values,
            x_train[targets].values,
            sample_weight=x_train['sample_weight'].values,
            validation_data=(x_validate[features].values, x_validate[targets].values, x_validate['sample_weight'].values),
            verbose=1, epochs=epochs, shuffle=True,
            batch_size=batch_size, callbacks=callbacks,
        )
        model.load_weights('/tmp/weights3.hdf5')

        h = keras_hash_of_model(model)
        logging.info("hash of {} after training {}".format(model_name, h))

        if self.log_dir:
            model.save(os.path.join(self.log_dir, f"{model_name}.h5"))

        self.models[model_name] = {
            'model': model,
            'features': features,
            'targets': targets,
            'scale_up': scale_up
        }

        y_pred = self._predict(model_name, x_validate)
        h = hash_of_numpy_array(y_pred.values)
        logging.info(f"h(y_validate_pred)={h}")

        nmae = compute_nmae(y_pred, y_validate, x_validate['ci'] / x_validate['wi'])
        logging.info(f"model={model_name} nmae={nmae:.6f}")

        if self.log_dir:
            fn = os.path.join(self.log_dir, f"{model_name}_validate.hdf5")
            x_validate.to_hdf(fn, "data", mode="w")
            y_pred.to_hdf(fn, "y_pred", mode="a")
            y_validate.to_hdf(fn, "y_true", mode="a")
            pd.Series(features).to_hdf(fn, "features", mode="a")
            pd.Series(targets).to_hdf(fn, "targets", mode="a")


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='data/processed/train_test.hdf5')
@click.option('--log-dir', type=str, default=None)
@click.option('--output', type=str, default=None)
@click.option('--save-model/--no-save-model', default=False)
@click.option('--prediction-window', type=str)
@click.option('--cold-start-days', type=int, default=1)
@click.option('--epochs', type=int, default=256)
@click.option('--check-submission/--no-check-submission', default=False)
@click.option('--random-state', type=int, default=0)
def main(input_filepath, log_dir, output, save_model, prediction_window, cold_start_days, epochs,
         check_submission, random_state):
    now = pd.Timestamp.now()
    if save_model and log_dir is None:
        log_dir = f"models/{now.strftime('%Y%m%d-%H%M%S')}/"

    if save_model or log_dir is not None:
        log_to_file(os.path.join(log_dir, "train.log"))

    p = PowerLawsModelV2(input_filepath, log_dir=log_dir, epochs=epochs)
    if prediction_window is not None:
        p._train(prediction_window, cold_start_days=cold_start_days, random_state=random_state)
        s = p.gen_submission(
            prediction_window=prediction_window,
            cold_start_days=cold_start_days,
            fn=output
        )
    else:
        p.train(random_state=random_state)
        s = p.gen_submission(fn=output)

    exp_result = "expected-results/twalen-0.2851.csv"
    if check_submission and os.path.exists(exp_result):
        logging.info("checking submission")
        manual = pd.read_csv("data/processed/selected-trivial-predictions.csv").set_index("pred_id")
        reference_s = pd.read_csv(exp_result).set_index("pred_id")
        c = pd.concat([reference_s, s['consumption'].rename("new_consumption")], axis=1).dropna()
        c = c.join(
            p.test_set.groupby('series_id')['cold_start_days'].first(),
            on='series_id', how='left'
        )
        c = c[~c.series_id.isin(manual.series_id)]

        c['mean'] = c.groupby('series_id')['consumption'].transform(np.mean)
        c['diff'] = (c['consumption'] - c['new_consumption']).abs().divide(c['mean'])
        logging.info("BEST_SUBMISSION_CHECK {}/cs={} diff_sum={:.6f} diff_mean={:.6f}".format(
            prediction_window, cold_start_days, c['diff'].sum(), c['diff'].mean()
        ))
        logging.info(str(c.groupby(['prediction_window', 'cold_start_days'], as_index=False)['diff'].mean().round(6)))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    keras_initialize_random_state()

    main()

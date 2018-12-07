# -*- coding: utf-8 -*-
import click
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
#
from src.features.build_features import calc_interim_features


def load_consumption_data(filename, entry_type):
    res = pd.read_csv(filename, parse_dates=['timestamp'])
    if entry_type in ['train', 'cold_start']:
        res = res.iloc[:, 1:]
    res['timestamp'] = pd.to_datetime(res['timestamp'])
    res['entry_type'] = entry_type
    if entry_type == 'test':
        res['consumption'] = None
    res['consumption'] = res['consumption'].astype('float64')
    if entry_type == 'test':
        res['submission_timestamp'] = res['timestamp']
        sel = res[res.prediction_window == 'weekly']
        res.loc[sel.index, 'timestamp'] = sel['timestamp'] - pd.DateOffset(days=6)
    else:
        res['submission_timestamp'] = res['timestamp']
    columns = [
        'series_id', 'timestamp', 'submission_timestamp', 
        'temperature', 'consumption', 'entry_type'
    ]
    return res[columns]


def load_submission(filename):
    res = pd.read_csv(filename, parse_dates=['timestamp']).set_index('pred_id')
    test = res.groupby('series_id', as_index=False).first().reset_index(drop=True)\
        [['series_id', 'timestamp', 'prediction_window']]
    return res, test


def load_meta(filename, fix_errors=True):
    res = pd.read_csv(filename).set_index('series_id')
    # cleanup
    if fix_errors:
        g = res.filter(regex=".*_is_day_off")
        columns = g.columns
        off = g.sum(axis=1)
        # if all columns are marked as day off => ignore
        for series_id in off[off==7].index:
            res.loc[series_id, columns] = False
        # if 5 days are marked as day off
        for series_id in off[off==5].index:
            res.loc[series_id, columns] = ~g.loc[series_id, columns]
    return res


def gen_daily_train_test(train_test):
    res = train_test.copy()
    res['timestamp'] = pd.to_datetime(res['timestamp'].dt.date)
    assert all(res.groupby(['series_id', 'timestamp'])['entry_type'].nunique() == 1)

    def p125(x):
        if all(pd.isnull(x)):
            return None
        return np.percentile(x, 12.5)

    def p875(x):
        if all(pd.isnull(x)):
            return None
        return np.percentile(x, 87.5)

    res = res.groupby(['series_id', 'timestamp'], as_index=False).agg({
        'submission_timestamp': 'first',
        'entry_type': 'first',
        'temperature': [np.mean, np.min, np.max, np.std, p125, p875, 'count'],
        'consumption': [np.mean, np.min, np.max, np.std, p125, p875, 'count', np.sum],
    })
    res.columns = ['_'.join(col).strip('_') for col in res.columns.values]

    res = res.rename(columns={
        'temperature_amax': 'temperature_max',
        'temperature_amin': 'temperature_min',
        'consumption_sum': 'consumption',
        'consumption_amin': 'consumption_min',
        'consumption_amax': 'consumption_max',
        'entry_type_first': 'entry_type',
    })
    res['has_some_temperature'] = res['temperature_count'] > 0
    res['has_full_temperature'] = res['temperature_count'] == 24
    train_or_cold = res['entry_type'].isin(['train', 'cold_start'])
    res['is_invalid_consumption'] = (res['consumption_min'] == res['consumption_max']) & \
        train_or_cold
    eps = 0.1
    res['has_zero_consumption'] = (res['consumption_min'] < eps) & train_or_cold
    res['consumption_series_max'] = res.groupby('series_id')['consumption_max'].transform(np.max)
    res['is_low_consumption_day'] = \
        ((res['consumption_p875'] / res['consumption_p125']) < 1.4) \
        & ((res['consumption_series_max'] / res['consumption_p875']) > 1.2) \
        & train_or_cold
    sel = res[res.entry_type == 'test']
    res.loc[sel.index, 'consumption'] = None
    return res


def gen_series_data(df):
    all_series = sorted(df.series_id.unique())
    res = pd.DataFrame(index=all_series)
    count_column = lambda x: x.groupby('series_id').size().reindex(all_series).fillna(0).astype('int32')
    res['is_train'] = count_column(df[df.entry_type == 'train']) > 0
    sel = df[df.entry_type.isin(['train', 'cold_start'])]
    res['data_days'] = count_column(sel)
    res['data_from_days_off'] = count_column(sel[sel.is_day_off])
    clean_sel = sel[~sel.is_invalid_consumption & ~sel.has_zero_consumption]
    res['clean_data_days'] = count_column(clean_sel)
    res['clean_data_from_days_off'] = count_column(clean_sel[clean_sel.is_day_off])
    res['is_clean'] = (res['data_days'] > 0) & (res['data_days'] == res['clean_data_days'])

    res = res.join(
        df.groupby('series_id').agg({
            'has_some_temperature': max,
            'has_full_temperature': min,
            'temperature_max': max,
            'temperature_min': min,
            'consumption_max': max,
            'consumption_min': min,
            'surface_id': 'first',
            'base_temperature_id': 'first',
            'series_start_timestamp': 'first',
            'series_start_dayofweek': 'first',
            'series_start_dayofyear': 'first'
        })
    )
    
    return res


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)

    logger.info("loading simple datasets from %s", input_filepath)
    res = []
    for filename, entry_type in [
        ('consumption_train.csv', 'train'),
        ('cold_start_test.csv', 'cold_start'),
        ('submission_format.csv', 'test'),
    ]:
        logging.info("loading %s", filename)
        df = load_consumption_data(os.path.join(input_filepath, filename), entry_type)
        res.append(df)

    submission, test = load_submission(os.path.join(input_filepath, "submission_format.csv"))
    meta = load_meta(os.path.join(input_filepath, "meta.csv"))
    meta_org = load_meta(os.path.join(input_filepath, "meta.csv"), fix_errors=False)

    train_test = pd.concat(res, axis=0, sort=False, ignore_index=True)
    train_test_d = gen_daily_train_test(train_test)

    # add features
    train_test = calc_interim_features(train_test, meta, meta_org=meta_org, mode='hourly')
    train_test_d = calc_interim_features(train_test_d, meta, meta_org=meta_org, mode='daily')

    series = gen_series_data(train_test_d)

    import warnings
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    logger.info("writing output to %s", output_filepath)
    train_test.to_hdf(output_filepath, "train_test", mode="w")
    train_test_d.to_hdf(output_filepath, "train_test_d", mode="a")
    series.to_hdf(output_filepath, "series", mode="a")
    submission.to_hdf(output_filepath, "submission", mode="a")
    test.to_hdf(output_filepath, "test", mode="a")
    meta.to_hdf(output_filepath, "meta", mode="a")
    meta_org.to_hdf(output_filepath, "meta_org", mode="a")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

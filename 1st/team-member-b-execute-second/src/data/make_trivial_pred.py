#!/usr/bin/env python
import os
import re
import logging
import click
import pandas as pd
import numpy as np


def load_submission(full_fn):
    df = pd.read_csv(full_fn).set_index('pred_id')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def save_submission(s, full_fn=None):
    if full_fn is None:
        t = pd.Timestamp.now().strftime("%Y%m%d-%H%M")
        full_fn = f'/tmp/dd-submission-{t}.csv'
    cols = ['series_id', 'timestamp', 'temperature', 'consumption', 'prediction_window']
    logging.info("writing %s", full_fn)
    s[cols].to_csv(full_fn, index=True, index_label='pred_id')


def manual_fix(base_s, train_test, series_id, day=1, hours=24, values=None):
    if values is None:
        sel_values = train_test[(train_test.series_id==series_id)
            & (train_test.entry_type=='cold_start')]\
            ['consumption'].tail(day*24).head(24).values
        if hours != 24:
            sel_values = sel_values[-hours:]
    else:
        sel_values = np.array(values)
    sel_values = np.tile(sel_values, 24//len(sel_values))
    fix = base_s[base_s.series_id==series_id].copy()
    assert len(fix)==len(sel_values)
    fix['consumption'] = sel_values
    return fix
    

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='data/interim/train_test.hdf5')
@click.option('-o', '--output', type=click.Path())
def main(input_filepath, output):
    train_test = pd.read_hdf(input_filepath, "train_test")

    base_submission = load_submission("data/raw/submission_format.csv")
    fixes = [
        manual_fix(base_submission, train_test, series_id=100492, day=1, hours=3),
        manual_fix(base_submission, train_test, series_id=101844, day=4),
        manual_fix(base_submission, train_test, series_id=102356, day=3),
        manual_fix(base_submission, train_test, series_id=102577, day=4),
        manual_fix(base_submission, train_test, series_id=103336, day=1, hours=3) 
    ]
    res = pd.concat(fixes, axis=0, sort=True)
    save_submission(res, full_fn=output)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

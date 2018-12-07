#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
    assert all(s['consumption'].notnull())
    s[cols].to_csv(full_fn, index=True, index_label='pred_id')


def overwrite_blend(base_submission, fix_submission):
    res = base_submission.copy()
    fix = fix_submission[fix_submission.index.isin(base_submission.index)]
    res.loc[fix.index, 'consumption'] = fix['consumption']
    return res


def mean_blend_s(s1, s2):
    assert all(s1.index == s2.index)
    res = s1.join(s2.groupby('pred_id')['consumption'].first().rename("new"), on='pred_id', how='left')
    assert res['new'].isnull().sum() == 0
    res['consumption'] = res[['consumption', 'new']].mean(axis=1)
    del res['new']
    return res


@click.command()
@click.option('--input-dir', type=click.Path(exists=True))
@click.option('--input-file', multiple=True, type=click.Path(exists=True))
@click.option('--patch-file', type=click.Path(exists=True))
@click.option('-o', '--output', type=str, default=None, required=True)
def main(input_dir, input_file, patch_file, output):
    base_submission = load_submission("data/raw/submission_format.csv")

    submissions = []
    if input_dir is not None:
        for fn in sorted(os.listdir(input_dir)):
            full_fn = os.path.join(input_dir, fn)
            if re.match(r'^(dd-)?submission.*\.csv$', fn):
                logging.info("loading %s", full_fn)
                submissions.append(load_submission(full_fn))
    if len(input_file) > 0:
        for full_fn in input_file:
            logging.info("loading %s", full_fn)
            submissions.append(load_submission(full_fn))

    s = pd.concat([
        submission['consumption'].rename(i)
        for i, submission in enumerate(submissions)
    ], axis=1).mean(axis=1, skipna=True).rename('consumption').to_frame()
    s['pred_id'] = s.index
    assert(s['consumption'].isnull().sum() == 0)

    res = base_submission.reindex(s.index)
    res = overwrite_blend(res, s)

    if patch_file is not None:
        logging.info("applying patch %s", patch_file)
        patch_submission = load_submission(patch_file)
        res = overwrite_blend(res, patch_submission)

    save_submission(res, output)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

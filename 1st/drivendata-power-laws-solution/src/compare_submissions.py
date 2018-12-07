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

@click.command()
@click.argument('file1', type=click.Path(exists=True), required=True)
@click.argument('file2', type=click.Path(exists=True), required=True)
def main(file1, file2):
    base_submission = load_submission("data/raw/submission_format.csv")

    s1 = load_submission(file1).sort_index()
    s2 = load_submission(file2).sort_index()

    assert all(s1.index==s2.index)
    # assert all(s1.index==base_submission.index)
    # assert all(s2.index==base_submission.index)

    df = pd.concat([
        s1['series_id'],
        s1['timestamp'],
        s1['prediction_window'],
        s1['consumption'].round(6).rename("c1"),
        s2['consumption'].round(6).rename("c2"),
    ], axis=1)
    assert all(df['c1'].notnull())
    assert all(df['c2'].notnull())

    df['diff'] = (df['c1'] - df['c2']).abs()
    df['mean'] = df.groupby('series_id')['c1'].transform(np.mean)
    df['diff_pr'] = df['diff'] / df['mean']

    print("comparing {} vs {} (both of length {})".format(file1, file2, len(df)))
    print(df.groupby('prediction_window').agg({'diff_pr': [np.sum, np.mean]}))

    g = df[df['diff'] > 0].groupby(['series_id', 'prediction_window'], as_index=False)['diff_pr'].mean()
    g = g.sort_values(by=['diff_pr'], ascending=False)
    if len(g) > 0:
        print(g.head(10))



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

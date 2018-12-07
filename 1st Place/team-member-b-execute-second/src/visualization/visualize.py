#!/usr/bin/env python
import os
import re
import logging
import click
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", font_scale=1.5)

def savefig(filename):
    logging.info("writing %s", filename)
    plt.savefig(filename, dpi=150, bbox_inches='tight', transparent=True)

def fig_dayofweek_dist(train_test, submission, output_directory):
    sel_train = train_test[
        (train_test.entry_type.isin(['train', 'cold_start']))
        & (train_test.cold_start_days >= 7)
    ]
    sel_test_s = submission[submission.prediction_window=='hourly'].series_id
    sel_test = train_test[
        (train_test.entry_type.isin(['test']))
        & (train_test.series_id.isin(sel_test_s))
        & (train_test.cold_start_days >= 7)
    ]
    f = lambda df: df.groupby(df.date.dt.dayofweek).size() * 100 / len(df)

    fig, axs = plt.subplots(figsize=(12, 4), ncols=2, sharey=True)
    f(sel_train).plot(kind='bar', color=sns.xkcd_rgb['denim blue'], ax=axs[0], rot=0)
    f(sel_test).plot(kind='bar', color=sns.xkcd_rgb['pale red'], ax=axs[1], rot=0)

    for ax in axs:
        sns.despine(ax=ax)
        ax.set_ylabel("% of samples")
        ax.set_xlabel("day of week")
    axs[0].set_title("Distribution in training set")
    axs[1].set_title("Distribution in test set")

    savefig(os.path.join(output_directory, "fig_dayofweek_dist.png"))


def fig_consumption_over_time(train_test_d, output_directory):
    fig, ax = plt.subplots(figsize=(15, 5))
    sel = train_test_d[train_test_d.entry_type.isin(['train', 'cold_start'])]
    for e in ['train', 'cold_start']:
        s = sel[sel.entry_type==e]
        c = {"train": sns.xkcd_rgb['denim blue'], "cold_start": sns.xkcd_rgb['pale red']}.get(e)
        ax.plot(s.date, s.consumption, 'o', ms=3, color=c, alpha=0.2)
        sns.despine(ax=ax)

    ax.set_yscale("log", nonposy='clip')
    ax.set_title("Consumption in train and cold start data")
    ax.legend(['train', 'cold start'], bbox_to_anchor=(0, 1), loc='upper left')

    savefig(os.path.join(output_directory, "fig_consumption_over_time.png"))


def fig_temperature_over_time(train_test_d, output_directory):
    fig, ax = plt.subplots(figsize=(15, 5))
    sel = train_test_d[train_test_d.entry_type.isin(['train', 'cold_start'])]
    for e in ['train', 'cold_start']:
        s = sel[sel.entry_type==e]
        c = {"train": sns.xkcd_rgb['denim blue'], "cold_start": sns.xkcd_rgb['pale red']}.get(e)
        ax.plot(s.date, s.temperature_mean, 'o', ms=3, color=c, alpha=0.2)
        sns.despine(ax=ax)

    g = sel.groupby('date', as_index=False)['temperature_mean'].mean()
    g['temperature_mean2'] = g['temperature_mean'].ewm(10).mean()
    ax.plot(g.date, g.temperature_mean2, '-', color=sns.xkcd_rgb['muted green'], linewidth=3)
    ax.set_title("Temperature in train and cold start data")
    ax.legend(['train', 'cold start'], bbox_to_anchor=(0, 1), loc='upper left')

    savefig(os.path.join(output_directory, "fig_temperature_over_time.png"))


def fig_series_consumption(train_test, series_id, output_directory, label):
    g = train_test[train_test.series_id==series_id].groupby('timestamp')['consumption'].first()
    fig, ax = plt.subplots(figsize=(15, 5))
    g.plot(ax=ax, kind='line', linewidth=2)
    ax.set_ylim(0, None)
    ax.set_xlabel("")
    ax.set_ylabel("consumption")
    ax.yaxis.grid(True, alpha=0.4)
    sns.despine(ax=ax)
    savefig(os.path.join(output_directory, f"fig_consumption_{label}.png"))

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='data/processed/train_test.hdf5')
@click.argument('interim_input_filepath', type=click.Path(exists=True), default='data/interim/train_test.hdf5')
@click.option('-o', '--output-directory', type=click.Path(), default='reports/figures')
def main(input_filepath, interim_input_filepath, output_directory):
    logging.info("reading %s", input_filepath)
    train_test = pd.concat([
        pd.read_hdf(input_filepath, "train"),
        pd.read_hdf(input_filepath, "validate"),
        pd.read_hdf(input_filepath, "test"),
    ], axis=0, sort=False, ignore_index=True)
    submission = pd.read_hdf(input_filepath, "submission")

    logging.info("reading %s", interim_input_filepath)
    interim_train_test = pd.read_hdf(interim_input_filepath, "train_test")
    interim_train_test_d = pd.read_hdf(interim_input_filepath, "train_test_d")

    fig_series_consumption(interim_train_test, 100111, output_directory, "missing_data")
    fig_series_consumption(interim_train_test, 102164, output_directory, "regular")
    fig_series_consumption(interim_train_test, 102772, output_directory, "random")
    fig_consumption_over_time(interim_train_test_d, output_directory)
    fig_temperature_over_time(interim_train_test_d, output_directory)
    fig_dayofweek_dist(train_test, submission, output_directory)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

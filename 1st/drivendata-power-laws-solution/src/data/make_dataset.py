# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.cross_validation import train_test_split
#
from src.features.build_features import calc_final_features


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logging.info("reading %s", input_filepath)
    train_test = pd.read_hdf(input_filepath, 'train_test')
    meta = pd.read_hdf(input_filepath, 'meta')
    meta_org = pd.read_hdf(input_filepath, 'meta_org')

    sel_series = train_test[train_test.entry_type.isin(['train', 'cold_start'])]\
        ['series_id'].unique()
    train_series, validate_series = train_test_split(sel_series, random_state=1)

    logging.info("calc train_test")
    train_test = calc_final_features(train_test, meta, meta_org=meta_org, verbose=True)

    sel = train_test[train_test.entry_type.isin(['train', 'cold_start'])]
    train = sel[sel.series_id.isin(train_series)]
    validate = sel[sel.series_id.isin(validate_series)]
    test = train_test[train_test.entry_type.isin(['test'])]

    logging.info("writing %s", output_filepath)
    train.to_hdf(output_filepath, "train", mode="w")
    validate.to_hdf(output_filepath, "validate", mode="a")
    test.to_hdf(output_filepath, "test", mode="a")
    for k in ['meta', 'submission']:
        df = pd.read_hdf(input_filepath, k)
        df.to_hdf(output_filepath, k, mode="a")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    import warnings
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

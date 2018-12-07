"""
Functions used during the train
"""
import pandas as pd
import numpy as np

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger

from coldstart.validation import split_series_id, stratified_cv_series_id
from coldstart.keras.train import _get_callbacks
from coldstart.seq2seq.data import prepare_data_for_train
from coldstart.utils import load_data
from coldstart.lstm.train import fit

def load_and_arrange_data(conf, verbose=False):
    train, test, _, metadata = load_data()

    if 'random_seed' in conf:
        train_ids, val_ids = stratified_cv_series_id(
            train.series_id.unique(), fold_idx=conf['fold_idx'], random_seed=conf['random_seed'])
    else:
        train_ids, val_ids = split_series_id(train.series_id.unique(), fold_idx=conf['fold_idx'])
    val = train[train.series_id.isin(val_ids)]
    val.reset_index(inplace=True, drop=True)
    train = train[train.series_id.isin(train_ids)]
    train.reset_index(inplace=True, drop=True)

    train = pd.concat([train, test])
    train.reset_index(inplace=True, drop=True)

    train_x, train_y = prepare_data_for_train(
        train, metadata, conf['input_days'], conf['window'], conf['only_working_day'], verbose=verbose)
    val_x, val_y = prepare_data_for_train(
        val, metadata, conf['input_days'], conf['window'], conf['only_working_day'], verbose=verbose)

    train_x = np.concatenate([train_x[key] for key in conf['inputs']], axis=2)
    val_x = np.concatenate([val_x[key] for key in conf['inputs']], axis=2)
    return train_x, train_y, val_x, val_y

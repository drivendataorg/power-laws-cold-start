"""
Architecture exploration
"""
import os
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

from coldstart.definitions import DATASET_PATH
from coldstart.utils import get_timestamp
from coldstart.keras.train_manager import train_manager, simple_train_manager

def _create_layers(n_units):
    if isinstance(n_units, int) or isinstance(n_units, np.int64):
        n_units = [n_units]
    return [{'layer': 'Dense', 'units': int(units), 'activation': 'relu'} for units in n_units]

RETRAIN_NAMES = [
    '2018_10_13_11_21_01', '2018_10_13_23_49_39', '2018_10_13_21_12_54', '2018_10_12_19_21_59',
    '2018_10_12_16_39_03', '2018_10_14_15_03_51', '2018_10_14_08_45_19', '2018_10_14_06_10_59',
    '2018_10_13_14_34_02', '2018_10_12_07_07_24']

def _load_model_conf(name):
    model_conf_path = os.path.join(
        DATASET_PATH, 'models', '2018_10_11_architecture_exploration', '%s_cv0' % name, 'daily_1.json')
    with open(model_conf_path, 'r') as f:
        info = json.load(f)
    return info['model_conf']

def main():
    model_conf = {
        'encoding': {
            'is_day_off': [{'layer': 'Dense', 'units': 32, 'activation': 'relu'}],
            'metadata_ohe': [{'layer': 'Dense', 'units': 8, 'activation': 'relu'}],
            'data_trend': [{'layer': 'Dense', 'units': 16, 'activation': 'relu'}],
            'metadata_days_off': [{'layer': 'Dense', 'units': 8, 'activation': 'relu'}],
        },
        'weights': [{'layer': 'Dense', 'units': 32, 'activation': 'relu'},
                    {'layer': 'Dense', 'units': 16, 'activation': 'relu'}],
        'repeat_weights': False,
    }
    train_conf = {
        'optimizer_kwargs': {'lr': 1e-3, 'clipvalue': 10},
        'train_kwargs': dict(batch_size=8, epochs=5000, verbose=0),
        'callbacks': {
            'EarlyStopping': {
                'patience': 25, 'mode': 'min', 'verbose': 0, 'monitor': 'val_loss',
                'min_delta': 0.0001},
            'ReduceLROnPlateau': {
                'patience': 0, 'factor': 0.95, 'mode': 'min', 'verbose': 0,
                'monitor': 'loss'},
            'ModelCheckpointRAM': {
                'mode': 'min', 'verbose': 0, 'monitor': 'val_loss'},
        },
    }
    conf = {
        'windows': ['hourly', 'daily', 'weekly'],
        'input_days': list(range(1, 8)),
        'fold_idxs': list(range(5)),
        'remove_inputs': ['temperature', 'temperature_normed', 'cluster_id_ohe'],
        'max_workers': 8,
        'verbose': False,
        'train_conf': train_conf,
        'model_conf': model_conf,
    }
    n_models_trained = 0
    for name in RETRAIN_NAMES:
        model_conf = _load_model_conf(name)
        model_conf['encoding']['cluster_features_v2'] = \
            [{'layer': 'Dense', 'units': 4, 'activation': 'relu'}]
        conf['model_conf'] = model_conf
        print(conf['model_conf'])
        print('Number of models trained: %i' % n_models_trained)
        n_models_trained += 1
        conf['models_dir'] = os.path.join(
            DATASET_PATH, 'models', '2018_10_17_architecture_exploration_cluster', name)
        print(conf['models_dir'])
        simple_train_manager(conf)

main()
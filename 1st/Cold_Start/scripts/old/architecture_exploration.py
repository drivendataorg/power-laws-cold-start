"""
Architecture exploration
"""
import os
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

    name = get_timestamp()
    print(name)
    conf = {
        'windows': ['hourly', 'daily', 'weekly'],
        'input_days': list(range(1, 8)),
        'fold_idxs': list(range(5)),
        'remove_inputs': ['temperature', 'temperature_normed', 'cluster_id_ohe'],
        'max_workers': 8,
        'verbose': False,
        'models_dir': os.path.join(
            DATASET_PATH, 'models', '2018_10_11_architecture_exploration', name),
        'train_conf': train_conf,
        'model_conf': model_conf,
    }
    simple_train_manager(conf)

    n_models_trained = 1
    while 1:
        model_conf = {
            'encoding': {
                'is_day_off': _create_layers(np.random.choice([8, 16, 32, 64])),
                'metadata_ohe': _create_layers(np.random.choice([4, 8, 16])),
                'data_trend': _create_layers(np.random.choice([8, 16, 32])),
                'metadata_days_off': _create_layers(np.random.choice([4, 8, 16])),
            },
            'weights': _create_layers(
                np.random.choice([8, 16, 32, 64], size=np.random.randint(1, 4))),
            'repeat_weights': False,
        }
        conf['model_conf'] = model_conf
        name = get_timestamp()
        print(name)
        print('Number of models trained: %i' % n_models_trained)
        n_models_trained += 1
        conf['models_dir'] = os.path.join(
            DATASET_PATH, 'models', '2018_10_11_architecture_exploration', name)
        print(conf['models_dir'])
        simple_train_manager(conf)

main()
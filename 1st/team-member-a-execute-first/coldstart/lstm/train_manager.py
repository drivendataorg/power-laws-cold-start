"""
Utils for training LSTM on parallel on different gpus
"""
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

from coldstart.keras.train_manager import _set_session, _print_train_summary
from coldstart.lstm.train import fit, load_and_arrange_data
from coldstart.lstm.model import create_model

class TrainManager(object):
    """
    Class for making easier to run different trains on parallel
    """
    def __init__(self, n_workers):
        self._pool = None
        self._submits = []
        self._create_pool(n_workers)

    def _create_pool(self, n_workers):
        """
        Creates the pool of workers and sets the session distributing the load
        between the two gpus giving preference to gpu 1
        """
        pool = ProcessPoolExecutor(max_workers=n_workers)
        gpus = [str(1-i%2) for i in range(n_workers)]
        submits = [pool.submit(_set_session, gpu) for gpu in gpus]
        self._submits += submits
        while not all([submit.done() for submit in submits]):
            time.sleep(1)
        self._pool = pool

    def submit(self, func, *args, **kwargs):
        """ Submits a single job to the pool """
        submit = self._pool.submit(func, *args, **kwargs)
        self._submits.append(submit)
        return submit

    def get_remaining_submits(self):
        self._submits = [submit for submit in self._submits if not submit.done()]
        return len(self._submits)


def train_single_model(conf):
    """
    Trains a single model using the given data and configuration.
    Saves the model, history and configuration to file
    """
    train_x, train_y, val_x, val_y = load_and_arrange_data(conf)
    if not os.path.exists(conf['models_dir']):
        os.makedirs(conf['models_dir'], exist_ok=True)

    model = create_model(train_x, conf['model_conf'])
    if conf['verbose']:
        model.summary()
    model_path = os.path.join(conf['models_dir'], '%s_%i.h5' % (conf['window'], conf['input_days']))
    conf['train_conf']['model_path'] = model_path
    conf_path = model_path.replace('.h5', '.json')
    with open(conf_path, 'w') as f:
        json.dump(conf, f)

    model, ret = fit(model, train_x, train_y, [val_x, val_y], conf['train_conf'])
    _print_train_summary(conf, ret.history)

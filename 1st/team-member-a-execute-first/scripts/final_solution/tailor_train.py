"""
Training for tailor made NN models
"""
import os
import glob
import json
import numpy as np
from tqdm import tqdm

from coldstart.definitions import DATASET_PATH, WINDOW_TO_PRED_DAYS
from coldstart.utils import get_timestamp
from coldstart.frankenstein.train_manager import TrainManager
from coldstart.keras.train_manager import train_single_model

def main_parallel():
    """
    This will train 8 models in parallel using 2 gpus.
    """
    train_manager = TrainManager(8)
    name = get_timestamp()
    random_seed = np.random.randint(1, int(1e5))
    for fold_idx in range(5):
        for window in WINDOW_TO_PRED_DAYS:
            for input_days in range(1, 8):
                conf = _get_random_conf(window, input_days)
                conf['window'] = window
                conf['input_days'] = input_days
                conf['fold_idx'] = fold_idx
                conf['random_seed'] = random_seed
                conf['verbose'] = False
                conf['train_conf']['train_kwargs']['verbose'] = 0
                conf['models_dir'] = os.path.join(
                    DATASET_PATH, 'models', 'tailor', '%s_cv%i' % (
                        name, fold_idx))
                train_manager.submit(train_single_model, conf)

def main_sequential():
    """
    This will train one model each time (not in parallel)
    """
    progress_bar = tqdm(desc='Training tailor', total=(5*3*7))
    name = get_timestamp()
    random_seed = np.random.randint(1, int(1e5))
    for fold_idx in range(5):
        for window in WINDOW_TO_PRED_DAYS:
            for input_days in range(1, 8):
                conf = _get_random_conf(window, input_days)
                conf['window'] = window
                conf['input_days'] = input_days
                conf['fold_idx'] = fold_idx
                conf['random_seed'] = random_seed
                conf['verbose'] = False
                conf['train_conf']['train_kwargs']['verbose'] = 0
                conf['models_dir'] = os.path.join(
                    DATASET_PATH, 'models', 'tailor', '%s_cv%i' % (
                        name, fold_idx))
                train_single_model(conf)
                progress_bar.update()

def _get_random_conf(window, input_days):
    conf_paths = glob.glob(os.path.join('tailor_confs', window, str(input_days), '*.json'))
    conf_path = np.random.choice(conf_paths)
    with open(conf_path, 'r') as f:
        conf = json.load(f)
    return conf

if __name__ == '__main__':
    main_sequential()

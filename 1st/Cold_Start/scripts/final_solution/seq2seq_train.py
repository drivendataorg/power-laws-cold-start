"""
Training for frankenstein models
"""
import os
import json
import numpy as np
import glob
from tqdm import tqdm

from coldstart.definitions import DATASET_PATH
from coldstart.utils import get_timestamp
from coldstart.seq2seq.train_manager import train_single_model, TrainManager

def main_parallel():
    """
    This will train 4 models in parallel using 2 gpus.
    """
    train_manager = TrainManager(4)
    name = get_timestamp()
    random_seed = np.random.randint(1, int(1e5))
    window = 'hourly'
    for fold_idx in range(5):
        for is_working in range(2):
            for input_days in range(1, 8):
                conf = _get_random_conf(input_days, is_working)
                conf['window'] = window
                conf['input_days'] = input_days
                conf['only_working_day'] = is_working
                conf['fold_idx'] = fold_idx
                conf['random_seed'] = random_seed
                conf['models_dir'] = os.path.join(
                    DATASET_PATH, 'models', 'seq2seq', '%s_cv%i' % (
                        name, fold_idx))
                train_manager.submit(train_single_model, conf)

def main_sequential():
    """
    This will train one model each time (not in parallel)
    """
    progress_bar = tqdm(desc='Training seq2seq', total=(5*2*7))
    name = get_timestamp()
    random_seed = np.random.randint(1, int(1e5))
    window = 'hourly'
    for fold_idx in range(5):
        for is_working in range(2):
            for input_days in range(1, 8):
                conf = _get_random_conf(input_days, is_working)
                conf['window'] = window
                conf['input_days'] = input_days
                conf['only_working_day'] = is_working
                conf['fold_idx'] = fold_idx
                conf['random_seed'] = random_seed
                conf['models_dir'] = os.path.join(
                    DATASET_PATH, 'models', 'seq2seq', '%s_cv%i' % (
                        name, fold_idx))
                train_single_model(conf)
                progress_bar.update()

def _get_random_conf(input_days, is_working):
    conf_paths = glob.glob(os.path.join('seq2seq_confs', str(input_days), str(is_working), '*.json'))
    conf_path = np.random.choice(conf_paths)
    with open(conf_path, 'r') as f:
        conf = json.load(f)
    return conf

if __name__ == '__main__':
    main_sequential()

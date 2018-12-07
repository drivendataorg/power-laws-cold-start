"""
Training for frankenstein models
"""
import os
import json
import numpy as np

from coldstart.definitions import DATASET_PATH, WINDOW_TO_PRED_DAYS
from coldstart.utils import get_timestamp
from coldstart.frankenstein.train_manager import TrainManager
from coldstart.keras.train_manager import train_single_model

def main():
    with open('taylor.json', 'r') as f:
        conf_paths = json.load(f)
    train_manager = TrainManager(8)
    name = get_timestamp()
    random_seed = np.random.randint(1, int(1e5))
    for fold_idx in range(5):
        for window in WINDOW_TO_PRED_DAYS:
            for input_days in range(1, 8):
                conf_path = np.random.choice(conf_paths[window][str(input_days)])
                with open(conf_path.replace('.h5', '.json'), 'r') as f:
                    conf = json.load(f)
                conf['window'] = window
                conf['input_days'] = input_days
                conf['fold_idx'] = fold_idx
                conf['random_seed'] = random_seed
                conf['verbose'] = False
                conf['train_conf']['train_kwargs']['verbose'] = 0
                conf['models_dir'] = os.path.join(
                    DATASET_PATH, 'models', '2018_10_29_taylor', '%s_cv%i' % (
                        name, fold_idx))
                train_manager.submit(train_single_model, conf)

if __name__ == '__main__':
    main()

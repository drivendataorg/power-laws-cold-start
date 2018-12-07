"""
Training for frankenstein models
"""
import os
import json
import numpy as np

from coldstart.definitions import DATASET_PATH
from coldstart.utils import get_timestamp
from coldstart.seq2seq.train_manager import train_single_model, TrainManager

def main():
    with open('seq2seq.json', 'r') as f:
        conf_paths = json.load(f)
    train_manager = TrainManager(4)
    name = get_timestamp()
    random_seed = np.random.randint(1, int(1e5))
    window = 'hourly'
    for fold_idx in range(5):
        for is_working in range(2):
            for input_days in range(1, 8):
                conf_path = np.random.choice(conf_paths[str(input_days)][str(is_working)])
                with open(conf_path.replace('.h5', '.json'), 'r') as f:
                    conf = json.load(f)
                conf['window'] = window
                conf['input_days'] = input_days
                conf['only_working_day'] = is_working
                conf['fold_idx'] = fold_idx
                conf['random_seed'] = random_seed
                # conf['verbose'] = False
                # conf['train_conf']['train_kwargs']['verbose'] = 0
                conf['models_dir'] = os.path.join(
                    DATASET_PATH, 'models', '2018_10_29_seq2seq', '%s_cv%i' % (
                        name, fold_idx))
                train_manager.submit(train_single_model, conf)

if __name__ == '__main__':
    main()

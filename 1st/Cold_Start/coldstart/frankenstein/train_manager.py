"""
Utils for training LSTM on parallel on different gpus
"""
import os
import json
import numpy as np

from coldstart.frankenstein.data import load_and_arrange_data
from coldstart.frankenstein.model import create_model, add_shapes_to_model_conf
from coldstart.lstm.train_manager import TrainManager, _print_train_summary
from coldstart.frankenstein.train import fit

def train_single_model(conf):
    """
    Trains a single model using the given data and configuration.
    Saves the model, history and configuration to file
    """
    train_x, train_y, val_x, val_y = load_and_arrange_data(conf, conf['verbose'])
    if not os.path.exists(conf['models_dir']):
        os.makedirs(conf['models_dir'], exist_ok=True)

    add_shapes_to_model_conf(train_x, conf['model_conf'])
    model = create_model(conf['model_conf'])
    if conf['verbose']:
        model.summary()
    model_path = os.path.join(conf['models_dir'], '%s_%i.h5' % (
        conf['window'], conf['input_days']))
    conf['train_conf']['model_path'] = model_path
    conf_path = model_path.replace('.h5', '.json')
    with open(conf_path, 'w') as f:
        json.dump(conf, f)

    model, ret = fit(model, train_x, train_y, [val_x, val_y], conf['train_conf'])
    _print_train_summary(conf, ret.history)

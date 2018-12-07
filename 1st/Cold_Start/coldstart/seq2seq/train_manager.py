"""
Utils for training LSTM on parallel on different gpus
"""
import os
import json
import numpy as np

from coldstart.seq2seq.train import fit, load_and_arrange_data
from coldstart.seq2seq.model import create_model
from coldstart.lstm.train_manager import TrainManager

def train_single_model(conf):
    """
    Trains a single model using the given data and configuration.
    Saves the model, history and configuration to file
    """
    train_x, train_y, val_x, val_y = load_and_arrange_data(conf, conf['verbose'])
    if not os.path.exists(conf['models_dir']):
        os.makedirs(conf['models_dir'], exist_ok=True)

    model = create_model(train_x, train_y, conf['model_conf'])
    if conf['verbose']:
        model.summary()
    model_path = os.path.join(conf['models_dir'], '%s_%i_working%i.h5' % (
        conf['window'], conf['input_days'], int(conf['only_working_day'])))
    conf['train_conf']['model_path'] = model_path
    conf_path = model_path.replace('.h5', '.json')
    with open(conf_path, 'w') as f:
        json.dump(conf, f)

    model, ret = fit(model, train_x, train_y, [val_x, val_y], conf['train_conf'])
    _print_train_summary(conf, ret.history)

def _print_train_summary(conf, history):
    print('%s input_days %i_%i\tcv: %i\tloss: %.4f\tval_loss: %.4f'% (
        conf['window'], conf['input_days'], conf['only_working_day'],
        conf['fold_idx'],
        np.min(history['loss']), np.min(history['val_loss'])))

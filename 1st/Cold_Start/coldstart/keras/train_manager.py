"""
Advanced train in parallel with intelligent selection of gpus
"""
import pandas as pd
import numpy as np
import os
import time
import json
import tensorflow as tf
from tqdm import tqdm_notebook, tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from keras.backend.tensorflow_backend import set_session
from joblib import Parallel, delayed

from coldstart.keras.data import prepare_data_for_train
from coldstart.keras.model import create_model
from coldstart.keras.train import fit
from coldstart.utils import load_data
from coldstart.validation import split_series_id, stratified_cv_series_id

def _set_session(gpu):
    """
    Function that allows to train multiple models on same gpu and also to
    select which gpu to use.

    Parameters
    ----------
    gpu : str
        A string with the number of the gpu: '0' or '1'
    """
    if gpu not in ['0', '1']:
        raise Exception('Unknown selected gpu: %s' % gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = gpu
    session = tf.Session(config=config)
    set_session(session)

def train_manager(conf):
    pool = ProcessPoolExecutor(max_workers=conf['max_workers'])
    submits = []
    gpu_use = {'0': 0, '1': 0}
    n_jobs = len(conf['windows'])*len(conf['input_days'])*len(conf['fold_idxs'])
    progress_bar = tqdm(total=n_jobs, desc='Training')

    for window in conf['windows']:
        for input_days in conf['input_days']:
            for fold_idx in conf['fold_idxs']:
                while len(submits) >= conf['max_workers']:
                    time.sleep(1)
                    for i, submit in enumerate(submits):
                        if submit.done():
                            gpu_use[submit.result()] -= 1
                            submits.pop(i)
                            progress_bar.update(1)
                            break
                new_conf = conf.copy()
                new_conf['window'] = window
                new_conf['input_days'] = input_days
                new_conf['fold_idx'] = fold_idx
                new_conf['gpu'] = _select_gpu(gpu_use)
                new_conf['models_dir'] += '_cv%i' % fold_idx
                if not os.path.exists(new_conf['models_dir']):
                    os.makedirs(new_conf['models_dir'])
                gpu_use[new_conf['gpu']] += 1
                submits.append(pool.submit(train_single_model, new_conf.copy()))

    while submits:
        time.sleep(1)
        for i, submit in enumerate(submits):
            if submit.done():
                progress_bar.update(1)
                submits.pop(i)
                break

def simple_train_manager(conf):
    gpu_use = {'0': 0, '1': 0}
    all_confs = []
    for window in conf['windows']:
        for input_days in conf['input_days']:
            for fold_idx in conf['fold_idxs']:
                new_conf = conf.copy()
                new_conf['window'] = window
                new_conf['input_days'] = input_days
                new_conf['fold_idx'] = fold_idx
                new_conf['gpu'] = _select_gpu(gpu_use)
                new_conf['models_dir'] += '_cv%i' % fold_idx
                if not os.path.exists(new_conf['models_dir']):
                    os.makedirs(new_conf['models_dir'])
                gpu_use[new_conf['gpu']] += 1
                all_confs.append(new_conf)

    pool_0 = ProcessPoolExecutor(max_workers=conf['max_workers']//2)
    pool_1 = ProcessPoolExecutor(max_workers=conf['max_workers']//2)
    n_jobs = len(conf['windows'])*len(conf['input_days'])*len(conf['fold_idxs'])
    progress_bar = tqdm(total=n_jobs, desc='Training')
    submits = []
    for new_conf in all_confs:
        if new_conf['gpu'] == '0':
            submits.append(pool_0.submit(train_single_model, new_conf.copy()))
        else:
            submits.append(pool_1.submit(train_single_model, new_conf.copy()))
    while submits:
        time.sleep(1)
        for i, submit in enumerate(submits):
            if submit.done():
                progress_bar.update(1)
                submits.pop(i)
                break
    progress_bar.close()
    pool_0.shutdown()
    pool_1.shutdown()

    # Parallel(n_jobs=conf['max_workers']//2)(
    #     delayed(train_single_model)(new_conf) for new_conf in all_confs if new_conf['gpu'] == '0')
    # Parallel(n_jobs=conf['max_workers']//2)(
    #     delayed(train_single_model)(new_conf) for new_conf in all_confs if new_conf['gpu'] == '1')

def _select_gpu(gpu_use):
    if gpu_use['0'] < gpu_use['1']:
        return '0'
    else:
        return '1'

def train_single_model(conf):
    """
    Trains a single model using the given data and configuration.
    Saves the model, history and configuration to file

    Returns
    -------
    gpu : str
        The string with the gpu used for the training
    """
    train_x, train_y, val_x, val_y = _load_and_arrange_data(conf)

    model = create_model(train_x, conf['model_conf'])
    if conf['verbose']:
        model.summary()
    model, ret = fit(model, train_x, train_y, [val_x, val_y], conf['train_conf'])
    _print_train_summary(conf, ret.history)

    if not os.path.exists(conf['models_dir']):
        os.makedirs(conf['models_dir'])
    model_path = os.path.join(conf['models_dir'], '%s_%i.h5' % (conf['window'], conf['input_days']))
    model.save(model_path)
    history_path = os.path.join(conf['models_dir'], '%s_%i.csv' % (conf['window'], conf['input_days']))
    df = pd.DataFrame(ret.history)
    df.to_csv(history_path, index=False)
    conf_path = os.path.join(conf['models_dir'], '%s_%i.json' % (conf['window'], conf['input_days']))
    with open(conf_path, 'w') as f:
        json.dump(conf, f)

def _load_and_arrange_data(conf):
    train, test, submission, metadata = load_data()

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
        train, metadata, conf['input_days'], conf['window'], verbose=False)
    val_x, val_y = prepare_data_for_train(
        val, metadata, conf['input_days'], conf['window'], verbose=False)
    train_x = {key:train_x[key] for key in train_x if key not in conf['remove_inputs']}
    val_x = {key:val_x[key] for key in val_x if key not in conf['remove_inputs']}
    return train_x, train_y, val_x, val_y

def _print_train_summary(conf, history):
    print('%s input_days %i\tcv: %i\tloss: %.4f\tval_loss: %.4f'% (
        conf['window'], conf['input_days'], conf['fold_idx'],
        np.min(history['loss']), np.min(history['val_loss'])))

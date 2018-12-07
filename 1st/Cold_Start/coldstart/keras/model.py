"""
Model definition
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Dense, Input, Concatenate, RepeatVector, Multiply, Add, Lambda
from keras.layers import LSTM, Dropout, BatchNormalization, AveragePooling1D, Reshape
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model
import keras.backend as K

from coldstart.definitions import WINDOW_TO_PRED_DAYS
from coldstart.keras.losses import nmae

def create_model(x, conf):
    """
    Given the dictionary with the input to the model a keras
    model is build. It is mandatory to have past_consumption as input.

    The structure will be the following:
        1. mlp for encoding each input except past_consumption
        2. merge of encodings
        3. mlp on top of encodings
        4. predict weights for prediction
        5. weighted average of past consumption
    """
    input_layers = _create_input_layers(x)
    encodings = _get_encodings(input_layers, conf['encoding'])
    merged_encodings = _merge_encodings(encodings)
    weights_features = _add_layers(merged_encodings, conf['weights'])
    weights = _get_weights_for_prediction(weights_features, x, conf['repeat_weights'])
    output = Multiply()([weights, input_layers['past_consumption']])
    output = Lambda(lambda x: K.sum(x, axis=2))(output)

    model = Model(list(input_layers.values()), output)
    return model

def _create_input_layers(x):
    input_layers = {}
    for key in x:
        input_layers[key] = Input(shape=(x[key].shape[1:]), name=key)
    return input_layers

def _get_encodings(input_layers, encoding_conf):
    encodings = []
    for key in input_layers:
        if key == 'past_consumption':
            continue
        if key in encoding_conf:
            encodings.append(_add_layers(input_layers[key], encoding_conf[key]))
        else:
            encodings.append(input_layers[key])
    return encodings

def _merge_encodings(encodings):
    if len(encodings) > 1:
        return Concatenate()(encodings)
    elif len(encodings) == 1:
        return encodings[0]
    else:
        raise Exception('No encoding found')

STR_TO_LAYER = {
    'Dense': Dense,
    'LSTM': LSTM,
    'Dropout': Dropout,
    'BatchNormalization': BatchNormalization,
}

def _add_layers(output, params):
    for layer_conf in params:
        layer_conf = layer_conf.copy()
        layer = STR_TO_LAYER[layer_conf.pop('layer')]
        output = layer(**layer_conf)(output)
    return output

def _get_weights_for_prediction(weights_features, x, repeat_weights):
    if repeat_weights:
        weights = Dense(x['past_consumption'].shape[2], activation='relu',
                        name='weights')(weights_features)
        weights = RepeatVector(x['past_consumption'].shape[1])(weights)
    else:
        weights = Dense(np.prod(x['past_consumption'].shape[1:]),
                        activation='relu', name='weights')(weights_features)
        weights = Reshape(x['past_consumption'].shape[1:])(weights)

    return weights

class MetaModel(object):
    def __init__(self):
        self.models = {key: {} for key in WINDOW_TO_PRED_DAYS}

    def predict(self, x, window):
        input_days = self._get_input_days_from_x(x)
        input_days = self._get_used_input_days(window, input_days)
        model = self.models[window][input_days]
        x = self._limit_x_to_model_inputs(x, model)
        x = self._limit_x_to_input_days(x, window, input_days)
        return model.predict(x)

    @staticmethod
    def _limit_x_to_input_days(x, window, input_days):
        pred_days = WINDOW_TO_PRED_DAYS[window]
        if 'past_consumption' in x:
            x['past_consumption'] = x['past_consumption'][:, :, -input_days:]
        if 'is_day_off' in x:
            x['is_day_off'] = x['is_day_off'][:, -input_days - pred_days:]
        if 'data_trend' in x:
            x['data_trend'] = x['data_trend'][:, -input_days:]
        return x

    @staticmethod
    def _limit_x_to_model_inputs(x, model):
        return {key: x[key] for key in model.input_names}

    def _get_used_input_days(self, window, input_days):
        """
        Verifies that there are is a model with the desired
        input of days, otherwise returns the closer model
        """
        models = self.models[window]
        if input_days in models:
            return input_days
        else:
            return max(models.keys())

    @staticmethod
    def _get_input_days_from_x(x):
        input_days = x['past_consumption'].shape[2]
        return input_days

def show_model(model):
    plot_model(model, show_shapes=True)
    img = mpimg.imread('model.png')
    plt.figure(figsize=(15, 15))
    plt.imshow(img, interpolation='none')
    plt.axis('off')
    plt.show()

def load_custom_model(model_path):
    return load_model(model_path, custom_objects={'nmae': nmae})

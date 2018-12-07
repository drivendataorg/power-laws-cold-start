"""
Model definition
"""
import json
from keras.models import Model
from keras.layers import Input, LSTM, Concatenate, RepeatVector

from coldstart.definitions import WINDOW_TO_PRED_DAYS
from coldstart.keras.model import _add_layers


def create_model(conf):
    """
    Creates a simple model using the given configuration
    """
    past_input = Input(shape=conf['past_features_shape'], name='past_features')
    _, state_h, state_c = LSTM(units=conf['LSTM_units'], return_state=True)(past_input)
    encoder_states = [state_h, state_c]

    future_input = Input(shape=conf['future_features_shape'], name='future_features')
    decoder_lstm = LSTM(conf['LSTM_units'], return_sequences=True, return_state=False)
    decoder_output = decoder_lstm(future_input, initial_state=encoder_states)

    cluster_input = Input(shape=conf['cluster_features_shape'], name='cluster_features')
    cluster_encoding = _add_layers(cluster_input, conf['cluster_encoding'])
    cluster_encoding = RepeatVector(conf['future_features_shape'][0])(cluster_encoding)

    top_input = Concatenate()([cluster_encoding, decoder_output])
    top = _add_layers(top_input, conf['top'])

    model = Model([past_input, future_input, cluster_input], [top])
    return model

def add_shapes_to_model_conf(x, model_conf):
    for key in x:
        model_conf['%s_shape' % key] = x[key].shape[1:]

def load_model(model_path):
    model_conf_path = model_path.replace('.h5', '.json')
    with open(model_conf_path, 'r') as f:
        model_conf = json.load(f)['model_conf']
    model = create_model(model_conf)
    model.load_weights(model_path)
    return model

class MetaModel(object):
    def __init__(self):
        self.models = {key: {} for key in WINDOW_TO_PRED_DAYS}

    def predict(self, x, window):
        input_days = self._get_input_days_from_x(x, window)
        model = self.models[window][input_days]
        return model.predict(x)

    @staticmethod
    def _get_input_days_from_x(x, window):
        input_days = x['past_features'].shape[1]
        if window == 'hourly':
            input_days = int(input_days/24)
        return input_days

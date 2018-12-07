"""
Model definition
"""
import json
from keras.models import Model
from seq2seq import Seq2Seq

from coldstart.keras.model import _add_layers
from coldstart.definitions import WINDOW_TO_PRED_DAYS


def create_model(x, y, conf):
    """
    Creates a simple model using the given configuration
    """
    conf['seq2seq']['input_dim'] = x.shape[2]
    conf['seq2seq']['input_length'] = x.shape[1]
    conf['seq2seq']['output_length'] = y.shape[1]
    return _create_model(conf)

def _create_model(conf):
    """
    Creates a simple model using the given configuration
    """
    model = Seq2Seq(**conf['seq2seq'])
    output = _add_layers(model.output, conf['top'])
    model = Model(model.input, output)
    return model

def load_model(model_path):
    model_conf_path = model_path.replace('.h5', '.json')
    with open(model_conf_path, 'r') as f:
        model_conf = json.load(f)['model_conf']
    model = _create_model(model_conf)
    model.load_weights(model_path)
    return model

class MetaModel(object):
    def __init__(self):
        self.models = {key: {i:{} for i in range(1, 8)} \
            for key in WINDOW_TO_PRED_DAYS}

    def predict(self, x, window, is_working):
        input_days = self._get_input_days_from_x(x, window)
        model = self.models[window][input_days][is_working]
        return model.predict(x)

    @staticmethod
    def _get_input_days_from_x(x, window):
        input_days = x.shape[1]
        if window == 'hourly':
            input_days = int(input_days/24)
        return input_days

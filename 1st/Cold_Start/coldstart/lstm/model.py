"""
Model definition
"""
from keras.layers import Input
from keras.models import Model

from coldstart.keras.model import _add_layers
from coldstart.definitions import WINDOW_TO_PRED_DAYS


def create_model(x, conf):
    """
    Creates a simple model using the given configuration
    """
    input_layer = Input(shape=(x.shape[1:]), name='input_layer')
    output = _add_layers(input_layer, conf)
    model = Model(input_layer, output)
    return model

class MetaModel(object):
    def __init__(self):
        self.models = {key: {} for key in WINDOW_TO_PRED_DAYS}

    def predict(self, x, window):
        input_days = self._get_input_days_from_x(x, window)
        model = self.models[window][input_days]
        return self._predict(model, x, window)

    @staticmethod
    def _predict(model, x, window):
        if window == 'hourly':
            for idx in range(24):
                x[:, -24+idx, 0] = model.predict(x[:, idx:-24+idx])[0]
            return x[0, -24:, 0]

    @staticmethod
    def _get_input_days_from_x(x, window):
        input_days = x.shape[1]
        if window == 'hourly':
            input_days = int(input_days/24 - 1)
        return input_days

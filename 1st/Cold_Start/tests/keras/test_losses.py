import numpy as np
import pytest
import keras.backend as K

from coldstart.keras.losses import nmae


def nmae_np(y_true, y_pred):
    cost = np.sqrt((y_true - y_pred)**2)
    return np.mean(cost)

@pytest.mark.parametrize("y_true, y_pred", [
    (np.zeros(10), np.ones(10)),
    (np.ones(10), np.ones(10)),
])
def test_nmae(y_true, y_pred):
    output_1 = K.eval(nmae(K.variable(y_true), K.variable(y_pred)))
    output_2 = nmae_np(y_true, y_pred)
    assert pytest.approx(output_1) == output_2

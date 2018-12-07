import pytest
import numpy as np

from coldstart.metrics import weighted_normalized_mean_abs_error, normalized_mean_abs_error
from coldstart.metrics import get_window_size_weights

@pytest.mark.parametrize('y_true, y_pred, mean_output', [
    (np.ones(5), np.ones(5), 0),
    (np.ones(5), np.zeros(5), 1)
])
def test_normalized_mean_abs_error(y_true, y_pred, mean_output):
    assert np.mean(normalized_mean_abs_error(y_true=y_true, y_pred=y_pred)) == pytest.approx(mean_output)

@pytest.mark.parametrize('y_trues, weights', [
    ([np.ones(24)], [1]),
    ([np.ones(7)], [24/7]),
    ([np.ones(2)], [12]),
    ([np.ones(2), np.ones(7), np.ones(24), np.ones(2)], [12, 24/7, 1, 12]),
])
def test_get_window_size_weights(y_trues, weights):
    assert get_window_size_weights(y_trues) == weights

@pytest.mark.parametrize('y_trues, y_preds, expected_metric', [
    ([np.ones(24)], [np.ones(24)], 0),
    ([np.ones(24)], [np.zeros(24)], 1),
    ([np.ones(24), np.ones(7)], [np.zeros(24), np.zeros(7)], (24+24)/31),
    ([np.ones(24), np.ones(7)], [np.ones(24), np.zeros(7)], (24)/31),
    ([np.ones(24), np.ones(7)], [np.zeros(24), np.ones(7)], (24)/31),
    ([np.ones(24), np.ones(7)], [np.ones(24), np.ones(7)], 0),
])
def test_weighted_normalized_mean_abs_error(y_trues, y_preds, expected_metric):
    metric = weighted_normalized_mean_abs_error(y_trues, y_preds)
    assert metric == pytest.approx(expected_metric)

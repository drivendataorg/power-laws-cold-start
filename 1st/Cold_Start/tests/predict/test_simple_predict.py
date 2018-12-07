import pytest
import numpy as np

from coldstart.predict.simple_repeat import _get_day_consumption, _get_week_consumption

@pytest.mark.parametrize('day', [0])
@pytest.mark.parametrize('metadata', [{
    'monday_is_day_off': False, 'tuesday_is_day_off': False, 'wednesday_is_day_off': True}])
@pytest.mark.parametrize('consumption, weekdays, output', [
    (np.ones(24), np.zeros(24), np.ones(24)),
    (np.ones(96), np.zeros(96), np.ones(24)),
    (np.ones(24), np.ones(24), np.ones(24)),
    (np.ones(24), np.ones(24)*2, np.ones(24)),
    (np.concatenate([np.ones(24), np.zeros(24)]), np.concatenate([np.ones(24), np.ones(24)*2]), np.ones(24)),
    (np.concatenate([np.zeros(24), np.ones(24), ]), np.concatenate([np.ones(24)*2, np.ones(24), ]), np.ones(24)),
])
def test_get_day_consumption(day, consumption, weekdays, metadata, output):
    ret = _get_day_consumption(day, consumption, weekdays, metadata)
    assert (output == ret).all()

_COLUMNS = ['monday_is_day_off', 'tuesday_is_day_off', 'wednesday_is_day_off', 'thursday_is_day_off',
               'friday_is_day_off', 'saturday_is_day_off', 'sunday_is_day_off']

@pytest.mark.parametrize('day', [0])
@pytest.mark.parametrize('metadata', [{column: False for column in _COLUMNS}])
@pytest.mark.parametrize('consumption, weekdays, output', [
    (np.ones(24), np.zeros(24), np.ones(7)*24),
])
def test_get_week_consumption(day, consumption, weekdays, metadata, output):
    ret = _get_week_consumption(day, consumption, weekdays, metadata)
    assert (output == ret).all()

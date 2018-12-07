import pytest

from coldstart.keras.model import MetaModel

def test_MetaModel_get_used_input_days_returns_same_day():
    model = MetaModel()
    model.models['hourly'][3] = None
    model.models['hourly'][5] = None
    assert 5 == model._get_used_input_days('hourly', 5)

def test_MetaModel_get_used_input_days_returns_max_day():
    model = MetaModel()
    model.models['hourly'][3] = None
    model.models['hourly'][5] = None
    assert 5 == model._get_used_input_days('hourly', 7)

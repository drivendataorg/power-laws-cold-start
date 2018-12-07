import pytest
import numpy as np

from coldstart.validation import arrange_train_data, split_series_id
from coldstart.validation import stratified_cv_series_id

@pytest.fixture(scope='module')
def simple_arrange():
    return arrange_train_data([0], 1)

@pytest.fixture(scope='module')
def all_arrange():
    return arrange_train_data([0], 28)

def test_size_of_simple_arrange(simple_arrange):
    assert len(simple_arrange) == 14*3

def test_all_windows_have_14_samples(simple_arrange):
    assert all(simple_arrange.window.value_counts()==14)

def test_all_input_days_have_3_samples(simple_arrange):
    assert all(simple_arrange.input_days.value_counts()==3)

def test_train_indexes_are_correct(simple_arrange):
    assert all(simple_arrange.train_end_idx == \
        simple_arrange.input_days*24 + simple_arrange.train_start_idx)

def test_val_indexes_are_correct(simple_arrange):
    for window, val_days in zip(['hourly', 'daily', 'weekly'], [1, 7, 14]):
        df = simple_arrange[simple_arrange.window == window]
        assert all(df.val_end_idx == val_days*24 + df.val_start_idx)

def test_size_of_all_arrange(all_arrange):
    assert len(all_arrange) == 595

@pytest.mark.parametrize('series_id', [(np.arange(10).tolist())])
@pytest.mark.parametrize('val_id, train_id, fold_idx', [
    ([0, 5], [1, 2, 3, 4, 6, 7, 8, 9], 0),
    ([1, 6], [0, 2, 3, 4, 5, 7, 8, 9], 1)
])
def test_split_series_id(series_id, val_id, train_id, fold_idx):
    ret = split_series_id(series_id, fold_idx)
    assert ret[0] == train_id
    assert ret[1] == val_id

@pytest.mark.parametrize('fold_idx', list(range(5)))
def test_cv_split_preserves_categories(fold_idx):
    categories = np.repeat(np.arange(3), 5)
    train, val = stratified_cv_series_id(categories, fold_idx, 5, 0)
    val = sorted(val)

    assert val == [0, 1, 2]

@pytest.mark.parametrize('fold_idx', list(range(5)))
def test_cv_split_preserves_categories_v2(fold_idx):
    categories = np.arange(15)
    train, val = stratified_cv_series_id(categories, fold_idx, 5, 0)
    val = sorted(val)

    assert len(val) == 3

    for i in range(3):

        assert val[i] in np.arange(i*5, (i+1)*5)
import pytest
import numpy as np

from coldstart.utils import group_sum

@pytest.mark.parametrize('x, group_size, output', [
    (np.arange(4), 1, np.arange(4)),
    (np.arange(4), 2, [1, 5]),
    (np.arange(4), 4, [6]),
])
def test_group_sum(x, group_size, output):
    assert all(output == group_sum(x, group_size))

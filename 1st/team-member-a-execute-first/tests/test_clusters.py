import pytest
import numpy as np

from coldstart.clusters import get_cluster_ohe, SERIES_ID_TO_CLUSTER
from coldstart.clusters import get_cluster_features_v2
from coldstart.utils import load_clusters, load_data

@pytest.fixture()
def all_series_ids():
    train, test, _, _ = load_data()
    all_series_id = train.series_id.unique().tolist() + test.series_id.unique().tolist()
    return all_series_id

def test_ẗhat_all_series_id_can_be_used(all_series_ids):
    for series_id in all_series_ids:
        get_cluster_ohe(series_id)

def test_that_all_elements_in_clusters_get_same_ohe():
    train_clusters, test_clusters = load_clusters()
    for cluster in test_clusters + train_clusters:
        ohes = [get_cluster_ohe(series_id) for series_id in cluster]
        ohe = np.mean(ohes, axis=0)
        assert np.sum(ohe == 1) == 1

def test_ẗhat_all_series_id_have_v2_features(all_series_ids):
    for series_id in all_series_ids:
        get_cluster_features_v2(series_id)

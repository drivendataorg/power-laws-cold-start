import json
import numpy as np
import pandas as pd

from coldstart.definitions import LIBRARY_PATH

with open(LIBRARY_PATH + '/data/series_id_to_cluster.json', 'r') as f:
    SERIES_ID_TO_CLUSTER = json.load(f)

with open(LIBRARY_PATH + '/data/clusters_v2/series_id_to_cluster.json', 'r') as f:
    SERIES_ID_TO_CLUSTER_V2 = json.load(f)

FEATURES_V2 = pd.read_csv(LIBRARY_PATH + '/data/clusters_v2/features.csv')


def get_cluster_ohe(series_id):
    ohe = np.zeros(281, dtype=np.int)
    key = str(series_id)
    if key in SERIES_ID_TO_CLUSTER:
        cluster_id = SERIES_ID_TO_CLUSTER[str(series_id)]
        ohe[cluster_id] = 1
    return ohe

def get_cluster_features_v2(series_id):
    key = str(series_id)
    cluster_id = SERIES_ID_TO_CLUSTER_V2[key]
    return FEATURES_V2.loc[cluster_id].values

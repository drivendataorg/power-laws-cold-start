"""
Functions commonly used in the challenge
"""
import time
from datetime import datetime
import pandas as pd
import json
import numpy as np

from coldstart.definitions import TRAIN_PATH, TEST_PATH, METADATA_PATH, SUBMISSION_PATH
from coldstart.definitions import TRAIN_CLUSTERS_PATH, TEST_CLUSTERS_PATH
from coldstart.definitions import TRAIN_SIMPLE_ARRANGE
from coldstart.definitions import EASTER_HOLIDAYS, HOLIDAYS

def get_timestamp():
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    return time_stamp

def get_datetime(text_date):
    return datetime.strptime(text_date, "%Y-%m-%d %H:%M:%S")

def get_weekday(text_date):
    return datetime.strptime(text_date, "%Y-%m-%d %H:%M:%S").weekday()

def load_data():
    """
    Loads data and applies typical transformations

    Return
    ------
    train, test, submission, metadata
    """
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    metadata = metadata.set_index('series_id')
    train['weekday'] = train.timestamp.apply(get_weekday)
    test['weekday'] = test.timestamp.apply(get_weekday)
    # submission['weekday'] = submission.timestamp.apply(get_weekday)
    train['timestamp'] = train.timestamp.apply(get_datetime)
    test['timestamp'] = test.timestamp.apply(get_datetime)

    train = _remove_bad_series(train)

    return train, test, submission, metadata

def load_clusters():
    """
    Loads clusters of series id

    Returns
    -------
    train_clusters, test_clusters
    """
    with open(TRAIN_CLUSTERS_PATH, 'r') as f:
        train_clusters = json.load(f)
    with open(TEST_CLUSTERS_PATH, 'r') as f:
        test_clusters = json.load(f)
    return train_clusters, test_clusters

def load_simple_arrange():
    simple_arrange = pd.read_csv(TRAIN_SIMPLE_ARRANGE)
    simple_arrange = _remove_bad_series(simple_arrange)
    return simple_arrange

def _remove_bad_series(df):
    for series_id in [102571, 101261]:
        df = df[df.series_id != series_id]
    df.reset_index(drop=True, inplace=True)
    return df

def _is_day_off(series_id, weekday, metadata):
    return metadata.loc[series_id][metadata.columns[2+weekday]]

def _get_next_weekday(day):
    if day < 6:
        return day + 1
    else:
        return 0

def _get_next_date(date, n_days=1):
    return date + np.timedelta64(n_days, 'D')

def group_sum(x, group_size):
    groups = np.reshape(x, (-1, group_size))
    return np.sum(groups, axis=1)

def group_mean(x, group_size):
    groups = np.reshape(x, (-1, group_size))
    return np.mean(groups, axis=1)

def _is_holiday(date):
    day = str(date)[:10]
    if day[-5:] in HOLIDAYS:
        return True
    if day in EASTER_HOLIDAYS:
        return True
    return False

def combine_window_scores(scores):
    values = [scores[0]]*24 + [scores[1]*24/7]*7 + [scores[2]*24/2]*2
    return np.mean(values)
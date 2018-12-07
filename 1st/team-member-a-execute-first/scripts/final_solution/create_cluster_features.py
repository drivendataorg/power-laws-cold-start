"""
This script creates cluster features for training

Adapted from notebook: 045_better_cluster_representation
"""
from termcolor import colored
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from coldstart.utils import load_data, load_clusters, group_sum
from coldstart.definitions import TRAIN_CLUSTERS_PATH, TEST_CLUSTERS_PATH
from coldstart.clusters import SERIES_ID_TO_CLUSTER

train, test, submission, metadata = load_data()
train_clusters, test_clusters = load_clusters()

all_series_id = train.series_id.unique().tolist() + test.series_id.unique().tolist()
for series_id in all_series_id:
    if not str(series_id) in SERIES_ID_TO_CLUSTER:
        new_cluster_id = max(list(SERIES_ID_TO_CLUSTER.values())) + 1
        print(new_cluster_id)
        SERIES_ID_TO_CLUSTER[str(series_id)] = new_cluster_id

CLUSTER_TO_SERIES_ID = {}
for key in SERIES_ID_TO_CLUSTER:
    cluster_id = SERIES_ID_TO_CLUSTER[key]
    if cluster_id in CLUSTER_TO_SERIES_ID:
        CLUSTER_TO_SERIES_ID[cluster_id].append(int(key))
    else:
        CLUSTER_TO_SERIES_ID[cluster_id] = [int(key)]

def main():
    print(colored('\tCreating cluster features', 'blue'))
    df = pd.DataFrame()
    df['hourly_same_day'] = compute_cluster_metrics(hourly_same_day)
    df['hourly_working_days'] = compute_cluster_metrics(hourly_working_days)
    df['hourly_days_off'] = compute_cluster_metrics(hourly_days_off)
    df['daily_same_day'] = compute_cluster_metrics(daily_same_day)
    df['daily_working_days'] = compute_cluster_metrics(daily_working_days)
    df['daily_days_off'] = compute_cluster_metrics(daily_days_off)
    for column in df.columns:
        df[column] = normalize_and_replace_missing_values(df[column].values)

    df.to_csv('../../data/clusters_v2/features.csv', index=False)
    with open('../../data/clusters_v2/series_id_to_cluster.json', 'w') as f:
         json.dump(SERIES_ID_TO_CLUSTER, f)


def compute_cluster_metrics(metric_func):
    cluster_ids = sorted(CLUSTER_TO_SERIES_ID.keys())
    test_series_ids = test.series_id.unique()
    metrics = []
    for cluster_id in tqdm(cluster_ids, desc='Computing cluster metrics'):
        cluster = CLUSTER_TO_SERIES_ID[cluster_id]
        if cluster[0] in test_series_ids:
            df = test
        else:
            df = train
        cluster_metrics = []
        for series_id in cluster:
            sub_df = df[df.series_id == series_id]
            cluster_metrics.append(metric_func(sub_df))
        metrics.append(np.nanmean(cluster_metrics))
    return metrics

def below_min_days(df, min_days):
    n_days = len(df)/24
    if n_days < 7:
        return True
    else:
        return False

## Hourly metrics
def hourly_working_days(df):
    if below_min_days(df, min_days=7):
        return np.nan
    consumption = np.reshape(df.consumption.values, (-1, 24))
    working_days = df.is_holiday.values[::24]
    consumption = consumption[working_days == 0]
    if len(consumption) < 2:
        return np.nan
    metrics = []
    for idx in range(len(consumption)-1):
        values = consumption[idx:idx+2]
        metrics.append(np.std(values, axis=0)/np.mean(values))
    return np.mean(metrics)

def hourly_days_off(df):
    if below_min_days(df, min_days=7):
        return np.nan
    consumption = np.reshape(df.consumption.values, (-1, 24))
    working_days = df.is_holiday.values[::24]
    consumption = consumption[working_days == 1]
    if len(consumption) < 2:
        return np.nan
    metrics = []
    for idx in range(len(consumption)-1):
        values = consumption[idx:idx+2]
        metrics.append(np.std(values, axis=0)/np.mean(values))
    return np.mean(metrics)

def hourly_same_day(df):
    if below_min_days(df, min_days=8):
        return np.nan
    consumption = np.reshape(df.consumption.values, (-1, 24))
    if not len(consumption):
        return np.nan
    metrics = []
    for idx in range(7):
        if len(consumption) > idx + 7:
            values = consumption[idx::7]
            metrics.append(np.std(values, axis=0)/np.mean(values))
    return np.mean(metrics)

## Daily metrics
def daily_working_days(df):
    if below_min_days(df, min_days=7):
        return np.nan
    consumption = group_sum(df.consumption.values, 24)
    working_days = df.is_holiday.values[::24]
    consumption = consumption[working_days == 0]
    if len(consumption) < 2:
        return np.nan
    metrics = []
    for idx in range(len(consumption)-1):
        values = consumption[idx:idx+2]
        metrics.append(np.std(values, axis=0)/np.mean(values))
    return np.mean(metrics)

def daily_days_off(df):
    if below_min_days(df, min_days=7):
        return np.nan
    consumption = group_sum(df.consumption.values, 24)
    working_days = df.is_holiday.values[::24]
    consumption = consumption[working_days == 1]
    if len(consumption) < 2:
        return np.nan
    metrics = []
    for idx in range(len(consumption)-1):
        values = consumption[idx:idx+2]
        metrics.append(np.std(values, axis=0)/np.mean(values))
    return np.mean(metrics)

def daily_same_day(df):
    if below_min_days(df, min_days=8):
        return np.nan
    consumption = group_sum(df.consumption.values, 24)
    if not len(consumption):
        return np.nan
    metrics = []
    for idx in range(7):
        if len(consumption) > idx + 7:
            values = consumption[idx::7]
            metrics.append(np.std(values, axis=0)/np.mean(values))
    return np.mean(metrics)

def normalize_and_replace_missing_values(values):
    mean_value = np.nanmean(values)
    std_value = np.nanstd(values)
    values[np.isnan(values)] = mean_value
    return (values - mean_value)/std_value

if __name__ == '__main__':
    main()

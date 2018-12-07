"""
This script finds clusters on train and test set

Adapted from notebook: 003_repeated_buildings_exploration
"""
from termcolor import colored
import json

from coldstart.utils import load_data
from coldstart.definitions import TRAIN_CLUSTERS_PATH, TEST_CLUSTERS_PATH

def main():
    print(colored('\tFinding clusters', 'blue'))
    train, test, _, metadata = load_data()

    train_clusters = get_clusters_with_time_continuity(train, metadata)
    train_clusters = [cluster_to_int(cluster) for cluster in train_clusters]
    with open(TRAIN_CLUSTERS_PATH, 'w') as f:
        json.dump(train_clusters, f)

    test_clusters = get_clusters_with_time_continuity(test, metadata)
    test_clusters = [cluster_to_int(cluster) for cluster in test_clusters]
    with open(TEST_CLUSTERS_PATH, 'w') as f:
        json.dump(test_clusters, f)

def get_clusters_with_time_continuity(df, metadata):
    clusters = get_just_metadata_clusters(df, metadata)
    new_clusters = []
    for old_cluster in clusters:
        current_cluster = [old_cluster[0]]
        for series_id in old_cluster[1:]:
            end_date = df.timestamp.values[df.series_id.values == current_cluster[-1]][-1]
            start_date = df.timestamp.values[df.series_id.values == series_id][0]
            if end_date < start_date:
                current_cluster.append(series_id)
            else:
                new_clusters.append(current_cluster)
                current_cluster = [series_id]
        new_clusters.append(current_cluster)
    print('The number of clusters with time continuity is: %i' % len(new_clusters))
    return new_clusters

def get_just_metadata_clusters(df, metadata):
    df_metadata = metadata.loc[df.series_id.unique()]
    clusters = []
    current_cluster = [df_metadata.index[0]]
    for i in range(1, len(df_metadata)):
        if all(df_metadata.values[i] == df_metadata.values[i-1]):
            current_cluster.append(df_metadata.index[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [df_metadata.index[i]]
    print('The number of clusters is: %i' % len(clusters))
    return clusters

def cluster_to_int(cluster):
    return [int(value) for value in cluster]

if __name__ == '__main__':
    main()

"""
Commonly used paths, labels, and other stuff
"""
import os

DATASET_PATH = '/home/ubuntu/schneider-cold-start-solutions/1st/Cold_Start'
TRAIN_PATH = os.path.join(DATASET_PATH, 'data', 'train.csv')
TEST_PATH = os.path.join(DATASET_PATH, 'data', 'test.csv')
METADATA_PATH = os.path.join(DATASET_PATH, 'data', 'meta.csv')
SUBMISSION_PATH = os.path.join(DATASET_PATH, 'data', 'submission_format.csv')
LIBRARY_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

TRAIN_CLUSTERS_PATH = os.path.join(LIBRARY_PATH, 'data/train_clusters.json')
TEST_CLUSTERS_PATH = os.path.join(LIBRARY_PATH, 'data/test_clusters.json')

TRAIN_SIMPLE_ARRANGE = os.path.join(LIBRARY_PATH, 'data/simple_train_arrange.csv')

EASTER_HOLIDAYS = {
    '2017-04-14', '2017-04-17',
    '2016-03-25', '2016-03-28',
    '2015-04-03', '2015-04-06',
    '2014-04-18', '2014-04-21',
    '2013-03-29', '2013-04-01',
}
HOLIDAYS = {
    '01-01', '01-02', '01-06', #christmas
    '05-01', #workers day
    '08-15', #Assumption of Mary
    '11-01', #all saints day
    '12-24', '12-25', '12-26', '12-27', #christmas
    '12-28', '12-29',  '12-31'
}

WINDOW_TO_PRED_DAYS = {
    'hourly': 1,
    'daily': 7,
    'weekly': 14,
}

SURFACE_OHE = {
    'x-large': [1, 0, 0, 0, 0, 0, 0],
    'x-small': [0, 1, 0, 0, 0, 0, 0],
    'medium': [0, 0, 1, 0, 0, 0, 0],
    'large': [0, 0, 0, 1, 0, 0, 0],
    'xx-large': [0, 0, 0, 0, 1, 0, 0],
    'xx-small': [0, 0, 0, 0, 0, 1, 0],
    'small': [0, 0, 0, 0, 0, 0, 1]
}

BASE_TEMPERATURE_OHE = {
    'low': [1, 0],
    'high': [0, 1]
}

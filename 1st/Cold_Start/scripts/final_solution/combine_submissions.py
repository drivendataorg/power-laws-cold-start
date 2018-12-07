"""
Combination of submissions to create final solution

Adapted from notebook: 035_combine_submissions and 071_megaensemble
"""
import os
import glob
from termcolor import colored
from tqdm import tqdm
import numpy as np
import pandas as pd

from coldstart.utils import load_data
from coldstart.definitions import DATASET_PATH
from coldstart.utils import _is_holiday, _is_day_off

def main():
    print(colored('\tCombine submissions', 'blue'))

    submission_paths = glob.glob(
        DATASET_PATH + '/submissions/seq2seq/*.csv')
    output_path = os.path.join(DATASET_PATH, 'submissions', 'seq2seq.csv')
    average_submissions(output_path, *submission_paths)

    submission_paths = glob.glob(
        DATASET_PATH + '/submissions/tailor/*.csv')
    output_path = os.path.join(DATASET_PATH, 'submissions', 'tailor.csv')
    average_submissions(output_path, *submission_paths)

    seq2seq_sub = pd.read_csv(DATASET_PATH + '/submissions/seq2seq.csv')
    tailor_sub = pd.read_csv(DATASET_PATH + '/submissions/tailor.csv')

    final_sub = tailor_sub.copy()
    final_sub['consumption'][final_sub.prediction_window == 'hourly'] += \
        seq2seq_sub['consumption'][final_sub.prediction_window == 'hourly']
    final_sub['consumption'][final_sub.prediction_window == 'hourly'] /= 2

    final_sub.to_csv(DATASET_PATH + '/submissions/final_solution.csv', index=False)



def average_submissions(output_path, *submission_paths):
    dfs = [pd.read_csv(submission_path) for submission_path in submission_paths]
    consumptions = [df.consumption.values for df in dfs]
    average_consumption = np.mean(consumptions, axis=0)
    submission = dfs[0].copy()
    submission['consumption'] = average_consumption
    submission.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()

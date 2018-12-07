import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

def arrange_train_data(series_ids, n_days, random_seed=0):
    """
    Creates a dataframe with the instructions for splitting the train
    data between train and validation.

    Parameters
    ----------
    series_ids : list
        List with the ids that we want to use
    n_days : int
        How many different samples we want to create for each window size and
        number of input days.
    """
    np.random.seed(random_seed)
    min_input_days = 1
    max_input_days = 14
    train_days = 28
    columns = ['series_id', 'window', 'input_days',
               'train_start_idx', 'train_end_idx',
               'val_start_idx', 'val_end_idx']
    df = {column:[] for column in columns}
    for series_id in tqdm(series_ids):
        for window, val_days in zip(['hourly', 'daily', 'weekly'], [1, 7, 14]):
            for input_days in range(min_input_days, max_input_days+1):
                start_days = np.arange(0, train_days - input_days - val_days + 1)
                if len(start_days) <= n_days:
                    chosen_start_days = start_days
                else:
                    chosen_start_days = np.random.choice(start_days, n_days, replace=False)
                for start_day in chosen_start_days:
                    train_start_idx = start_day*24
                    train_end_idx = train_start_idx + 24*input_days
                    val_start_idx = train_end_idx
                    val_end_idx = val_start_idx + 24*val_days
                    #Add to the dataframe
                    data = [series_id, window, input_days,
                            train_start_idx, train_end_idx,
                            val_start_idx, val_end_idx]
                    for i, column in enumerate(columns):
                        df[column].append(data[i])
    df = pd.DataFrame(df)
    return df

def split_series_id(series_id, fold_idx, n_folds=5):
    val_ids = series_id[fold_idx::n_folds]
    train_ids = [_id for _id in series_id if _id not in val_ids]
    return train_ids, val_ids

def stratified_cv_series_id(series_id, fold_idx, n_folds=5, random_seed=0):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    categories = np.arange(0, len(series_id)//n_folds+1)
    categories = np.repeat(categories, n_folds)[:len(series_id)]
    splits = [(train_index, test_index) for train_index, test_index in skf.split(series_id, categories)]

    val_ids = [series_id[idx] for idx in splits[fold_idx][1]]
    train_ids =[series_id[idx] for idx in splits[fold_idx][0]]
    return train_ids, val_ids

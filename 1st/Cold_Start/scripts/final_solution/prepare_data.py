"""
This script transforms the original data:
- deletes index
- adds weekday
- adds is_day_off
- adds is_holiday

Adapted from notebook: 022_adding_more_information_to_dataset
"""
from termcolor import colored
from tqdm import tqdm

from coldstart.utils import load_data
from coldstart.definitions import TRAIN_PATH, TEST_PATH, SUBMISSION_PATH
from coldstart.utils import _is_holiday, _is_day_off

def main():
    print(colored('\tPreparing data', 'blue'))
    train, test, _, metadata = load_data()

    add_is_off_column(train, metadata)
    add_is_off_column(test, metadata)
    add_is_holiday_column(train)
    add_is_holiday_column(test)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

def add_is_off_column(df, metadata):
    if 'is_day_off' in df.columns:
        return
    is_day_off = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Adding day off'):
        is_day_off.append(_is_day_off(row['series_id'], row['weekday'], metadata))
    df['is_day_off'] = is_day_off

def add_is_holiday_column(df):
    if 'is_holiday' in df.columns:
        return
    is_holiday = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Adding holidays'):
        is_holiday.append(_is_holiday(row['timestamp']) or row['is_day_off'])
    df['is_holiday'] = is_holiday

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from fastai import *
from fastai.tabular import *
from pathlib import Path
from time import time


def mae(pred:Tensor, targ:Tensor):
    return torch.mean(torch.abs(targ - pred))

def compress(df):
    int64_columns = [c for c in df.columns if df.loc[:, c].dtype=='int64']
    float64_columns = [c for c in df.columns if df.loc[:, c].dtype=='float64']
    for col in int64_columns:
        df.loc[:, col] = df.loc[:, col].astype(np.int32)
    for col in float64_columns:
        df.loc[:, col] = df.loc[:, col].astype(np.float32)
    return df


print('Loading data')
t0 = time()
repo_path = Path(os.path.dirname(os.getcwd()))
models_path = repo_path / 'models'

X = pd.read_csv(repo_path/'data/processed/X.csv', index_col=0)
h_test = pd.read_csv(repo_path/'data/processed/h.csv', index_col=0)
d_test = pd.read_csv(repo_path/'data/processed/d.csv', index_col=0)
w_test = pd.read_csv(repo_path/'data/processed/w.csv', index_col=0)

X = compress(X)
h_test = compress(h_test)
d_test = compress(d_test)
w_test = compress(w_test)

len_h_test, len_d_test, len_w_test = h_test.shape[0], d_test.shape[0], w_test.shape[0]
test = h_test.append(d_test).append(w_test)
X.drop(['timestamp'], axis=1, inplace=True)
test.drop(['timestamp'], axis=1, inplace=True)

day_off_cols = [c for c in X.columns if c.endswith('_day_off')]
cat_names = ['series_id', 'surface', 'base_temperature', 'timestampYear', 'timestampMonth', 'timestampWeek',
            'timestampDay', 'timestampDayofweek', 'timestampDayofyear', 'timestampIs_year_end', 'timestampIs_year_start', 
            'timestampIs_month_end', 'timestampIs_month_start', 'timestampIs_quarter_end', 'timestampIs_quarter_start', 'hour', 'temperature_nan', 
            'working_day', 'work_schedule', 'worked_yesterday', 'works_tomorrow', 'has_day_off', 'has_working_day'] + day_off_cols

cont_time_cols = [c for c in X.columns if c.endswith('_cont')]
circ_cols = [c for c in X.columns if c.startswith('circ_')]
cont_names = ['temperature', 'timestampElapsed', 'temperature_d', 'temperature_w','temp_rolling_3', 'temp_rolling_6', 'temp_rolling_12',
               'temp_rolling_24', 'len', 'current_day', 'id_min', 'id_max'] + cont_time_cols  + circ_cols

len_test = test.shape[0]
full = X.append(test)
for n in cat_names:
    full[n] = full[n].astype('category').cat.as_ordered()
X = full[:-len_test]
test = full[-len_test:]

print('Starting fitting')
preds = []
scores = []
ids = X.series_id.unique()
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for i, (train_idxs, val_idxs) in enumerate(kf.split(X, X.series_id)):
    print(f'fold: {i}')
    X_train = X.iloc[train_idxs, :].copy().reset_index(drop=True)
    X_val = X.iloc[val_idxs, :].copy().reset_index(drop=True)
    X_test = test.copy().reset_index(drop=True)
    tfms = [FillMissing, Categorify]
    cat_szs = [(c, len(X_train[c].cat.categories)+1) for c in cat_names]
    emb_szs = {c: min(50,v//2) for c,v in cat_szs}
    
    data = TabularDataBunch.from_df(models_path, X_train, X_val, test_df=X_test, dep_var='consumption', tfms=tfms, cat_names=cat_names, cont_names=cont_names, log_output=False, bs=2048)
    learn = get_tabular_learner(data, [512, 512, 256, 64, 8, 1], emb_szs=emb_szs, ps=[0.0, 0.0, 0.0, 0.0, 0.0], emb_drop=0.0, y_range=None, 
                                use_bn=True, loss_func=F.l1_loss, true_wd=True, callback_fns=[ShowGraph, SaveModelCallback], wd=0.001, bn_wd=False)
    learn.fit_one_cycle(50, 0.01)
    learn.save(f'model_fold_{i}')
    
    pred, targets = learn.get_preds(is_test=False)
    score = mae(pred, targets).item()
    scores.append(score)
    print(f'Score on fold {i}: {score}, time: {time()-t0}')
    pred_test = learn.get_preds(is_test=True)[0].numpy()
    preds.append(pred_test)

print(f'Mean val score across validation: {np.round(np.mean(scores), 5)}')

print('Generating submission')
predict = np.mean(preds, axis=0)
h_test.consumption = predict[:len_h_test]
d_test.consumption = predict[len_h_test:-len_w_test]
w_test.consumption = predict[-len_w_test:]
for df in (h_test, d_test, w_test):
    df.loc[:, 'predict'] = df.consumption*(df.id_max-df.id_min+1) + df.id_min

my_submission = pd.read_csv(repo_path/'data/raw/submission_format.csv', index_col='pred_id', parse_dates=['timestamp'])
for _id in my_submission.series_id.unique():
    mask = my_submission.series_id == _id

    if my_submission.loc[mask, 'prediction_window'].iloc[0] == 'hourly':
        arr = np.array(h_test.loc[h_test.series_id == _id, 'predict'])
        my_submission.loc[mask, 'consumption'] = arr
    if my_submission.loc[mask, 'prediction_window'].iloc[0] == 'daily':
        arr = np.array(d_test.loc[d_test.series_id == _id, 'predict'])
        arr = np.sum(arr.reshape(-1, 24), axis=1)
        my_submission.loc[mask, 'consumption'] = arr
    if my_submission.loc[mask, 'prediction_window'].iloc[0] == 'weekly':
        arr = np.array(w_test.loc[w_test.series_id == _id, 'predict'])
        arr = np.sum(arr.reshape(-1, 24*7), axis=1)
        my_submission.loc[mask, 'consumption'] = arr

my_submission.to_csv(repo_path/f'data/submit.csv', index_label='pred_id')
print('Saved submittion')
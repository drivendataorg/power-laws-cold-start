"""
Visualization of predictions
"""
import matplotlib.pyplot as plt

from coldstart.utils import _is_day_off, group_sum

def visualize_idx(idx, train, train_arrange, preds, metadata):
    row = train_arrange.loc[idx]
    df = train[train.series_id == row['series_id']]
    consumption = df.consumption.values[row['train_start_idx']: row['val_end_idx']]
    dates = df.timestamp.values[row['train_start_idx']: row['val_end_idx']]
    weekdays = df.weekday.values[row['train_start_idx']: row['val_end_idx']]

    if row['window'] == 'hourly':
        batch_size = 24
    elif row['window'] == 'daily':
        batch_size = 1
        weekdays = weekdays[::24]
        dates = dates[::24]
        consumption = group_sum(consumption, 24)

    plt.plot(dates[-len(preds[idx]):], preds[idx], color='green', lw=3)
    plt.plot(dates[-len(preds[idx]):][::batch_size], preds[idx][::batch_size],
             'o', color='green', lw=3)
    for i in range(len(dates)//batch_size):
        weekday = weekdays[i*batch_size]
        if _is_day_off(row['series_id'], weekday, metadata):
            color = 'orange'
        else:
            color = 'blue'
        plt.plot(
            dates[i*batch_size: (i+1)*batch_size + 1],
            consumption[i*batch_size: (i+1)*batch_size + 1],
            color=color)
        plt.plot(
            dates[i*batch_size: (i)*batch_size + 1],
            consumption[i*batch_size: (i)*batch_size + 1],
            'o', color=color)
    plt.title('%i Nmae: %.3f' % (idx, row['nmae']))

def show_biggest_errors(window, start_idx, train, train_arrange, preds, metadata):
    df = train_arrange[train_arrange.window == window]
    df = df.sort_values('nmae')
    indexes = df.index[::-1][start_idx:start_idx + 4]
    show_4_prediction_plots(indexes, train, train_arrange, preds, metadata)

def show_4_prediction_plots(indexes, train, train_arrange, preds, metadata):
    n_rows, n_cols = (2, 2)
    plt.figure(figsize=(15, 7))
    for i, idx in enumerate(indexes):
        plt.subplot(n_rows, n_cols, i+1)
        visualize_idx(idx, train, train_arrange, preds, metadata)

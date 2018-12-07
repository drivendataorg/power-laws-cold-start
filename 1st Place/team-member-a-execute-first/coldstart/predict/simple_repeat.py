"""
Implementation of predictor that simply repeats the input data
"""
import numpy as np

def simple_repeat(window, consumption, weekdays, metadata):
    """
    Given input data creates the best prediction possible by repeating
    the input data
    """
    next_day = _get_next_day(weekdays[-1])
    if window == 'hourly':
        return _get_day_consumption(next_day, consumption, weekdays, metadata)
    elif window == 'daily':
        return _get_week_consumption(next_day, consumption, weekdays, metadata)
    elif window == 'weekly':
        week_consumption = _get_week_consumption(next_day, consumption, weekdays, metadata)
        week_consumption = np.sum(week_consumption)
        return np.ones(2)*week_consumption
    else:
        raise Exception('Unknown window type')

def _get_day_consumption(day, consumption, weekdays, metadata):
    matches = weekdays == day
    if np.any(matches):
        consumption = consumption[matches]
        consumption = np.reshape(consumption, (-1, 24))
        return np.mean(consumption, axis=0)
    else:
        # Then we have to use the most similar day. is off is the best option
        # This means that we have less than a week of data
        is_day_off = _is_day_off(day, metadata)
        similar_consumption = []
        for _day in np.unique(weekdays):
            if is_day_off == _is_day_off(_day, metadata):
                similar_consumption.append(consumption[weekdays == _day])
        if similar_consumption:
            return np.mean(similar_consumption, axis=0)
        else:
            # Then we don't have similar days, simply compute mean of all input data
            consumption = np.reshape(consumption, (-1, 24))
            return np.mean(consumption, axis=0)

def _is_day_off(day, metadata):
    columns = ['monday_is_day_off', 'tuesday_is_day_off', 'wednesday_is_day_off', 'thursday_is_day_off',
               'friday_is_day_off', 'saturday_is_day_off', 'sunday_is_day_off']
    return metadata[columns[int(day)]]

def _get_next_day(day):
    if day < 6:
        return day + 1
    else:
        return 0

def _get_week_consumption(start_day, consumption, weekdays, metadata):
    current_day = start_day
    week_consumption = np.zeros(7)
    for i in range(7):
        day_consumption = _get_day_consumption(current_day, consumption, weekdays, metadata)
        week_consumption[i] = np.sum(day_consumption)
        current_day = _get_next_day(current_day)
    return week_consumption

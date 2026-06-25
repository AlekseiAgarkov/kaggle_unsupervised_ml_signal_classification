from functools import wraps

import numpy as np
import pandas as pd


def transform(func):
    @wraps(func)
    def wrapper(df, **kwargs):
        return df.apply(lambda row: pd.Series({**row.to_dict(), **func(row, **kwargs)}), axis=1)

    return wrapper


def filter_signal_end_to_window(arr, window_size):
    """Отбрасывает индексы сигнала после закрытия окна"""
    return [x for i, x in enumerate(arr) if i == 0 or x <= arr[i - 1] + window_size]


@transform
def calc_base_signal_components(row,
                                signal_level_threshold_at_end=0.1,
                                noise_threshold: int = 24,
                                start_idx: int = 140,
                                end_offset=20):
    arr = row['data_array']
    if row.name == 1929:
        arr = row['data_array'][:300]
    if row.name == 19696:
        arr = row['data_array'][:250]

    peak_idx = len(arr) - 1 - np.argmax(arr[::-1])
    amplitude = arr[peak_idx]

    signal_indices_in_data_array = np.where(arr > max(signal_level_threshold_at_end * amplitude, noise_threshold))[0]
    signal_indices_in_data_array = filter_signal_end_to_window(signal_indices_in_data_array, end_offset)

    end = min(len(arr), signal_indices_in_data_array[-1] + end_offset)
    signal_indices_in_data_array = np.arange(start_idx, end)

    signal_array = arr[signal_indices_in_data_array]
    peak_idx_in_signal = len(signal_array) - 1 - np.argmax(signal_array[::-1])

    return {'signal_start': signal_indices_in_data_array[0],
            'signal_end': signal_indices_in_data_array[-1],
            'signal_indices_in_data_array': signal_indices_in_data_array,
            'signal_array': signal_array,
            'signal_own_indices': (np.arange(len(signal_array))),
            'peak_idx': peak_idx,
            'peak_idx_in_signal': peak_idx_in_signal,
            'amplitude': amplitude,
            'signal_area': (np.trapezoid(signal_array)),
            'signal_duration': (len(signal_array))}

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

from src.signal_processing.preprocessing import transform


@transform
def calc_psd_at_step_offset(row, offset_ticks, short_gate, min_long_len: int = 8):
    """
    Расчёт PSD по отступам offset_ticks, short_gate.
    """
    signal_array = row['signal_array']
    signal_end_idx = len(row['signal_own_indices'])
    peak_idx_in_signal = row['peak_idx_in_signal']

    long_start_idx = min(peak_idx_in_signal + offset_ticks, signal_end_idx)
    signal = np.clip(signal_array[long_start_idx:signal_end_idx + 1], 0.0, None)
    long_length = len(signal)

    if long_length < max(short_gate + 1, min_long_len):
        return {
            "psd": np.nan,
            "long_area": np.nan,
            "short_area": np.nan,
            "long_length": long_length
        }

    long_area = np.sum(signal)
    short_area = np.sum(signal[:int(short_gate)])
    psd = (long_area - short_area) / long_area

    return {
        "psd": psd,
        "long_area": long_area,
        "short_area": short_area,
        "long_length": long_length
    }


def vote_labels(arr1, arr2):
    """
    Выбор метки кластера по двум наборам меток:
    - 0 и 0 -> 0
    - 1 и 1 -> 1
    - 0 и 1 -> 2
    - 1 и 0 -> 2

    :param arr1: Массив меток 1
    :param arr2: Массив меток 2
    :return: Новые метки
    """
    arr1, arr2 = np.asarray(arr1), np.asarray(arr2)
    result = np.where(arr1 == arr2, arr1, 2)
    return result

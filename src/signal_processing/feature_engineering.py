import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

from src.signal_processing.preprocessing import transform


def exp_decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


@transform
def calc_tau_at_amp_pcnt(row, amp_pcnt_offset: float):
    signal_array = row['signal_array']
    signal_own_indices = row['signal_own_indices']
    amplitude = row['amplitude']
    peak_idx_in_signal = row['peak_idx_in_signal']
    decay_array = signal_array[peak_idx_in_signal:]
    time_array = signal_own_indices[peak_idx_in_signal:]

    popt, _ = curve_fit(exp_decay, time_array, decay_array, p0=[amplitude, 10, 0])
    tau = popt[1]
    tau_at_target_amp = -tau * np.log((1 - amp_pcnt_offset) * amplitude / popt[0]) if popt[0] > 0 else None
    amp_at_offset = exp_decay(tau_at_target_amp, *popt)

    area_fast, _ = quad(exp_decay, time_array[0], tau_at_target_amp, args=tuple(popt))
    area_slow, _ = quad(exp_decay, tau_at_target_amp, time_array[-1], args=tuple(popt))
    pcnt_str = f"p{amp_pcnt_offset * 100:.0f}_offset"
    return {f"tau_at_amp_{pcnt_str}": tau_at_target_amp,
            f"amp_at_offset_{pcnt_str}": amp_at_offset,
            f"area_fast_{pcnt_str}": area_fast,
            f"area_slow_{pcnt_str}": area_slow}

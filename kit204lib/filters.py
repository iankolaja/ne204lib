import copy
import numpy as np


def trapezoidal_filter(peaking_time, gap_time, tau, waveform, sampling_interval=4e-9, pre_trigger_delay=1000):
    """
    function to apply a trapezoidal filter to a waveform
    peaking_time = Time in seconds
    gap_time = Time in seconds
    tau = Tau value of exponential decay for pole zero correction
    waveform = Individual 1D raw waveform
    sampling_interval = Time per sample (default 4ns)
    pre_trigger_delay = Number of samples before waveform rise (default 1000)
    """
    k = np.int(peaking_time / sampling_interval)
    l = np.int(k + gap_time / sampling_interval)
    M = 1 / (np.exp(sampling_interval / tau) - 1)
    M = (1 + 1 / tau)

    noise_average = np.mean(waveform[0:pre_trigger_delay])

    averaged_waveform = copy.copy(waveform)
    averaged_waveform[:l + k] = noise_average

    d_n = averaged_waveform[k + l:]
    d_nk = averaged_waveform[l:-k]
    d_nl = averaged_waveform[k:-l]
    d_nlk = averaged_waveform[:-(k + l)]

    filter_shape = d_n - d_nk - d_nl + d_nlk
    filter_shape[0] = 0
    ds_M = filter_shape * M
    s = np.cumsum(ds_M)
    peak_val = np.max(s)
    return s, ds_M, peak_val
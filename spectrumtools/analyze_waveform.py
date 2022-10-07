import scipy.integrate as integrate
import numpy as np
from scipy.optimize import curve_fit
import scipy.signal
import matplotlib.pyplot as plt
from .generate_trapezoid_filter import generate_trapezoid_filter


def plot_pulses(raw_data, num_pulses):
    plt.figure()
    for i in range(num_pulses):
        plt.plot(raw_data[i])
    plt.show()


def exponential(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def take_rolling_average(waveform, width):
    smooth_waveform = np.convolve(waveform, np.ones(width), 'valid') / width
    return smooth_waveform


def find_peak(waveform, prominence_val=30):
    max_location = scipy.signal.find_peaks(waveform, prominence=prominence_val)[0][0]
    return max_location


def fit_tau(waveform):
    max_location = find_peak(waveform)
    n = np.arange(0, len(waveform))
    decay_indices = n[max_location:-1]
    decay_indices_norm = (decay_indices - decay_indices[0]) / (decay_indices[-1] - decay_indices[0])
    c_0 = waveform[-1]
    tau_0 = 1
    a_0 = (waveform[0] - waveform[-1])
    popt, pcov = curve_fit(exponential, decay_indices_norm, waveform[decay_indices], p0=(a_0, tau_0, c_0))
    a, tau, c = popt
    tau *= np.max(n)
    return tau


def shape_waveform(waveform, pulse_filter, k, pre_trigger, plot_filtered=False):
    peak_location = pre_trigger + k
    sample_length = len(pulse_filter)

    # Convolve the filter with the decay part of the waveform
    decay = waveform[range(peak_location, peak_location + sample_length)]
    filtered = np.convolve(decay, pulse_filter)[:sample_length]

    # Integrate the trapezoid
    amplitude = np.max(filtered)

    if plot_filtered:
        plt.plot(filtered)
        plt.legend()
        plt.show()

    return amplitude, filtered
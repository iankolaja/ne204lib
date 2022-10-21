import scipy.integrate as integrate
import numpy as np
from scipy.optimize import curve_fit
import scipy.signal
import matplotlib.pyplot as plt
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import HoverTool


def plot_pulses(raw_data, num_pulses, tau=None):
    TOOLTIPS = [
        ("(x,y)", "($x, $y)"),
    ]
    x_max = len(raw_data[0])
    p = figure(background_fill_color="#fafafa",
               width=1200, height=700,
               x_range=(0, x_max),
               tooltips=TOOLTIPS, tools = "pan,wheel_zoom,box_zoom,reset")
    p.yaxis.axis_label = 'Counts'
    x_range = np.arange(x_max)
    for i in range(num_pulses):
        p.line(x_range, raw_data[i])
    show(p)


def exponential(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def take_rolling_average(waveform, width):
    smooth_waveform = np.convolve(waveform, np.ones(width), 'valid') / width
    return smooth_waveform


def find_peak(waveform, prominence_val=30):
    max_location = scipy.signal.find_peaks(waveform, prominence=prominence_val)[0][0]
    return max_location


def fit_tau(waveform, pre_sample_length, fit_length = 1000, show_plot=False):
    #max_location = find_peak(waveform)
    waveform = take_rolling_average(waveform, 20)
    max_location = pre_sample_length
    n = np.arange(0, len(waveform))
    decay_indices = n[max_location:max_location+fit_length]
    decay_indices_norm = (decay_indices - decay_indices[0]) / (decay_indices[-1] - decay_indices[0])
    c_0 = waveform[-1]
    tau_0 = 1
    a_0 = (waveform[0] - waveform[-1])
    popt, pcov = curve_fit(exponential, decay_indices_norm, waveform[decay_indices], p0=(a_0, tau_0, c_0))
    a, tau_norm, c = popt
    tau = tau_norm * np.max(n)
    if show_plot:
        plt.figure()
        fit_vals = exponential(decay_indices_norm, a, tau_norm, c)
        plt.plot(waveform)
        plt.plot(decay_indices, fit_vals)
        plt.show()
    return tau


def shape_waveform(waveform, pulse_filter, k, pre_trigger, plot_filtered=False):
    peak_location = pre_trigger + k
    sample_length = len(pulse_filter)

    # Convolve the filter with the decay part of the waveform
    decay = waveform[range(peak_location, peak_location + sample_length)]
    filtered = np.convolve(decay, pulse_filter)[:sample_length]

    # Integrate the trapezoid
    amplitude = np.sum(filtered)

    if plot_filtered:
        plt.plot(filtered)
        plt.legend()
        plt.show()

    return amplitude, filtered
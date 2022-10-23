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


def exponential(t, a, tau):
    return a * np.exp(-t / tau)


def take_rolling_average(waveform, width):
    smooth_waveform = np.convolve(waveform, np.ones(width), 'valid') / width
    return smooth_waveform


def fit_tau(waveform, pre_sample_length, fit_length = 1000, show_plot=False):
    smooth_waveform = take_rolling_average(waveform, 20)
    decay_waveform = smooth_waveform[pre_sample_length:pre_sample_length+fit_length]
    x = np.arange(0, fit_length)
    x_norm = (x - x[0]) / (x[-1] - x[0])
    tau_0 = 10000/fit_length
    a_0 = decay_waveform[0]
    popt, pcov = curve_fit(exponential, x_norm, decay_waveform, p0=(a_0, tau_0))
    a, tau_norm = popt
    tau = tau_norm * fit_length
    if show_plot:
        plt.figure()
        fit_vals = exponential(x_norm, a, tau_norm)
        plt.plot(waveform)
        plt.plot(x+pre_sample_length, fit_vals)
        plt.show()
    return tau
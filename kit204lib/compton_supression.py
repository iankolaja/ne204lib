import numpy as np


def r59(waveform, pre_sample_length=1000, rise_width=200, baseline_sample_width = 30):
    time = np.arange(int(pre_sample_length-rise_width), int(pre_sample_length+rise_width))
    waveform_rise = waveform[time]
    baseline = np.mean(waveform[0:baseline_sample_width])
    waveform_rise = waveform_rise+np.abs(baseline)
    waveform_rise = np.maximum.accumulate(waveform_rise)
    amplitude = np.max(waveform_rise)
    amplitude_10 = amplitude*0.10
    amplitude_50 = amplitude*0.50
    amplitude_90 = amplitude*0.90
    t_10 = np.interp(amplitude_10, waveform_rise, time)-time[0]
    t_50 = np.interp(amplitude_50, waveform_rise, time)-time[0]
    t_90 = np.interp(amplitude_90, waveform_rise, time)-time[0]
    r_15 = t_50-t_10
    r_19 = t_90-t_10
    r_59 = t_90-t_50
    R59 = r_15/r_19
    intersections = ((t_10, amplitude_10), (t_50, amplitude_50), (t_90, amplitude_90))
    return R59, r_15, r_59, intersections, waveform_rise
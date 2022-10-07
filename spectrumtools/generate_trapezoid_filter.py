import numpy as np


def _truncated_ramp(t, k):
    end_index = int(np.where(t == k)[0][0])+1
    rise = t[0:end_index]
    fall = np.zeros(len(t)-np.argmin(np.abs(t-k))-1)
    return np.concatenate((rise,fall))


def _moving_average(t, l):
    return np.concatenate((np.ones(np.argmin(np.abs(t-l))+1),
            np.zeros(len(t)-np.argmin(np.abs(t-l))-1)))


def generate_trapezoid_filter(tau, k, l):
    t = np.arange(0, k+l)
    rising_ramp = _truncated_ramp(t,k)
    first_average = _moving_average(t,l)
    second_average = np.roll(_truncated_ramp(t, k), l)
    descending_ramp = np.roll(_moving_average(t, l), k)
    filter_shape = rising_ramp + tau*first_average + (k-tau)*descending_ramp - second_average
    return filter_shape

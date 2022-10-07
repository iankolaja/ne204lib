import numpy as np


def generate_trapezoid_filter(tau, k, l):
    rising_ramp = np.ones(k)*np.exp(np.arange(k)/tau)
    first_average = np.zeros(l)
    descending_ramp = -np.ones(k)*np.exp(np.flip(np.arange(k))/tau)
    filter_shape = np.hstack((rising_ramp,first_average,descending_ramp))
    return filter_shape


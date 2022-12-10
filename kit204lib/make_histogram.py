import numpy as np
import matplotlib.pyplot as plt


def make_histogram(trapezoid_heights, num_bins, upper_noise_cutoff=0.995, do_plot=False,
                   log_plot = True):
    spectrum_start = np.min(trapezoid_heights)
    spectrum_end = np.quantile(trapezoid_heights, upper_noise_cutoff)


    counts, amplitude_bins = np.histogram(trapezoid_heights,
                                bins=num_bins,
                                range=[spectrum_start, spectrum_end])
    channels = np.arange(1, 1+len(counts))
    amplitude_bins = (amplitude_bins[1:] + amplitude_bins[:-1]) / 2
    if do_plot:
        plt.plot(channels, counts)
        plt.xlabel("Channel")
        plt.ylabel("Counts")
        if log_plot:
            plt.yscale("log")
        plt.show()
    return counts, amplitude_bins
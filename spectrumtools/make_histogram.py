import numpy as np
import matplotlib.pyplot as plt


def make_histogram(trapezoid_heights, num_bins, upper_noise_cutoff=0.995, do_plot=False):
    spectrum_start = np.min(trapezoid_heights)
    spectrum_end = np.quantile(trapezoid_heights, upper_noise_cutoff)


    counts, bins = np.histogram(trapezoid_heights,
                                bins=num_bins,
                                range=[spectrum_start, spectrum_end])
    channels = np.arange(1, 1+len(counts))
    if do_plot:
        plt.plot(channels, counts)
        plt.xlabel("Channel")
        plt.ylabel("Counts")
        plt.show()
    return counts
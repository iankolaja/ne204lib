import numpy as np
from bokeh.io import show, output_file
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import scipy.signal as sig
import copy
import h5py


class Spectrum:
    def __init__(self, input_file_path=None, num_channels=8040):
        self.counts = np.zeros(num_channels)
        self.channels = np.arange(1,num_channels+1)
        self.energies = self.channels
        self.num_channels = num_channels
        self.peaks = []
        self.calibrated = False
        self.calibration_coeffs = [1.0]
        if type(input_file_path) == str:
            with open(input_file_path, 'r') as f:
                for line in f:
                    line = line.split(',')
                    time = line[0]
                    channel = int(line[1])
                    self.counts[channel] += 1

    def show_histogram(self, title):
        if self.calibrated:
            x = np.concatenate((np.array([0.0]),self.energies))
        else:
            x = np.concatenate((np.array([0]),self.channels))
        p = figure(title=title, background_fill_color="#fafafa",
                   y_axis_type="log")
        p.quad(top=self.counts, bottom=1, left=x[:-1], right=x[1:],
                   fill_color="#036564", line_color="#033649")
        if self.calibrated:
            p.xaxis.axis_label = 'Energy (keV)'
        else:
            p.xaxis.axis_label = 'Channel'
        p.yaxis.axis_label = 'Counts'
        output_file("{0}.html".format(title))
        if len(self.peaks) > 0:
            for peak in self.peaks:
                p.quad(top=self.counts[peak.centroid], bottom=self.counts[peak.left-10], left=x[peak.left], right=x[peak.right],
                       fill_color="#00FF00")
        show(p)

    def __add__(self, other):
        new = copy.deepcopy(self)
        new.counts += other.counts
        return new

    def __sub__(self, other):
        new = copy.deepcopy(self)
        new.counts -= other.counts
        return new

    def find_peaks(self):
        self.peaks = []
        peaks, props = sig.find_peaks(self.counts, prominence = 200)
        widths, width_heights, left_ips, right_ips = sig.peak_widths(self.counts, peaks)
        for i in range(len(peaks)):
            centroid = peaks[i]
            left_th = int(left_ips[i])
            right_th = int(right_ips[i])
            self.peaks += [Peak(centroid, int(self.counts[centroid]), left_th, right_th)]
        return self.peaks

    def calibrate(self, source_list, auto_calibrate = True):
        gamma_energies = []
        gamma_intensities = []
        for source in source_list:
            for i in range(len(source.energies)):
                gamma_energies += [source.energies[i]]
                gamma_intensities += [source.activity * source.intensities[i]]
        source_data = np.transpose(np.array((gamma_energies, gamma_intensities)))
        source_data = source_data[source_data[:, 0].argsort()]
        calibration_channels = []
        calibration_energies = []
        if auto_calibrate:
            self.find_peaks()
            for i in range(len( source_data[:,0] ) ):
                calibration_channels += [self.peaks[i].centroid]
                calibration_energies += [source_data[i,0]]
        else:
            print("Opening plot for manual calibration...")
            self.show_histogram("Calibration")
            for i in range(len(source_data[:, 0])):
                current_energy = [source_data[i,0]]
                calibration_energies += [current_energy]
                prompt = "What channel do you think the {0} keV gamma is in? ".format(current_energy)
                calibration_channels += [int(input(prompt))]
        coefficients = np.polyfit(calibration_channels, calibration_energies, 2)
        self.energies = np.polyval(coefficients, self.channels)
        print("Energy(c) = {0}c^2 + {1}c + {2}".format( *np.round(coefficients,4) ) )
        self.calibrated = True
        self.calibration_coeffs = coefficients
        return coefficients


class Peak:
    def __init__(self, centroid, value, left, right):
        self.centroid = centroid
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self):
        return "{0} count peak at {1} between {2} and {3}".format(
            self.value, self.centroid, self.left, self.right)


class Source:
    def __init__(self, activity, energies, intensities):
        self.activity = activity
        self.energies = energies
        self.intensities = intensities


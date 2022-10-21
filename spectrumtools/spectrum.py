import numpy as np
from bokeh.io import show, output_file
from bokeh.plotting import figure
import matplotlib.pyplot as plt
from bokeh.models import HoverTool
import scipy.signal as sig
import copy


class Spectrum:
    def __init__(self, counts):
        self.num_channels = len(counts)
        self.counts = counts
        self.channels = np.arange(1, self.num_channels+1)
        self.energies = self.channels
        self.peaks = []
        self.calibrated = False
        self.calibration_coeffs = [1.0]
        self.calibration_channels = []
        self.calibration_energies = []

    def show_histogram(self, title, show_peaks = False):
        if self.calibrated:
            x = np.concatenate((np.array([0.0]),self.energies))
        else:
            x = np.concatenate((np.array([0]),self.channels))
        TOOLTIPS = [
            ("(x,y)", "($x, $y)"),
        ]

        p = figure(title=title, background_fill_color="#fafafa",
                   y_axis_type="log", width=1200, height=700,
                   x_range=(0, np.max(self.energies)),
                   tooltips=TOOLTIPS, tools = "pan,wheel_zoom,box_zoom,reset")
        p.add_tools(HoverTool(mode='vline'))
        p.quad(top=self.counts, bottom=1, left=x[:-1], right=x[1:],
                   fill_color="#036564", line_color="#033649")
        if self.calibrated:
            p.xaxis.axis_label = 'Energy (keV)'
        else:
            p.xaxis.axis_label = 'Channel'
        p.yaxis.axis_label = 'Counts'
        output_file("{0}.html".format(title))
        if len(self.peaks) > 0 and show_peaks:
            for peak in self.peaks:
                p.quad(top=self.counts[peak.centroid], bottom=self.counts[peak.left], left=x[peak.left], right=x[peak.right],
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

    def find_peaks(self, auto_prominence, peak_min_distance=20):
        self.peaks = []
        peaks, props = sig.find_peaks(self.counts, prominence = auto_prominence, distance = peak_min_distance)
        widths, width_heights, left_ips, right_ips = sig.peak_widths(self.counts, peaks, rel_height=0.5)
        for i in range(len(peaks)):
            centroid = peaks[i]
            left_th = int(left_ips[i])
            right_th = int(right_ips[i])
            width = widths[i]
            self.peaks += [Peak(centroid, int(self.counts[centroid]), left_th, right_th)]
        return self.peaks

    def calibrate(self, gamma_energies, known_channels = [], auto_calibrate = False, reset_calibration = False,
                  auto_prominence = 100, show_fit = False):
        if not reset_calibration:
            self.calibration_energies = []
            self.calibration_channels = []
        if auto_calibrate:
            self.find_peaks(auto_prominence)
            while len(self.peaks) < len(gamma_energies):
                auto_prominence -= 10
                self.find_peaks(auto_prominence)
            print(self.peaks)
            for i in range(len( gamma_energies ) ):
                self.calibration_channels += [self.peaks[i].centroid]
                self.calibration_energies += [gamma_energies[i]]
        elif len(known_channels) > 0:
            for i in range(len( gamma_energies ) ):
                self.calibration_channels += [known_channels[i]]
                self.calibration_energies += [gamma_energies[i]]
        else:
            print("Opening plot for manual calibration...")
            self.show_histogram("Calibration")
            for i in range(len(gamma_energies)):
                current_energy = gamma_energies[i]
                self.calibration_energies += [current_energy]
                prompt = "What channel do you think the {0} keV gamma is in? ".format(current_energy)
                self.calibration_channels += [int(input(prompt))]
        coefficients = np.polyfit(self.calibration_channels, self.calibration_energies, 2)
        self.energies = np.polyval(coefficients, self.channels)
        calibration_func = "Energy(c) = {0}c^2 + {1}c + {2}".format( *np.round(coefficients,4) )
        print(calibration_func)
        if show_fit:
            plt.figure()
            plt.plot(self.calibration_channels, self.calibration_energies, 'k.')
            plt.plot(self.channels, self.energies)
            plt.xlabel("Channel")
            plt.ylabel("Energy (keV)")
            plt.show()
        self.calibrated = True
        self.calibration_coeffs = coefficients
        return coefficients

    def calc_FWHMs(self):
        peak_energies = []
        peak_fwhm_vals = []
        for peak in self.peaks:
            peak_energy, peak_fwhm = peak.get_FWHM(self.calibration_coeffs)
            peak_energies += [peak_energy]
            peak_fwhm_vals += [peak_fwhm]
        return (peak_energies, peak_fwhm_vals)


class Peak:
    def __init__(self, centroid, value, left, right):
        self.centroid = centroid
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self):
        return "{0} count peak at {1} between {2} and {3}".format(
            self.value, self.centroid, self.left, self.right)

    def get_FWHM(self, coefficients):
        left_energy = np.polyval(coefficients, self.left)
        right_energy = np.polyval(coefficients, self.right)
        self.FWHM = right_energy-left_energy
        return (self.centroid, self.FWHM)



class Source:
    def __init__(self, activity, energies, intensities):
        self.activity = activity
        self.energies = energies
        self.intensities = intensities
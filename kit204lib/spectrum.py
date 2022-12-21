import numpy as np
from bokeh.io import show, output_file
from bokeh.plotting import figure
import matplotlib.pyplot as plt
from bokeh.models import HoverTool
import scipy.signal as sig
from scipy.optimize import curve_fit
import copy

def gauss(x, A, center, sigma, floor):
    return A * np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) + floor

class Spectrum:
    def __init__(self, counts):
        self.num_channels = len(counts)
        self.counts = counts
        self.channels = np.arange(1, self.num_channels+1)
        self.energies = self.channels
        self.peaks = []
        self.peak_channels = np.array(0.0)
        self.peak_energies = np.array(0.0)
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
                p.line(peak.energies, peak.fit, line_color="#00FF00", line_width=2)
        show(p)

    def __add__(self, other):
        new = copy.deepcopy(self)
        new.counts += other.counts
        return new

    def __sub__(self, other):
        new = copy.deepcopy(self)
        new.counts -= other.counts
        return new

    def find_peaks(self, auto_prominence, num_peaks, starting_channel=0, peak_min_distance=20):
        self.peak_channels = []
        peaks, props = sig.find_peaks(self.counts, prominence = auto_prominence, distance = peak_min_distance)
        self.peak_channels = np.array(peaks)
        return self.peak_channels

    def fit_peaks(self, fit_width=6, do_plot=False):
        num_peaks = len(self.peak_energies)
        self.peaks = [None]*num_peaks
        peak_energies = np.zeros(num_peaks)
        peak_fwhms = np.zeros(num_peaks)
        for i in range(num_peaks):
            peak_energy = self.peak_energies[i]
            subset = np.argwhere(np.abs(self.energies - peak_energy) < fit_width)[:,0]
            energy_set = self.energies[subset]
            peak_counts = self.counts[subset]
            center0 = peak_energy

            A0 = np.max(peak_counts)
            sigma0 = 3.0
            floor0 = np.mean(peak_counts[0:3])
            parameters, covariance = curve_fit(gauss, energy_set, peak_counts, p0=(A0, center0, sigma0, floor0))
            A, x0, sigma, floor = parameters
            x_vals = np.linspace(energy_set[0], energy_set[-1], np.int(fit_width)*150)
            gaussian_fit = gauss(x_vals, A, x0, sigma, floor)
            height = np.max(gaussian_fit) - floor
            fwhm_height = height + floor
            half_height = fwhm_height / 2 + floor
            idx = np.argwhere(np.diff(np.sign(gaussian_fit - half_height))).flatten()
            fwhm = x_vals[idx[1]] - x_vals[idx[0]]
            counts = sum(peak_counts - floor)
            if do_plot:
                plt.figure()
                plt.plot(energy_set, peak_counts)
                plt.plot(x_vals, gaussian_fit)
                plt.plot(x_vals[idx], gaussian_fit[idx], 'r-')
                plt.show()
            self.peaks[i] = Peak(peak_energy, x_vals, gaussian_fit, parameters, fwhm, counts)
            peak_energies[i] = peak_energy
            peak_fwhms[i] = fwhm
        return peak_energies, peak_fwhms

    def calibrate(self, gamma_energies, starting_channel=0, known_channels = [], auto_calibrate = False, reset_calibration = False,
                  auto_prominence = 100, show_fit = False):
        if not reset_calibration:
            self.calibration_energies = []
            self.calibration_channels = []
        if auto_calibrate:
            peak_channels = self.find_peaks(auto_prominence, len(gamma_energies), starting_channel)
            while len(self.peak_channels) < len(gamma_energies):
                auto_prominence -= 10
                peak_channels = self.find_peaks(auto_prominence, len(gamma_energies), starting_channel)
            print("Identified peaks in channels: {0}".format(peak_channels))
            for i in range(len( gamma_energies ) ):
                self.calibration_channels += [peak_channels[i]]
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
        self.peak_energies = np.polyval(coefficients, self.peak_channels)
        self.calibrated = True
        self.calibration_coeffs = coefficients
        calibration_func = "Energy(c) = {0}c^2 + {1}c + {2}".format( *np.round(coefficients,4) )
        print(calibration_func)
        if show_fit:
            plt.figure()
            plt.plot(self.calibration_channels, self.calibration_energies, 'k.')
            plt.plot(self.channels, self.energies)
            plt.xlabel("Channel")
            plt.ylabel("Energy (keV)")
            plt.show()
        return coefficients


class Peak:
    def __init__(self, centroid, energies, gaussian_fit, fit_parameters, fwhm, counts):
        self.centroid = centroid
        self.energies = energies
        self.fit = gaussian_fit
        self.A = fit_parameters[0]
        self.x0 = fit_parameters[1]
        self.sigma = fit_parameters[2]
        self.floor = fit_parameters[3]
        self.fwhm = fwhm
        self.counts = counts


    def __repr__(self):
        return "{0} peak keV with {1} keV FWHM".format(
            self.centroid, self.fwhm)





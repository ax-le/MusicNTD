# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 17:33:39 2019

@author: amarmore
"""
import context

import current_plot
import soundfile as sf
import numpy as np

from constants import Constants
from scipy import signal
from sklearn import preprocessing


class STFT:
    """ A class containing the stft coefficients and important values reltaed to the STFT of a signal """
    
    def __init__(self, path, time = None, channel = 0, plotting = False):
        """STFT of a temporal signal, given a path"""
        # For now, this function returns the stft of only one channel        
        the_signal, sampling_rate_local = sf.read(path)
        if time != None:
            the_signal = the_signal[0:time*sampling_rate_local,:]
        if plotting:
            current_plot.plot_signal_in_a_range(np.arange(0, the_signal[:,channel].size/sampling_rate_local, 1/sampling_rate_local), the_signal[:,channel])
        frequencies, time_atoms, coeff =  signal.stft(the_signal[:,channel], fs=sampling_rate_local, nperseg = int(sampling_rate_local*Constants.TEMPORAL_FRAME_SIZE), nfft=int(sampling_rate_local*Constants.TEMPORAL_FRAME_SIZE))
        self.time_bins = time_atoms
        self.freq_bins = frequencies
        self.sampling_rate = sampling_rate_local
        self.stft_coefficients = coeff
    
    #Most important spectrograms
    
    def get_magnitude_spectrogram(self, threshold = None):
        if threshold == None:
            return np.abs(self.stft_coefficients)
        else:
            spec = np.abs(self.stft_coefficients)
            spec[spec < threshold] = 0
            
            # Other version, potentially helpful
            #spec = np.where(spec < np.percentile(spec, 99), 0, spec) # Forcing saprsity by keeping only the highest values

            return spec
        
    
    def get_power_spectrogram(self, threshold = None):
        if threshold == None:
            return np.abs(self.stft_coefficients)**2
        else:
            spec = np.abs(self.stft_coefficients)**2
            spec[spec < threshold] = 0
            return spec
        
    def get_log_spectrogram(self, threshold = None):
        log_coefficients = 20*np.log10(np.abs(self.stft_coefficients) + 10e-10) # Avoid the case of -infinity in the log with 0 value
        return preprocessing.minmax_scale(log_coefficients, feature_range=(0, 100)) # Rescalling values (for nonnegativity)"""
    
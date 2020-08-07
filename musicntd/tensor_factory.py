# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:11:12 2020

@author: amarmore
"""
import tensorly as tl
import numpy as np
import warnings
import musicntd.data_manipulation as dm
import musicntd.model.errors as err


def tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision, midi = False):
    """
    Returns a tensor-spectrogram from a spectrogram and bars starts and ends.
    Each bar of the tensor_spectrogram will contain the same number of frames, define by the "subdivision" parameter.
    These frames are selected from an over-sampled spectrogram, to adapt to the specific size of each bar.

    Parameters
    ----------
    spectrogram : list of list of floats or numpy array
        The spectrogram to return as a tensor-spectrogram.
    bars : list of tuples
        List of the bars (start, end), in seconds, to cut the spectrogram at bar delimitation.
    hop_length_seconds : float
        The hop_length, in seconds.
    subdivision : integer
        The number of subdivision of the bar to be contained in each slice of the tensor.
    midi : boolean, optional
        A boolean to know if the spectrogram is in midi.
        If it is, adds a correction to deletes void bars.
        The default is False.

    Returns
    -------
    tensorly tensor
        The tensor-spectrogram as a tensorly tensor.

    """
    freq_len = spectrogram.shape[0]
    hop = int(hop_length_seconds*44100)
    if hop != 32:
        print("hop_length a 44100Hz = " + str(hop) + ", normal ?")
    bars_idx = dm.segments_from_time_to_frame_idx(bars[1:], hop_length_seconds)
    #if hop == 512:
        #raise NotImplementedError("Probably wrong hop here, to debug")
    samples_init = [int(round(bars_idx[0][0] + k * (bars_idx[0][1] - bars_idx[0][0])/subdivision)) for k in range(subdivision)]
    
    if midi:
        raise err.OutdatedBehaviorException("Should'nt be used, still bugged")
        
    tens = np.array(spectrogram[:,samples_init]).reshape(freq_len, subdivision, 1)
    #tens = padded_tens_with_zeros(tens_init, longest_bar)
    
    for bar in bars_idx[1:]:
        t_0 = bar[0]
        t_1 = bar[1]
        samples = [int(round(t_0 + k * (t_1 - t_0)/subdivision)) for k in range(subdivision)]
        if samples[-1] < spectrogram.shape[1]:
            current_bar_tensor_spectrogram = spectrogram[:,samples].reshape(freq_len, subdivision,1)
            tens = np.append(tens, current_bar_tensor_spectrogram, axis = 2)
        else:
            break
    
    return tl.tensor(tens, dtype=tl.float32)

# %% OUTDATED BEHVAIOR
def padded_tens_with_zeros(tens, longest_bar_len): #, freq_dim):
    """
    DEPRECATED
    Pad this tensor (a single matrix interpreted as a tensor slice, so a tensor with third mode) with zeros, 
    for it to be of the same size as every bar.

    Parameters
    ----------
    tens : numpy array
        The current matrix, containing spectrogram of a single bar, interpreted as a tensor (3rd dimension of size 1).
    longest_bar_len : integer
        Size of the longest bar, also the desired size for the tensor to return.

    Returns
    -------
    numpy array
        The tensor, padded with zero, for its 2nd dimension to be of the desired size.

    """
    raise err.OutdatedBehaviorException("Padding the tensor with zeroes correspond to an outdated behvaior, shouldn't be invoked.")
    return np.concatenate((tens, np.zeros(tens.shape[0] * (longest_bar_len - tens.shape[1])).reshape(tens.shape[0], (longest_bar_len - tens.shape[1]), 1)), axis = 1)

def longest_bar_len(bars_in_frames):
    """
    DEPRECATED
    Returns the size of the longest bar among all, in number of frames.

    Parameters
    ----------
    bars_in_frames : list of tuples of integers
        The bars, as tuples (start, end), in number of frames.

    Returns
    -------
    max_len : integer
        The size of the longest bar.
    """
    max_len = 0
    for bar in bars_in_frames:
        if bar[1] - bar[0] > max_len:
            max_len = bar[1] - bar[0]
    return max_len

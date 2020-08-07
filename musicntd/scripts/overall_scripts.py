# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:34:39 2020

@author: amarmore
"""

# A module containing some high-level scripts for decomposition and/or segmentation.

import soundfile as sf
import librosa
import tensorly as tl
import os
import numpy as np
import pathlib

import nn_fac.ntd as NTD

import musicntd.autosimilarity_segmentation as as_seg
import musicntd.data_manipulation as dm
import musicntd.model.features as features
import musicntd.model.errors as err
from musicntd.model.current_plot import *

def load_RWC_dataset(music_folder_path, annotations_type = "MIREX10", desired_format = None, downbeats = None):
    """
    Load the data on the RWC dataset, ie path of songs and annotations.
    The annotations can be either AIST or MIREX 10.

    Parameters
    ----------
    music_folder_path : String
        Path of the folder to parse.
    annotations_type : "AIST" [1] or "MIREX10" [2]
        The type of annotations to load (both have a specific behavior and formatting)
        The default is "MIREX10"
    desired_format : DEPRECATED
    downbeats : DEPRECATED

    Raises
    ------
    NotImplementedError
        If the format is not taken in account.

    Returns
    -------
    numpy array
        list of list of paths, each sublist being of the form [song, annotations, downbeat(if specified)].
        
    References
    ----------
    [1] Goto, M. (2006, October). AIST Annotation for the RWC Music Database. In ISMIR (pp. 359-360).
    
    [2] Bimbot, F., Sargent, G., Deruty, E., Guichaoua, C., & Vincent, E. (2014, January). 
    Semiotic description of music structure: An introduction to the Quaero/Metiss structural annotations.

    """
    if downbeats != None or desired_format != None:
        raise err.OutdatedBehaviorException("This version of loading is deprecated.")
    # Load dataset paths at the format "song, annotations, downbeats"
    paths = []
    for file in os.listdir(music_folder_path):
        if file[-4:] == ".wav":
            file_number = "{:03d}".format(int(file[:-4]))
            ann = dm.get_annotation_name_from_song(file_number, annotations_type)
            paths.append([file, ann])
    return np.array(paths)

# %% Loading or persisting bars, spectrograms and NTD computation
def load_or_save_bars(persisted_path, song_path):
    """
    Computes the bars for this song, or load them if they were already computed.

    Parameters
    ----------
    persisted_path : string
        Path where the bars should be found.
    song_path : string
        The path of the signal of the song.

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    """
    song_name = song_path.split("\\")[-1].replace(".wav","")
    try:
        bars = np.load("{}\\bars\\{}.npy".format(persisted_path, song_name))
    except:
        bars = dm.get_bars_from_audio(song_path)
        np.save("{}\\bars\\{}".format(persisted_path, song_name), bars)
    return bars

def load_bars(persisted_path, song_name):
    """
    Loads the bars for this song, which were persisted after a first computation.

    Parameters
    ----------
    persisted_path : string
        Path where the bars should be found.
    song_name : string
        Name of the song (identifier of the bars to load).

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    """
    raise err.OutdatedBehaviorException("You should use load_or_save_bars(persisted_path, song_path) instead, as it handle the fact that bars weren't computed yet.")
    bars = np.load("{}\\bars\\{}.npy".format(persisted_path, song_name))
    return bars
    
def load_or_save_spectrogram(persisted_path, song_path, feature, hop_length, fmin = 98):
    """
    Computes the spectrogram for this song, or load it if it were already computed.

    Parameters
    ----------
    persisted_path : string
        Path where the spectrogram should be found.
    song_path : string
        The path of the signal of the song.
    feature : string
        Feature of the spectrogram, part of the identifier of the spectrogram.
    hop_length : integer
        hop_length of the spectrogram, part of the identifier of the spectrogram.
    fmin : integer
        Minimal frequence for the spectrogram, part of the identifier of the spectrogram.
        The default is 98.

    Returns
    -------=
    spectrogram : numpy array
        The pre-computed spectorgram.
    """
    song_name = song_path.split("\\")[-1].replace(".wav","")
    try:
        spectrogram = np.load("{}\\spectrograms\\{}_{}_stereo_{}_{}.npy".format(persisted_path, song_name, feature, hop_length, fmin))
    except:
        the_signal, original_sampling_rate = sf.read(song_path)
        if original_sampling_rate != 44100:
            the_signal = librosa.core.resample(np.asfortranarray(the_signal), original_sampling_rate, 44100)
        if feature == "tonnetz":
            hop_length = "fixed"
            fmin = "fixed"
        if feature == "pcp_tonnetz":
            # If chromas are already computed, try to load them instead of recomputing them.
            chromas = load_or_save_spectrogram(persisted_path, song_path, "pcp", hop_length, fmin = fmin)
            spectrogram = librosa.feature.tonnetz(y=None, sr = None, chroma = chromas)
            np.save("{}\\spectrograms\\{}_{}_stereo_{}_{}".format(persisted_path, song_name, feature, hop_length, fmin), spectrogram)
            return spectrogram
        # If it wasn't pcp_tonnetz, compute the spectrogram, and then save it.
        spectrogram = features.get_spectrogram(the_signal, 44100, feature, hop_length, fmin = fmin)
        np.save("{}\\spectrograms\\{}_{}_stereo_{}_{}".format(persisted_path, song_name, feature, hop_length, fmin), spectrogram)
    return spectrogram

def load_or_save_spectrogram_and_bars(persisted_path, song_path, feature, hop_length, fmin = 98):
    """
    Loads the spectrogram and the bars for this song, which were persisted after a first computation.

    Parameters
    ----------
    persisted_path : string
        Path where the bars and the spectrogram should be found.
    song_path : string
        The path of the signal of the song.
    feature : string
        Feature of the spectrogram, part of the identifier of the spectrogram.
    hop_length : integer
        hop_length of the spectrogram, part of the identifier of the spectrogram.
    fmin : integer
        Minimal frequence for the spectrogram, part of the identifier of the spectrogram.
        The default is 98.

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    spectrogram : numpy array
        The pre-computed spectorgram.
    """
    bars = load_or_save_bars(persisted_path, song_path)
    spectrogram = load_or_save_spectrogram(persisted_path, song_path, feature, hop_length, fmin = fmin)
    return bars, spectrogram


def load_spectrogram_and_bars(persisted_path, song_name, feature, hop_length, fmin = 98):
    """
    Loads the spectrogram and the bars for this song, which were persisted after a first computation.

    Parameters
    ----------
    persisted_path : string
        Path where the bars and the spectrogram should be found.
    song_name : string
        Name of the song (identifier of the bars to load).
    feature : string
        Feature of the spectrogram, part of the identifier of the spectrogram.
    hop_length : integer
        hop_length of the spectrogram, part of the identifier of the spectrogram.
    fmin : integer
        Minimal frequence for the spectrogram, part of the identifier of the spectrogram.
        The default is 98.

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    spectrogram : numpy array
        The pre-computed spectorgram.
    """
    raise err.OutdatedBehaviorException("You should use load_or_save_spectrogram_and_bars(persisted_path, song_path, feature, hop_length, fmin) instead, as it handle the fact that bars weren't computed yet.")
    bars = np.load("{}\\bars\\{}.npy".format(persisted_path, song_name))
    spectrogram = np.load("{}\\spectrograms\\{}_{}_stereo_{}_{}".format(persisted_path, song_name, feature, hop_length, fmin))
    return bars, spectrogram

def NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "chromas"): 
    """
    Computes the NTD from the tensor_spectrogram and with specified ranks.
    On the first hand, if the NTD is persisted, it will load and return its results.
    If it's not, it will compute the NTD, store it, and return it.

    Parameters
    ----------
    persisted_path : String
        Path of the persisted decompositions and bars.
    persisted_arguments : String
        Identifier of the specific NTD to load/save.
    tensor_spectrogram : tensorly tensor
        The tensor to decompose.
    ranks : list of integers
        Ranks of the decomposition.
    init : String, optional
        The type of initialization of the NTD.
        See the NTD module to have more information regarding initialization.
        The default is "chromas",
        meaning that the first factor will be set to the 12-size identity matrix,
        and the other factors will be initialized by HOSVD.

    Raises
    ------
    NotImplementedError
        Errors in the arguments.

    Returns
    -------
    core : tensorly tensor
        The core of the decomposition.
    factors : numpy array
        The factors of the decomposition.

    """
    path_for_ntd = "{}\\ntd\\{}_{}_{}".format(persisted_path, ranks[0], ranks[1], ranks[2])
    if "512" in persisted_arguments:
        raise NotImplementedError("Probably an error in the code, as old hop_length seems to be passed")
    if persisted_arguments[-2:] == "32":
        raise NotImplementedError("Probably an error in the code, as the hop_length seems to be passed")
    try:
        a_core_path = "{}\\core{}.npy".format(path_for_ntd, persisted_arguments)
        a_core = np.load(a_core_path)
        a_factor_path = "{}\\factors{}.npy".format(path_for_ntd, persisted_arguments)
        a_factor = np.load(a_factor_path, allow_pickle=True)
        return a_core, a_factor
    except FileNotFoundError:
        core, factors = NTD.ntd(tensor_spectrogram, ranks = ranks, init = init, verbose = False, hals = False,
                            sparsity_coefficients = [None, None, None, None], normalize = [True, True, False, True], mode_core_norm = 2,
                            deterministic = True)
        
        pathlib.Path(path_for_ntd).mkdir(parents=True, exist_ok=True)
    
        core_path = "{}\\core{}".format(path_for_ntd, persisted_arguments)
        np.save(core_path, core)
        factors_path = "{}\\factors{}".format(path_for_ntd, persisted_arguments)
        np.save(factors_path, factors)
        return core, factors

# %% Outdated and deprecated functions

def run_results_on_signal(tensor_spectrogram, reference_segments, bars, plotting = True):
    """
    Compute all desired results on the signal (precision, recall and f measure).
    The window_length is set to 0.5 seconds.

    Parameters
    ----------
    tensor_spectrogram : tensorly tensor
        The tensor_spectrogram of the song.
    references_segments : list of tuples
        The segments from the annotations.
    bars : list of tuple of float
        The bars of the song.
    plotting : boolean, optional
        A boolean whether to plot the autosimilarity with the segmentation or not.
        The default is True.

    Returns
    -------
    all_res : nested list
        Scores of all computation.

    """
    unfolded = tl.unfold(tensor_spectrogram, 2)
    if plotting:
        plot_spec_with_annotations_abs_ord(as_seg.get_autosimilarity(unfolded, transpose = True, normalize = True), dm.frontiers_from_time_to_bar(dm.segments_to_frontiers(reference_segments), bars), cmap = cm.Greys)

    return compute_all_results(unfolded, reference_segments, bars, window_length = 0.5), compute_all_results(unfolded, reference_segments, bars, window_length = 3)

def run_NTD_on_this_song(persisted_path, persisted_arguments, tensor_spectrogram, ranks, reference_segments, bars, init = "tucker", plotting = True):
    """
    Computes (or load) a NTD and returns segmentation scores associated with it.
    Segmentation techniques are probably outdated though.

    Parameters
    ----------
    persisted_path : String
        Path of the persisted decompositions and bars.
    persisted_arguments : String
        Identifier of the specific NTD to load/save.
    tensor_spectrogram : tensorly tensor
        The tensor to decompose.
    ranks : list of integers
        Ranks of this decomposition.
    reference_segments : list of tuple of floats
        Annotated segments.
    bars : list of tuple of floats
        Bars of the songs.
    init : String, optional
        The type of initialization of the NTD.
        See the NTD module to have more information regarding initialization.
        The default is "tucker",
        meaning that the factors will be initialized by HOSVD.
    plotting : boolean, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    tuple of lists
        Nested list of segmentation scores at 0.5 and 3 seconds.

    """
    core, factors = NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = init)

    if plotting:
        plot_spec_with_annotations_abs_ord(as_seg.get_autosimilarity(factors[2], transpose = True, normalize = True), dm.frontiers_from_time_to_bar(dm.segments_to_frontiers(reference_segments), bars), cmap = cm.Greys)

    return compute_all_results(factors[2], reference_segments, bars, window_length = 0.5), compute_all_results(factors[2], reference_segments, bars, window_length = 3)

def segmentation_from_c_factor(c_factor, normalize_autosimil = False, segmentation = "expanding_mixed"):
    """
    Segmenting the C factor (Q matrix, the third one of the decomposition) by computing its autosimilarity.
    
    This function is now deprecated as the segmentation measures are old and have been updated since.

    Parameters
    ----------
    c_factor : numpy array
        The third factor of the decomposition (for our case).
    normalize_autosimil : boolean, optional
        Whether to normalize the autosimilarity matrix or not.
        The default is False.
    segmentation : String, optional
        Type of the segmentation:
            - novelty: novelty segmentation
            - expanding_convolution: segmentation based on the dynamic convolutionnal cost programming algorithm
            - expanding_mixed: segmentation based on the dynamic convolutionnal cost programming algorithm with novelty penalty on each point (mixed algorithm)
        The default is "expanding_mixed".

    Raises
    ------
    NotImplementedError
        If a non specified type of segmentation is chosen.

    Returns
    -------
    list of tuples
        The segmentation of the autosimilarity matrix.

    """
    autosimilarity = as_seg.get_autosimilarity(c_factor, transpose = True, normalize = normalize_autosimil)
    if segmentation == "novelty":
        ker_size = 16
        novelty = as_seg.novelty_computation(autosimilarity, ker_size)
        ends = as_seg.select_highest_peaks_thresholded_indexes(novelty, percentage = 0.22)
        ends.append(len(autosimilarity) - 1)
        return dm.frontiers_to_segments(ends)
    if segmentation == "novelty_max_slope":
        ker_size = 16
        novelty = as_seg.novelty_computation(autosimilarity, ker_size)
        ponderated_novelty = as_seg.values_as_slop(novelty, choice_func = max)
        ends = as_seg.select_highest_peaks_thresholded_indexes(ponderated_novelty, percentage = 0.08)
        ends.append(len(autosimilarity) - 1)
        return dm.frontiers_to_segments(ends)
    if segmentation == "novelty_min_slope":
        ker_size = 16
        novelty = as_seg.novelty_computation(autosimilarity, ker_size)
        ponderated_novelty = as_seg.values_as_slop(novelty, choice_func = min)
        ends = as_seg.select_highest_peaks_thresholded_indexes(ponderated_novelty, percentage = 0.16)
        ends.append(len(autosimilarity) - 1)
        return dm.frontiers_to_segments(ends)
    if segmentation == "novelty_mean_slope":
        ker_size = 16
        novelty = as_seg.novelty_computation(autosimilarity, ker_size)
        ponderated_novelty = as_seg.values_as_slop(novelty, choice_func = as_seg.mean)
        ends = as_seg.select_highest_peaks_thresholded_indexes(ponderated_novelty, percentage = 0.13)
        ends.append(len(autosimilarity) - 1)
        return dm.frontiers_to_segments(ends)
    elif segmentation == "expanding_convolution":
        return as_seg.dynamic_convolution_computation(autosimilarity, mix = 1)[0]
    elif segmentation == "expanding_mixed":
        return as_seg.dynamic_convolution_computation(autosimilarity, mix = 0.5)[0]
    else:
        raise NotImplementedError("Other types are not implemtend yet")
        
def compute_all_results(c_factor, references_segments, bars, window_length = 0.5):
    """
    Compute all desired results (precision, recall and f measure, encapsulated by the data_manipulation compute_score_of_segmentation() function).
    
    This function is now deprecated as the segmentation measures are old and have been updated since.

    Parameters
    ----------
    c_factor : numpy array
        The third factor of the NTD decomposition.
    references_segments : list of tuples
        The segments from the annotations.
    bars : list of tuple of float
        The bars of the song.
    window_length : float, optional
        The window tolerance for the frontiers. The default is 0.5.

    Returns
    -------
    all_res : nested list
        Scores of all computation.

    """
    all_res = []
        
    # Novelty scores
    # novelty_segments = segmentation_from_c_factor(c_factor, normalize_autosimil=False, segmentation="novelty")
    # novelty_segments_in_time = dm.segments_from_bar_to_time(novelty_segments, dbt)
    # all_res.append(dm.compute_score_of_segmentation(references_segments, novelty_segments_in_time, window_length=window_length))
    
    novelty_segments_on_normalized = segmentation_from_c_factor(c_factor, normalize_autosimil=True, segmentation="novelty")
    novelty_segments_on_normalized_in_time = dm.segments_from_bar_to_time(novelty_segments_on_normalized, bars)
    all_res.append(dm.compute_score_of_segmentation(references_segments, novelty_segments_on_normalized_in_time, window_length=window_length))

    # New novelty peak picking
    # ponderated_novelty_segments = segmentation_from_c_factor(c_factor, normalize_autosimil=False, segmentation="novelty_max_slope")
    # ponderated_novelty_segments_in_time = dm.segments_from_bar_to_time(ponderated_novelty_segments, dbt)
    # all_res.append(dm.compute_score_of_segmentation(references_segments, ponderated_novelty_segments_in_time, window_length=window_length))
    
    ponderated_novelty_segments_on_normalized = segmentation_from_c_factor(c_factor, normalize_autosimil=True, segmentation="novelty_max_slope")
    ponderated_novelty_segments_on_normalized_in_time = dm.segments_from_bar_to_time(ponderated_novelty_segments_on_normalized, bars)
    all_res.append(dm.compute_score_of_segmentation(references_segments, ponderated_novelty_segments_on_normalized_in_time, window_length=window_length))
    
    ponderated_novelty_segments_on_normalized_2 = segmentation_from_c_factor(c_factor, normalize_autosimil=True, segmentation="novelty_min_slope")
    ponderated_novelty_segments_on_normalized_in_time_2 = dm.segments_from_bar_to_time(ponderated_novelty_segments_on_normalized_2, bars)
    all_res.append(dm.compute_score_of_segmentation(references_segments, ponderated_novelty_segments_on_normalized_in_time_2, window_length=window_length))
    
    ponderated_novelty_segments_on_normalized_3 = segmentation_from_c_factor(c_factor, normalize_autosimil=True, segmentation="novelty_mean_slope")
    ponderated_novelty_segments_on_normalized_in_time_3 = dm.segments_from_bar_to_time(ponderated_novelty_segments_on_normalized_3, bars)
    all_res.append(dm.compute_score_of_segmentation(references_segments, ponderated_novelty_segments_on_normalized_in_time_3, window_length=window_length))
    # Convolution scores
    # conv_segments = segmentation_from_c_factor(c_factor, normalize_autosimil=False, segmentation="expanding_convolution")
    # conv_segments_in_time = dm.segments_from_bar_to_time(conv_segments, dbt)
    # all_res.append(dm.compute_score_of_segmentation(references_segments, conv_segments_in_time, window_length=window_length))
    
    conv_segments_on_normalized = segmentation_from_c_factor(c_factor, normalize_autosimil=True, segmentation="expanding_convolution")
    conv_segments_on_normalized_in_time = dm.segments_from_bar_to_time(conv_segments_on_normalized, bars)
    all_res.append(dm.compute_score_of_segmentation(references_segments, conv_segments_on_normalized_in_time, window_length=window_length))
  
    # Mixed scores
    # conv_and_novelty_segments = segmentation_from_c_factor(c_factor, normalize_autosimil=False, segmentation="expanding_mixed")
    # conv_and_novelty_segments_in_time = dm.segments_from_bar_to_time(conv_and_novelty_segments, dbt)
    # all_res.append(dm.compute_score_of_segmentation(references_segments, conv_and_novelty_segments_in_time, window_length=window_length))
    
    conv_and_novelty_segments_on_normalized = segmentation_from_c_factor(c_factor, normalize_autosimil=True, segmentation="expanding_mixed")
    conv_and_novelty_segments_on_normalized_in_time = dm.segments_from_bar_to_time(conv_and_novelty_segments_on_normalized, bars)
    all_res.append(dm.compute_score_of_segmentation(references_segments, conv_and_novelty_segments_on_normalized_in_time, window_length=window_length))
    
    return all_res
      
# def old_and_for_midi_load_RWC_dataset(folder_path, desired_format = "wav", downbeats = True):
#     """
#     Load the data on the RWC dataset, ie path of songs and annotations.

#     Parameters
#     ----------
#     folder_path : String
#         Path of the folder to parse.
#     desired_format : String, optional
#         The format of song files to study.
#         The default is "wav".
#     downbeats : boolean, optional
#         Whether to list the paths of downbeats or not.
#         The default is True.

#     Raises
#     ------
#     NotImplementedError
#         If the format is not taken in account.

#     Returns
#     -------
#     numpy array
#         list of list of paths, each sublist being of the form [song, annotations, downbeat(if specified)].

#     """
#     # Load dataset paths at the format "song, annotations, downbeats"
#     paths = []
#     if desired_format == "mid":
#         for file in os.listdir(folder_path):
#             if file[-4:] == ".mid":
#                 if downbeats == True:
#                     paths.append([file, file.replace("mid", "chorus.txt").upper(), file.replace("mid", "beat.txt").upper()])
#                 else:
#                     paths.append([file, file.replace("mid", "chorus.txt").upper()])
#         return np.array(paths)
#     elif desired_format == "wav":
#         for file in os.listdir(folder_path):
#             if file[-4:] == ".wav":
#                 file_number = "{:03d}".format(int(file[:-4]))
#                 annotations_prefix = "RM-P"
#                 if downbeats == True:
#                     paths.append([file, annotations_prefix + file_number + (".chorus.txt").upper(), annotations_prefix + file_number + (".beat.txt").upper()])
#                 else:
#                     paths.append([file, annotations_prefix + file_number + (".chorus.txt").upper()])
#         return np.array(paths)
#     else:
#         raise NotImplementedError("Unknown format.")
       
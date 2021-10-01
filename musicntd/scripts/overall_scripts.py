# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:34:39 2020

@author: amarmore
"""

# A module containing some high-level scripts for decomposition and/or segmentation.

import soundfile as sf
import librosa.core
import librosa.feature
import tensorly as tl
import os
import numpy as np
import pathlib
import random

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
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        bars = np.load("{}/bars/{}.npy".format(persisted_path, song_name))
    except:
        bars = dm.get_bars_from_audio(song_path)
        np.save("{}/bars/{}".format(persisted_path, song_name), bars)
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
    bars = np.load("{}/bars/{}.npy".format(persisted_path, song_name))
    return bars
    
def load_or_save_spectrogram(persisted_path, song_path, feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20):
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
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        if "stft" in feature:   
            if "nfft" not in feature:
                spectrogram = np.load("{}/spectrograms/{}_{}-nfft{}_stereo_{}.npy".format(persisted_path, song_name, feature, n_fft, hop_length))
            else:
                spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}.npy".format(persisted_path, song_name, feature, hop_length))
        elif feature == "mel" or feature == "log_mel":
            raise err.InvalidArgumentValueException("Invalid mel parameter, are't you looking for mel_grill?")
        elif "mfcc" in feature:
            if "nmfcc" not in feature:
                spectrogram = np.load("{}/spectrograms/{}_{}-nmfcc{}_stereo_{}.npy".format(persisted_path, song_name, feature, n_mfcc, hop_length))
            else:
                spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}.npy".format(persisted_path, song_name, feature, hop_length))
        elif feature == "pcp":
            spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}_{}.npy".format(persisted_path, song_name, feature, hop_length, fmin))
        else:
            spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}.npy".format(persisted_path, song_name, feature, hop_length))

    except FileNotFoundError:
        the_signal, original_sampling_rate = sf.read(song_path)
        #the_signal, original_sampling_rate = librosa.load(song_path)
        if original_sampling_rate != 44100:
            the_signal = librosa.core.resample(np.asfortranarray(the_signal), original_sampling_rate, 44100)
        if "stft" in feature:
            if "nfft" not in feature: 
                spectrogram = features.get_spectrogram(the_signal, 44100, feature, hop_length, n_fft = n_fft)
                np.save("{}/spectrograms/{}_{}-nfft{}_stereo_{}".format(persisted_path, song_name, feature, n_fft, hop_length), spectrogram)
                return spectrogram
            else:              
                n_fft_arg = int(feature.split("nfft")[1])
                spectrogram = features.get_spectrogram(the_signal, 44100, feature, hop_length, n_fft = n_fft_arg)
                np.save("{}/spectrograms/{}_{}_stereo_{}".format(persisted_path, song_name, feature, hop_length), spectrogram)
                return spectrogram
        if feature == "mel" or feature == "log_mel":
            raise err.InvalidArgumentValueException("Invalid mel parameter, are't you looking for mel_grill?")
        if "mfcc" in feature:
            if "nmfcc" not in feature:
                spectrogram = features.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc)
                np.save("{}/spectrograms/{}_{}-nmfcc{}_stereo_{}".format(persisted_path, song_name, feature, n_mfcc, hop_length), spectrogram)
                return spectrogram        
            else:
                n_mfcc_arg = int(feature.split("nmfcc")[1])
                spectrogram = features.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc_arg)
                np.save("{}/spectrograms/{}_{}_stereo_{}".format(persisted_path, song_name, feature, hop_length), spectrogram)
                return spectrogram
        if feature == "pcp_tonnetz":
            # If chromas are already computed, try to load them instead of recomputing them.
            chromas = load_or_save_spectrogram(persisted_path, song_path, "pcp", hop_length, fmin = fmin)
            spectrogram = librosa.feature.tonnetz(y=None, sr = None, chroma = chromas)
            np.save("{}/spectrograms/{}_{}_stereo_{}_{}".format(persisted_path, song_name, feature, hop_length, fmin), spectrogram)
            return spectrogram
        # if feature == "tonnetz":
        #     hop_length = "fixed"
        #     fmin = "fixed"
        if feature == "pcp":
            # If it wasn't pcp_tonnetz, compute the spectrogram, and then save it.
            spectrogram = features.get_spectrogram(the_signal, 44100, feature, hop_length, fmin = fmin)
            np.save("{}/spectrograms/{}_{}_stereo_{}_{}".format(persisted_path, song_name, feature, hop_length, fmin), spectrogram)
            return spectrogram
        
        spectrogram = features.get_spectrogram(the_signal, 44100, feature, hop_length)
        np.save("{}/spectrograms/{}_{}_stereo_{}".format(persisted_path, song_name, feature, hop_length), spectrogram)
        return spectrogram

    return spectrogram

def load_or_save_spectrogram_and_bars(persisted_path, song_path, feature, hop_length, fmin = 98, n_fft = 2048):
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
    spectrogram = load_or_save_spectrogram(persisted_path, song_path, feature, hop_length, fmin = fmin, n_fft = n_fft)
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
    bars = np.load("{}/bars/{}.npy".format(persisted_path, song_name))
    spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}_{}".format(persisted_path, song_name, feature, hop_length, fmin))
    return bars, spectrogram

def NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "chromas", update_rule = "hals", beta = None): 
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
    if update_rule == "hals":
        path_for_ntd = "{}/ntd/{}_{}_{}".format(persisted_path, ranks[0], ranks[1], ranks[2])
    elif update_rule == "mu":
        path_for_ntd = "{}/ntd_mu/{}_{}_{}".format(persisted_path, ranks[0], ranks[1], ranks[2])
    else:
        raise NotImplementedError(f"Update rule type not understood: {update_rule}")
    
    if update_rule == "mu" and beta == None:
        raise NotImplementedError("Inconsistent arguments. Beta should be set if the update_rule is the MU.")
        
    try:
        a_core_path = "{}/core{}.npy".format(path_for_ntd, persisted_arguments)
        a_core = np.load(a_core_path)
        a_factor_path = "{}/factors{}.npy".format(path_for_ntd, persisted_arguments)
        a_factor = np.load(a_factor_path, allow_pickle=True)
        return a_core, a_factor
    except FileNotFoundError:
        if update_rule == "hals":
            core, factors = NTD.ntd(tensor_spectrogram, ranks = ranks, init = init, verbose = False,
                                sparsity_coefficients = [None, None, None, None], normalize = [True, True, False, True], mode_core_norm = 2,
                                deterministic = True)
        elif update_rule == "mu":
            core, factors = NTD.ntd_mu(tensor_spectrogram, ranks = ranks, init = init, verbose = False, beta = beta, n_iter_max=1000,
                                sparsity_coefficients = [None, None, None, None], normalize = [True, True, False, True], mode_core_norm = 2,
                                deterministic = True)
        
        pathlib.Path(path_for_ntd).mkdir(parents=True, exist_ok=True)
    
        core_path = "{}/core{}".format(path_for_ntd, persisted_arguments)
        np.save(core_path, core)
        factors_path = "{}/factors{}".format(path_for_ntd, persisted_arguments)
        np.save(factors_path, factors)
        return core, factors
    
def NTD_random_init_to_persist(persisted_path, persisted_arguments, tensor_spectrogram, ranks, seed): 
    """
    TODO.

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
    seed : integer
        The seed, fixing the random state.

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
    path_for_ntd = "{}/ntd_several_random_init/{}_{}_{}".format(persisted_path, ranks[0], ranks[1], ranks[2])
    if "512" in persisted_arguments:
        raise NotImplementedError("Probably an error in the code, as old hop_length seems to be passed")
    if persisted_arguments[-2:] == "32":
        raise NotImplementedError("Probably an error in the code, as the hop_length seems to be passed")
    try:
        a_core_path = "{}/core{}_{}.npy".format(path_for_ntd, persisted_arguments, seed)
        a_core = np.load(a_core_path)
        a_factor_path = "{}/factors{}_{}.npy".format(path_for_ntd, persisted_arguments, seed)
        a_factor = np.load(a_factor_path, allow_pickle=True)
        return a_core, a_factor
    except FileNotFoundError:
        factors_0 = []
        for mode in range(len(tensor_spectrogram.shape)):
            seeded_random = np.random.RandomState(seed)
            random_array = seeded_random.rand(tensor_spectrogram.shape[mode], ranks[mode])
            factors_0.append(tl.tensor(random_array))

        seeded_random = np.random.RandomState(seed)
        core_0 = tl.tensor(seeded_random.rand(np.prod(ranks)).reshape(tuple(ranks)))
        
        core, factors = NTD.compute_ntd(tensor_spectrogram, ranks, core_0, factors_0, n_iter_max=100, tol=1e-6,
                                       sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [True, True, False, True], hals = False,mode_core_norm=2,
                                       verbose=False, return_errors=False, deterministic=True)
        
        pathlib.Path(path_for_ntd).mkdir(parents=True, exist_ok=True)
    
        core_path = "{}/core{}_{}".format(path_for_ntd, persisted_arguments, seed)
        np.save(core_path, core)
        factors_path = "{}/factors{}_{}".format(path_for_ntd, persisted_arguments, seed)
        np.save(factors_path, factors)
        return core, factors

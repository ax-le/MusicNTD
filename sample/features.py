# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:54:59 2020

@author: amarmore
"""

import numpy as np
import librosa
from math import inf
import soundfile as sf

def get_spectrogram(signal, sr, feature, hop_length, n_fft = 2048, fmin = 98):
    """
    Returns a spectrogram, from the signal.
    Different types of spectrogram can be computed, and it's specified by the argument "feature".
        
    All these spectrograms are computed by using the toolbox librosa [1].
    
    #TODO: make all functions compatible with multichannel signals.

    Parameters
    ----------
    signal : numpy array
        Signal of the song.
        At this time, for most songs, the signal must be single-channel/mono signal.
    sr : float
        Sampling rate of the signal, generally 44100Hz.
    feature : String
        The type of spectrogram to compute.
            - stft : computes the Short-Time Fourier Transform of the signal.
            - pcp : computes a chromagram.
            NB: this chromagram has been specificly fitted as a team, 
            and the arguments are non standard but rather technical choices.
            - old_pcp : computes a chromagram, as computed in the MSAF toolbox [2].
            NB: here for comparison test, shoudln't be used.
            - pcp_stft : computes a chromagram from the stft of the song.
            - cqt : computes a Constant-Q transform of the song.
    hop_length : integer
        The desired hop_length, which is the step between two frames (ie the time "discretization" step)
        It is expressed in terms of number of samples, which are defined by the sampling rate.
    n_fft : integer, optional
        Number of frames by stft feature.
        The default is 2048.
    fmin : integer, optional
        The minimal frequence to consider, used for denoizing.
        The default is 98.

    Raises
    ------
    NotImplementedError
        If the "feature" argument is not one of thoses presented above.

    Returns
    -------
    numpy array
        Spectrogram of the signal.
        
    References
    ----------
    [1] McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015, July).
    librosa: Audio and music signal analysis in python. 
    In Proceedings of the 14th python in science conference (Vol. 8).
    
    [2] Nieto, O., & Bello, J. P. (2015). 
    Msaf: Music structure analytis framework. 
    In Proceedings of 16th International Society for Music Information Retrieval Conference (ISMIR 2015).

    """
    if feature.lower() == "stft":
        stft = librosa.core.stft(np.asfortranarray(signal), n_fft=n_fft, hop_length = hop_length)
        power_spectrogram = np.abs(stft) ** 2
        return power_spectrogram
    elif feature.lower() == "pcp_stft":
        audio_harmonic, _ = librosa.effects.hpss(y=np.asfortranarray(signal))
        chroma_stft = librosa.feature.chroma_stft(y=audio_harmonic, sr=sr, n_fft = n_fft, hop_length=hop_length)
        return chroma_stft
    elif feature == "pcp":
        norm=inf # Normalization of columns
        win_len_smooth=82 # Size of the smoothing window
        
        n_octaves=6
        bins_per_chroma = 3
        bins_per_octave=bins_per_chroma * 12
        if len(signal.shape) == 1:
            return librosa.feature.chroma_cens(y=np.asfortranarray(signal),sr=sr,hop_length=hop_length,
                                   fmin=fmin, n_chroma=12, n_octaves=n_octaves, bins_per_octave=bins_per_octave,
                                   norm=norm, win_len_smooth=win_len_smooth)
        
        pcp = librosa.feature.chroma_cens(y=np.asfortranarray(signal[:,0]),sr=sr,hop_length=hop_length,
                                   fmin=fmin, n_chroma=12, n_octaves=n_octaves, bins_per_octave=bins_per_octave,
                                   norm=norm, win_len_smooth=win_len_smooth)
        for i in range(1,signal.shape[1]):
            pcp += librosa.feature.chroma_cens(y=np.asfortranarray(signal[:,i]),sr=sr,hop_length=hop_length,
                                   fmin=fmin, n_chroma=12, n_octaves=n_octaves, bins_per_octave=bins_per_octave,
                                   norm=norm, win_len_smooth=win_len_smooth)
    
        return pcp
    elif feature.lower() == "cqt":
        constant_q_transf = librosa.core.cqt(np.asfortranarray(signal), sr = sr, hop_length = hop_length)
        power_cqt = np.abs(constant_q_transf) ** 2
        return power_cqt
    else:
        raise NotImplementedError("Unknown signal representation.")

def get_and_persist_spectrogram(song_path, feature, hop_length, fmin, persisted_path):
    """
    Function trying to get a spectrogram if it was already computed (and persisted somewhere),
    computes it otherwise.

    Parameters
    ----------
    song_path : String
        Path to the song.
    feature : String
        Type of the spectrogram to compute.
        See "get_spectrogram()" for further details.
    hop_length : integer
        The desired hop_length, which is the step between two frames (ie the time "discretization" step)
        It is expressed in terms of number of samples, which are defined by the sampling rate.
    fmin : integer, optional
        The minimal frequence to consider, used for denoizing.
        The default is 98.
    persisted_path : string, OPTIONAL
        The path where to check if the data is persisted,
        and the path to persist it otherwise.

    Raises
    ------
    NotImplementedError
        Errors.

    Returns
    -------
    spectrogram : numpy array
        The spectrogram if the signal.

    """

    if feature not in ['pcp', 'old_pcp']:
        raise NotImplementedError("Ces features ne sont pas prises en compte pour la persistance.")
    song_name = song_path.split("\\")[-1].replace(".wav","")
    try:
        spectrogram = np.load(persisted_path + "spectrograms\\{}_{}_stereo_{}_{}.npy".format(song_name, feature, hop_length, fmin))
    except FileNotFoundError:
        print("Spectrogramme non trouvé")
        signal, sr = sf.read(song_path)
        spectrogram = get_spectrogram(signal, sr, feature, hop_length, fmin)
        np.save(persisted_path + "spectrograms\\{}_{}_stereo_{}_{}".format(song_name, feature, hop_length, fmin), spectrogram)
    return spectrogram

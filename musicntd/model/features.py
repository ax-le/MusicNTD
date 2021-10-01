# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:54:59 2020

@author: amarmore
"""

import numpy as np
#import librosa
import librosa.core
import librosa.feature
import librosa.effects
from math import inf
import soundfile as sf
import musicntd.model.errors as err

def get_spectrogram(signal, sr, feature, hop_length, n_fft = 2048, fmin = 98, n_mfcc = 20):
    """
    Returns a spectrogram, from the signal.
    Different types of spectrogram can be computed, and it's specified by the argument "feature".
        
    All these spectrograms are computed by using the toolbox librosa [1].
    
    Parameters
    ----------
    signal : numpy array
        Signal of the song.
    sr : float
        Sampling rate of the signal, generally 44100Hz.
    feature : String
        The types of spectrograms to compute.
            - stft : computes the Short-Time Fourier Transform of the signal.
            - pcp : computes a chromagram.
            NB: this chromagram has been specificly fitted as a team, 
            and the arguments are non standard but rather technical choices.
            - pcp_stft : computes a chromagram from the stft of the song.
            - cqt : computes a Constant-Q transform of the song.
            - tonnetz : computes the tonnetz representation of the song.
            - pcp_tonnetz : computes the tonnetz representation of the song, starting from the chromas.
                It allows us to better control paramaters over the computation of tonnetz, 
                and can reduce computation when chromas are already computed (for scripts loading already computed spectrograms).
            - mfcc : computes the Mel-Frequency Cepstral Coefficients of the song.
            - mel : computes the mel-spectrogram of the song.

    hop_length : integer
        The desired hop_length, which is the step between two frames (ie the time "discretization" step)
        It is expressed in terms of number of samples, which are defined by the sampling rate.
    n_fft : integer, optional
        Number of frames by stft feature.
        The default is 2048.
    fmin : integer, optional
        The minimal frequence to consider, used for denoizing.
        The default is 98.
    n_mfcc : integer, optional
        Number of mfcc features.
        The default is 20 (as in librosa).

    Raises
    ------
    InvalidArgumentValueException
        If the "feature" argument is not presented above.

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
        if len(signal.shape) == 1:
            stft = librosa.core.stft(np.asfortranarray(signal), n_fft=n_fft, hop_length = hop_length)
            power_spectrogram = np.abs(stft) ** 2
            return power_spectrogram
        
        power_spectrogram = np.abs(librosa.core.stft(np.asfortranarray(signal[:,0]), n_fft=n_fft, hop_length = hop_length))**2
        for i in range(1,signal.shape[1]):
            power_spectrogram += np.abs(librosa.core.stft(np.asfortranarray(signal[:,i]), n_fft=n_fft, hop_length = hop_length))**2
        return power_spectrogram
    
    elif feature.lower() == "pcp_stft":
        if len(signal.shape) == 1:
            audio_harmonic, _ = librosa.effects.hpss(y=np.asfortranarray(signal))
            chroma_stft = librosa.feature.chroma_stft(y=audio_harmonic, sr=sr, n_fft = n_fft, hop_length=hop_length)
            return chroma_stft
        audio_harmonic, _ = librosa.effects.hpss(y=np.asfortranarray(signal[:,0]))
        chroma_stft = librosa.feature.chroma_stft(y=audio_harmonic, sr=sr, n_fft = n_fft, hop_length=hop_length)
        for i in range(1,signal.shape[1]):
            audio_harmonic, _ = librosa.effects.hpss(y=np.asfortranarray(signal[:,i]))
            chroma_stft += librosa.feature.chroma_stft(y=audio_harmonic, sr=sr, n_fft = n_fft, hop_length=hop_length)   
        return chroma_stft
    elif feature == "pcp":
        norm=inf # Columns normalization
        win_len_smooth=82 # Size of the smoothign window
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
        if len(signal.shape) == 1:
            constant_q_transf = librosa.core.cqt(np.asfortranarray(signal), sr = sr, hop_length = hop_length)
            power_cqt = np.abs(constant_q_transf) ** 2
            return power_cqt
        power_cqt = np.abs(librosa.core.cqt(np.asfortranarray(signal[:,0]), sr = sr, hop_length = hop_length)) ** 2
        for i in range(1,signal.shape[1]):
            power_cqt += np.abs(librosa.core.cqt(np.asfortranarray(signal[:,i]), sr = sr, hop_length = hop_length)) ** 2
        return power_cqt
    elif feature.lower() == "log_cqt":
        if len(signal.shape) == 1:
            constant_q_transf = librosa.core.cqt(np.asfortranarray(signal), sr = sr, hop_length = hop_length)
            power_cqt = np.abs(constant_q_transf) ** 2
            log_cqt = ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(np.array(power_cqt)), ref=np.max)) + 1.0
            return log_cqt
        power_cqt = np.abs(librosa.core.cqt(np.asfortranarray(signal[:,0]), sr = sr, hop_length = hop_length)) ** 2
        for i in range(1,signal.shape[1]):
            power_cqt += np.abs(librosa.core.cqt(np.asfortranarray(signal[:,i]), sr = sr, hop_length = hop_length)) ** 2
        log_cqt = ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(np.array(power_cqt)), ref=np.max)) + 1.0
        return log_cqt
    elif feature.lower() == "tonnetz":
        if len(signal.shape) == 1:
            return librosa.feature.tonnetz(np.asfortranarray(signal), sr = sr)
        tonnetz = librosa.feature.tonnetz(np.asfortranarray(signal[:,0]), sr = sr)
        for i in range(1,signal.shape[1]):
            tonnetz += librosa.feature.tonnetz(np.asfortranarray(signal[:,i]), sr = sr)
        return tonnetz
    elif feature.lower() == "pcp_tonnetz":
        return librosa.feature.tonnetz(y=None, sr = None, chroma = get_spectrogram(signal, sr, "pcp", hop_length, fmin = fmin))
    elif feature.lower() == "hcqt":
        return my_compute_hcqt(np.asfortranarray(signal[:,0]), sr)
    
    elif feature.lower() == "mfcc":
        if len(signal.shape) == 1:
            return librosa.feature.mfcc(np.asfortranarray(signal), sr = sr, hop_length = hop_length, n_mfcc=n_mfcc)
        mfcc = librosa.feature.mfcc(np.asfortranarray(signal[:,0]), sr = sr, hop_length = hop_length, n_mfcc=n_mfcc)
        for i in range(1,signal.shape[1]):
            mfcc += librosa.feature.mfcc(np.asfortranarray(signal[:,i]), sr = sr, hop_length = hop_length, n_mfcc=n_mfcc)
        return mfcc
    
    # For Mel spectrograms, we use the same parameters as the ones of [1].
    # [1]Grill, Thomas, and Jan Schlüter. "Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations." ISMIR. 2015.
    elif feature.lower() == "mel_grill":
        if len(signal.shape) == 1:
            return np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        mel = np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,0]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        for i in range(1,signal.shape[1]):
            mel += np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,i]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        return mel
    
    elif feature == "log_mel_grill":
        if len(signal.shape) == 1:
            return librosa.power_to_db(np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000)))
        mel = np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,0]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        for i in range(1,signal.shape[1]):
            mel += np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,i]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        return librosa.power_to_db(mel)
    
    elif feature == "nn_log_mel_grill":
        if len(signal.shape) == 1:
            mel = np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
            return librosa.power_to_db(mel + np.ones(mel.shape))
        mel = np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,0]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        for i in range(1,signal.shape[1]):
            mel += np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,i]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        return librosa.power_to_db(mel + np.ones(mel.shape))
    
    # elif feature == "padded_log_mel_grill":
    #     log_mel = get_spectrogram(signal, sr, "log_mel_grill", hop_length, n_fft = n_fft)
    #     return log_mel - np.amin(log_mel) * np.ones(log_mel.shape)

    elif feature == "mel" or feature == "log_mel":
        raise err.InvalidArgumentValueException("Invalid mel parameter, are't you looking for mel_grill?")
    else:
        raise err.InvalidArgumentValueException("Unknown signal representation.")

def get_and_persist_spectrogram(song_path, feature, hop_length, fmin, persisted_path = "blank_path"):
    raise err.BuggedFunctionException("Shouldn't be used, try load_or_save_spectrogram() from overall_scripts.py instead.")
    
def get_hcqt_params():
    """
    Credit to & al. [1] (comes directly from https://github.com/rabitt/ismir2017-deepsalience)
    
    Fixing parameters for the HCQT computation.

    Returns
    -------
    bins_per_octave : TYPE
        DESCRIPTION.
    n_octaves : TYPE
        DESCRIPTION.
    harmonics : TYPE
        DESCRIPTION.
    sr : TYPE
        DESCRIPTION.
    fmin : TYPE
        DESCRIPTION.
    hop_length : TYPE
        DESCRIPTION.
        
    References
    ----------
    [1] Bittner, R. M., McFee, B., Salamon, J., Li, P., & Bello, J. P. (2017, October). 
    Deep Salience Representations for F0 Estimation in Polyphonic Music. In ISMIR (pp. 63-70).

    """
    bins_per_octave = 60
    n_octaves = 6
    harmonics = [0.5, 1, 2, 3, 4, 5]
    sr = 22050
    fmin = 32.7
    hop_length = 256
    return bins_per_octave, n_octaves, harmonics, sr, fmin, hop_length


def compute_hcqt_bittner(signal, sr):
    """
    Credit to Bittner & al. [1] (comes from https://github.com/rabitt/ismir2017-deepsalience).
    
    Computes HCQT representation of the signal, as presented in [1] (3-rd order tensor).

    Parameters
    ----------
    signal : numpy array
        Signal of the song.
    sr : int
        the sampling_rate

    Returns
    -------
    log_hcqt : np array
        The tensor of logarithm HCQT.
        
    References
    ----------
    [1] Bittner, R. M., McFee, B., Salamon, J., Li, P., & Bello, J. P. (2017, October). 
    Deep Salience Representations for F0 Estimation in Polyphonic Music. In ISMIR (pp. 63-70).
    """
    (bins_per_octave, n_octaves, harmonics,
     sr, f_min, hop_length) = get_hcqt_params()
    #y, fs = librosa.load(audio_fpath, sr=sr)

    cqt_list = []
    shapes = []
    for h in harmonics:
        cqt = librosa.cqt(
            signal, sr=sr, hop_length=hop_length, fmin=f_min*float(h),
            n_bins=bins_per_octave*n_octaves,
            bins_per_octave=bins_per_octave
        )
        cqt_list.append(cqt)
        shapes.append(cqt.shape)

    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[1] for s in shapes])
        new_cqt_list = []
        for i in range(len(cqt_list)):
            new_cqt_list.append(cqt_list[i][:, :min_time])
        cqt_list = new_cqt_list

    log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(
        np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

    return log_hcqt

def my_compute_hcqt(signal, sr):
    """
    Credit to Bittner & al. [1] (comes from https://github.com/rabitt/ismir2017-deepsalience).
    
    Computes HCQT representation of the signal, as presented in [1] (3-rd order tensor).
    The order of the mode is changed though, so tht first two modes correspond to frequency and time respectively,
    and that the third corresponds to harmonic content.

    Parameters
    ----------
    signal : numpy array
        Signal of the song.
    sr : int
        the sampling_rate

    Returns
    -------
    log_hcqt : np array
        The tensor of logarithm HCQT.
        
    References
    ----------
    [1] Bittner, R. M., McFee, B., Salamon, J., Li, P., & Bello, J. P. (2017, October). 
    Deep Salience Representations for F0 Estimation in Polyphonic Music. In ISMIR (pp. 63-70).
    """
    (bins_per_octave, n_octaves, harmonics, sr, f_min, hop_length) = get_hcqt_params()

    freq_mode_len = bins_per_octave*n_octaves

    first_cqt = librosa.cqt(signal, sr=sr, hop_length=hop_length, fmin=f_min*float(harmonics[0]),
                            n_bins=freq_mode_len, bins_per_octave=bins_per_octave)

    time_mode_len = first_cqt.shape[1]
    
    h_cqt = np.array(first_cqt).reshape(freq_mode_len, time_mode_len, 1)
    
    for h in harmonics[1:]:
        cqt = librosa.cqt(signal, sr=sr, hop_length=hop_length, fmin=f_min*float(h),
            n_bins=bins_per_octave*n_octaves,bins_per_octave=bins_per_octave)
        current_cqt = cqt.reshape(freq_mode_len, time_mode_len, 1)
        h_cqt = np.append(h_cqt, current_cqt, axis = 2)

    log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(h_cqt), ref=np.max)) + 1.0

    return log_hcqt



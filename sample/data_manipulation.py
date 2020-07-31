# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:09:10 2020

@author: amarmore
"""

# A file which contains all code regarding conversion of data, or extracting information from it
# (typically getting the bars, converting segments in frontiers, sonifying segmentation or computing its score).

import context

import numpy as np
import math
import madmom.io.midi as midi_handler
import madmom.features.downbeats as dbt
import soundfile as sf
import mir_eval
import scipy

# %% Read and treat inputs
def get_piano_roll(path, hop_length_seconds, threshold_offset = None):
    """
    Returns a piano roll representation from a midi file input,
    ie returns a spectrogram with time in x-axis and midi note in y-axis.
    
    Parameters
    ----------
    path : String
        The path of the midi file to handle.
    hop_length_seconds : float
        hop_length, in seconds.
    threshold_offset : int, optional
        Maximum value (in number of frames) for the duration a note, 
        to prevent a note from being constantly played (as there is no decay in this representation).
        If set to None, the default value of the offset in the midi file will be used.
        The default is None.
        
    Returns
    -------
    piano_roll : numpy array
        The piano roll representation of the midi file.
    """
    fs = hop_length_seconds
    
    # ‘onset time’ ‘note number’ ‘duration’ ‘velocity’ ‘channel’
    midi = midi_handler.load_midi(path)
    nb_time_bins = int(round((midi[-1][0] + midi[-1][2])/fs))
    nb_midi_bins = 130
    piano_roll = np.zeros((nb_midi_bins, nb_time_bins))
    max_bins = math.inf
    if threshold_offset is not None:
        max_bins = threshold_offset
    for note in midi:
        note_onset = int(round(note[0]/fs))
        note_duration = min(int(round(note[2]/fs)), max_bins) # Thresholded offset, to avoid constant notes over a bar
        piano_roll[int(note[1]),note_onset:note_onset + note_duration] = 1
    return piano_roll

def get_bars_from_audio(song):
    """
    Returns the bars of a song, directly from its audio signal.
    Encapsulates the downbeat estimator from the madmom toolbox [1].

    Parameters
    ----------
    song : String
        Path to the desired song.

    Returns
    -------
    downbeats_times : list of tuple of float
        List of the estimated bars, as (start, end) times.
        
    References
    ----------
    [1] Böck, S., Korzeniowski, F., Schlüter, J., Krebs, F., & Widmer, G. (2016, October). 
    Madmom: A new python audio and music signal processing library. 
    In Proceedings of the 24th ACM international conference on Multimedia (pp. 1174-1178).

    """
    act = dbt.RNNDownBeatProcessor()(song)
    proc = dbt.DBNDownBeatTrackingProcessor(beats_per_bar=4, fps=100)
    song_beats = proc(act)
    downbeats_times = []
    if song_beats[0][1] != 1:
        downbeats_times.append(0.1)
    for beat in song_beats:
        if beat[1] == 1:
            downbeats_times.append(beat[0])
    mean_bar = np.mean([downbeats_times[i + 1] - downbeats_times[i] for i in range(len(downbeats_times) - 1)])
    signal_length = act.shape[0]/100
    while downbeats_times[-1] + mean_bar < 1.1 * signal_length:
        downbeats_times.append(round(downbeats_times[-1] + mean_bar, 2))
    downbeats_times.append(signal_length)
    return frontiers_to_segments(downbeats_times)

def get_segmentation_from_txt(path, annotations_type):
    """
    Reads the segmentation annotations, and returns it in a list of tuples (start, end, index as a number)
    This function has been developped for AIST and MIREX10 annotations, adapted for these types of annotations.
    It will not work with another set of annotation.

    Parameters
    ----------
    path : String
        The path to the annotation.
    annotations_type : "AIST" [1] or "MIREX10" [2]
        The type of annotations to load (both have a specific behavior and formatting)
        
    Raises
    ------
    NotImplementedError
        If the type of annotations is neither AIST or MIREX10

    Returns
    -------
    segments : list of tuples (float, float, integer)
        The segmentation, formatted in a list of tuples, and with labels as numbers (easier to interpret computationnally).

    References
    ----------
    [1] Goto, M. (2006, October). AIST Annotation for the RWC Music Database. In ISMIR (pp. 359-360).
    
    [2] Bimbot, F., Sargent, G., Deruty, E., Guichaoua, C., & Vincent, E. (2014, January). 
    Semiotic description of music structure: An introduction to the Quaero/Metiss structural annotations.

    """
    file_seg = open(path)
    segments = []
    labels = []
    for part in file_seg.readlines():
        tupl = part.split("\t")
        if tupl[2] not in labels: # If label wasn't already found in this annotation
            idx = len(labels)
            labels.append(tupl[2])
        else: # If this label was found for another segment
            idx = labels.index(tupl[2])
        if annotations_type == "AIST":
            segments.append(((int(tupl[0]) / 100), (int(tupl[1]) / 100), idx))
        elif annotations_type == "MIREX10":
            segments.append((round(float(tupl[0]), 3), round(float(tupl[1]), 3), idx))
        else:
            raise NotImplementedError("Annotations type not understood")
    return segments

def get_annotation_name_from_song(song_number, annotations_type):
    """
    Returns the name of the annotation of this song according to the desired annotation type
    
    Specificly designed for RWC Pop dataset, shouldn't be used otherwise.
    For now are available:
        - AIST annotations [1]
        - MIREX 10 annotations [2]
    
    Parameters
    ----------
    song_number : integer or string
        The number of the song (which is its name).
    annotations_type : string
        The desired type of annotation.

    Raises
    ------
    NotImplementedError
        If the annotatipn type is not implemented.

    Returns
    -------
    string
        The name of the file containing the annotation.
        
    References
    ----------
    [1] Goto, M. (2006, October). AIST Annotation for the RWC Music Database. In ISMIR (pp. 359-360).
    
    [2] Bimbot, F., Sargent, G., Deruty, E., Guichaoua, C., & Vincent, E. (2014, January). 
    Semiotic description of music structure: An introduction to the Quaero/Metiss structural annotations.

    """
    if annotations_type == "MIREX10":
        return "RM-P{:03d}.BLOCKS.lab".format(int(song_number))
    elif annotations_type == "AIST":
        return "RM-P{:03d}.CHORUS.TXT".format(int(song_number))
    else:
        raise NotImplementedError("Annotations type not understood")

# %% Conversion of data (time/frame/beat and segment/frontiers)
def frontiers_from_time_to_frame_idx(seq, hop_length_seconds):
    """
    Converts a sequence of frontiers in time to their values in frame indexes.

    Parameters
    ----------
    seq : list of float/times
        The list of times to convert.
    hop_length_seconds : float
        hop_length (time between two consecutive frames), in seconds.

    Returns
    -------
    list of integers
        The sequence, as a list, in frame indexes.
    """
    
    return [int(round(frontier/hop_length_seconds)) for frontier in seq]

def segments_from_time_to_frame_idx(segments, hop_length_seconds):
    """
    Converts a sequence of segments (start, end) in time to their values in frame indexes.

    Parameters
    ----------
    segements : list of tuple
        The list of segments, as tuple (start, end), to convert.
    hop_length_seconds : float
        hop_length (time between two consecutive frames), in seconds.

    Returns
    -------
    list of integers
        The sequence, as a list, in frame indexes.
    """
    to_return = []
    for segment in segments:
        bar_in_frames = [int(round(segment[0]/hop_length_seconds)), int(round(segment[1]/hop_length_seconds))]
        if bar_in_frames[0] != bar_in_frames[1]:
            to_return.append(bar_in_frames)
    return to_return
    
def frontiers_from_time_to_bar(seq, bars):
    """
    Convert the frontiers in time to a bar index.
    The selected bar is the one which end is the closest from the frontier.

    Parameters
    ----------
    seq : list of float
        The list of frontiers, in time.
    bars : list of tuple of floats
        The bars, as (start time, end time) tuples.

    Returns
    -------
    seq_barwise : list of integers
        List of times converted in bar indexes.

    """
    seq_barwise = []
    for frontier in seq:
        for idx, bar in enumerate(bars):
            if frontier >= bar[0] and frontier < bar[1]:
                if bar[1] - frontier < frontier - bar[0]:
                    seq_barwise.append(idx)
                else:
                    if idx == 0:
                        seq_barwise.append(idx)
                        #raise NotImplementedError("The current frontier is labelled in the start silence, which is incorrect.")
                        #print("The current frontier {} is labelled in the start silence ({},{}), which is incorrect.".format(frontier, bar[0], bar[1]))
                    else:
                        seq_barwise.append(idx - 1)
                break
    return seq_barwise

def frontiers_from_bar_to_time(seq, bars):
    """
    Converts the frontiers (or a sequence of integers) from bar indexes to absolute times of the bars.
    The frontier is considered as the end of the bar.

    Parameters
    ----------
    seq : list of integers
        The frontiers, in bar indexes.
    bars : list of tuple of floats
        The bars, as (start time, end time) tuples.

    Returns
    -------
    to_return : list of float
        The frontiers, converted in time (from bar indexes).

    """
    to_return = []
    for frontier in seq:
        bar_frontier = bars[frontier][1]
        if bar_frontier not in to_return:
            to_return.append(bar_frontier)
    return to_return
    
def frontiers_to_segments(frontiers):
    """
    Computes a list of segments starting from the frontiers between them.

    Parameters
    ----------
    frontiers : list of floats
        The list of frontiers.

    Returns
    -------
    to_return : list of tuples of floats
        The segments, as tuples (start, end).

    """
    to_return = []
    if frontiers[0] != 0:
        to_return.append((0,frontiers[0]))
    for idx in range(len(frontiers) - 1):
        to_return.append((frontiers[idx], frontiers[idx + 1]))
    return to_return

def segments_to_frontiers(segments):
    """
    Computes a list of frontiers from the segments.

    Parameters
    ----------
    segments : list of tuples of floats
        The segments, as tuples.

    Returns
    -------
    list
        Frontiers between segments.

    """
    return [i[1] for i in segments]

def segments_from_bar_to_time(segments, bars):
    """
    Converts segments from bar indexes to time.

    Parameters
    ----------
    segments : list of tuple of integers
        The indexes of the bars defining the segments (start, end).
    bars : list of tuple of float
        Bars, as tuples (start, end), in time.

    Returns
    -------
    numpy array
        Segments, in time.

    """
    to_return = []
    for start, end in segments:
        if end >= len(bars):
            to_return.append([bars[start][1], bars[-1][1]])
        else:
            to_return.append([bars[start][1], bars[end][1]])
    return np.array(to_return)

def align_segments_on_bars(segments, bars):
    """
    Aligns the estimated segments to the closest bars (in time).
    The idea is that segments generally start and end on downbeats,
    and that realigning the estimation could improve perfomance for low tolerances scores.
    Generally used for comparison with techniques which don't align their segmentation on bars.

    Parameters
    ----------
    segments : list of tuple of float
        Time of the estimated segments, as (start, end).
    bars : list of tuple of float
        The bars of the signal.

    Returns
    -------
    list of tuple of floats
        Segments, realigned on bars.

    """
    frontiers = segments_to_frontiers(segments)
    return frontiers_to_segments(align_frontiers_on_bars(frontiers, bars))
    
def align_frontiers_on_bars(frontiers, bars):
    """
    Aligns the frontiers of segments to the closest bars (in time).
    The idea is that frontiers generally occurs on downbeats,
    and that realigning the estimation could improve perfomance for low tolerances scores.
    Generally used for comparison with techniques which don't align their segmentation on bars.

    Parameters
    ----------
    frontiers : list of float
        Time of the estimated frontiers.
    bars : list of tuple of float
        The bars of the signal.

    Returns
    -------
    frontiers_on_bars : list of floats
        Frontiers, realigned on bars.

    """
    frontiers_on_bars = []
    i = 1
    for frontier in frontiers:
        while i < len(bars) - 1 and bars[i][1] < frontier:
            i+=1
        if i == len(bars) - 1:
            frontiers_on_bars.append(frontier)
        else:
            if bars[i][1] - frontier < frontier - bars[i][0]:
                frontiers_on_bars.append(bars[i][1])
            else:
                frontiers_on_bars.append(bars[i][0])
    return frontiers_on_bars
            
# %% Sonification of the segmentation
def sonify_frontiers_path(audio_file_path, frontiers_in_seconds, output_path):
    """
    Takes the path of the song and frontiers, and write a song with the frontiers sonified ("bip" in the song).
    Function inspired from MSAF.

    Parameters
    ----------
    audio_file_path: String
        The path to the song, (as signal).
    frontiers_in_seconds: list of floats
        The frontiers, in time/seconds.
    output_path: String
        The path where to write the song with sonified frontiers.

    Returns
    -------
    Nothing, but writes a song at output_path

    """
    the_signal, sampling_rate = sf.read(audio_file_path)
    sonify_frontiers_song(the_signal, sampling_rate, frontiers_in_seconds, output_path)

def sonify_frontiers_song(song_signal, sampling_rate, frontiers_in_seconds, output_path):
    """
    Takes a song as a signal, and add the frontiers to this signal.
    It then writes it as a file.
    Function inspired from MSAF.

    Parameters
    ----------
    song_signal : numpy array
        The song as a signal.
    sampling_rate : integer
        The sampling rate of the signal, in Hz.
    frontiers_in_seconds: list of floats
        The frontiers, in time/seconds.
    output_path: String
        The path where to write the song with sonified frontiers.

    Returns
    -------
    Nothing, but writes a song at the output_path.

    """
    frontiers_signal = mir_eval.sonify.clicks(frontiers_in_seconds, sampling_rate)
    
    signal_with_frontiers = np.zeros(max(len(song_signal[:,0]), len(frontiers_signal)))
    
    signal_with_frontiers[:len(song_signal[:,0])] = song_signal[:,0]
    signal_with_frontiers[:len(frontiers_signal)] += frontiers_signal
    
    scipy.io.wavfile.write(output_path, sampling_rate, signal_with_frontiers)
    
# %% Score calculation encapsulation
def compute_score_from_frontiers_in_bar(reference, frontiers_in_bar, bars, window = 0.5):
    """
    Computes precision, recall and f measure from estimated frontiers (in bar indexes) and the reference (in seconds).
    Scores are computed from the mir_eval toolbox.

    Parameters
    ----------
    reference : list of tuples
        The reference annotations, as a list of tuples (start, end), in seconds.
    frontiers : list of integers
        The frontiers between segments, in bar indexes.
    bars : list of tuples
        The bars of the song.
    window : float, optional
        The window size for the score (tolerance for the frontier to be validated).
        The default is 0.5.

    Returns
    -------
    precision: float \in [0,1]
        Precision of these frontiers,
        ie the proportion of accurately found frontiers among all found frontiers.
    recall: float \in [0,1]
        Recall of these frontiers,
        ie the proportion of accurately found frontiers among all accurate frontiers.
    f_measure: float \in [0,1]
        F measure of these frontiers,
        ie the geometric mean of both precedent scores.
        
    """
    try:
        np.array(bars).shape[1]
    except:
        raise NotImplementedError("Bug de passage a nouvelle version: bars est dbt (une liste), traquer et corriger erreur.")
    frontiers_in_time = frontiers_from_bar_to_time(frontiers_in_bar, bars)
    return compute_score_of_segmentation(reference, frontiers_to_segments(frontiers_in_time), window_length = window)

def compute_score_of_segmentation(reference, segments_in_time, window_length = 0.5):
    """
    Computes precision, recall and f measure from estimated segments and the reference, both in seconds.    
    Scores are computed from the mir_eval toolbox.

    Parameters
    ----------
    reference : list of tuples
        The reference annotations, as a list of tuples (start, end), in seconds.
    segments_in_time : list of tuples
        The segments, in seconds, as tuples (start, end).
    window_length : float, optional
        The window size for the score (tolerance for the frontier to be validated).
        The default is 0.5.

    Returns
    -------
    precision: float \in [0,1]
        Precision of these frontiers,
        ie the proportion of accurately found frontiers among all found frontiers.
    recall: float \in [0,1]
        Recall of these frontiers,
        ie the proportion of accurately found frontiers among all accurate frontiers.
    f_measure: float \in [0,1]
        F measure of these frontiers,
        ie the geometric mean of both precedent scores.

    """
    ref_intervals, useless = mir_eval.util.adjust_intervals(reference,t_min=0)
    est_intervals, useless = mir_eval.util.adjust_intervals(np.array(segments_in_time), t_min=0, t_max=ref_intervals[-1, 1])
    try:
        return mir_eval.segment.detection(ref_intervals, est_intervals, window = window_length, trim = False)
    except ValueError:
        cleaned_intervals = []
        #print("A segment is (probably) composed of the same start and end. Can happen with time -> bar -> time conversion, but should'nt happen for data originally segmented in bars.")
        for idx in range(len(est_intervals)):
            if est_intervals[idx][0] != est_intervals[idx][1]:
                cleaned_intervals.append(est_intervals[idx])
        return mir_eval.segment.detection(ref_intervals, np.array(cleaned_intervals), window = window_length, trim = False)

def compute_rates_of_segmentation(reference, segments_in_time, window_length = 0.5):
    """
    Computes True Positives, False Positives and False Negatives from estimated segments and the reference, both in seconds.    
    Scores are computed from the mir_eval toolbox.
    (What happens is that precision/rap/F1 are computed via mir_eval, by computing these rates but never releasing them.
    Hence, they are recomputed here from these values.)

    Parameters
    ----------
    reference : list of tuples
        The reference annotations, as a list of tuples (start, end), in seconds.
    segments_in_time : list of tuples
        The segments, in seconds, as tuples (start, end).
    window_length : float, optional
        The window size for the score (tolerance for the frontier to be validated).
        The default is 0.5.

    Returns
    -------
    True Positives: Integer
        The number of True Positives, 
        ie the number of accurately found frontiers.
    False Positives: Integer
        The number of False Positives,
        ie the number of wrongly found frontiers (estimated frontiers which are incorrect).
    False Negative : Integer
        The number of False Negatives,
        ie the number of frontiers undetected (accurate frontiers which are not found in teh estimation).

    """
    ref_intervals, useless = mir_eval.util.adjust_intervals(reference,t_min=0)
    prec, rec, _ = compute_score_of_segmentation(reference, segments_in_time, window_length = window_length)
    tp = int(round(rec * (len(ref_intervals) + 1)))
    fp = int(round((tp * (1 - prec))/prec))
    fn = int(round((tp * (1 - rec))/rec))
    return tp, fp, fn

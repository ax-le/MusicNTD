# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:10:09 2020

@author: amarmore
"""

### Ugly code, for little and specific tests.
### These functions where used during development but not for final tests.
### Needs to be parsed and factorized.

# Future versions should not contain this file.

from IPython.display import display, Markdown

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorly as tl
import pandas as pd
pd.set_option('precision', 4)

import nn_fac.ntd as NTD
import musicntd.autosimilarity_segmentation as as_seg
import musicntd.data_manipulation as dm
import musicntd.tensor_factory as tf
import musicntd.scripts.overall_scripts as scr
import musicntd.model.features as features
import musicntd.model.errors as err
from musicntd.model.current_plot import *

import copy

from mpl_toolkits.mplot3d import Axes3D


default_persisted_path = "C:\\Users\\amarmore\\Desktop\\data_persisted"
rwc_folder_path = "C:\\Users\\amarmore\\Desktop\\Audio samples\\RWC Pop\\Entire RWC"
root_rwc_folder = "C:\\Users\\amarmore\\Desktop\\Audio samples\\RWC Pop"
annotations_folder_path = "C:\\Users\\amarmore\\Desktop\\Audio samples\\RWC Pop\\annotations"
hop_length = 32 #44*2**6 # Must be a multiple of 2^6
sampling_rate = 44100
hop_length_seconds = hop_length/sampling_rate
subdivision_default = 96

def slice_tensor_spectrogram(idx, stft_tensor_spectrogram, cqt_tensor_spectrogram, pcp_tensor_spectrogram):
    print(("Bar {} of this song, on different representation").format(idx))
    cmap = cm.Greys
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].set_ylabel("Discretized frequencies")
    axs[0].pcolormesh(np.arange(stft_tensor_spectrogram[:,:,idx].shape[1]), np.arange(stft_tensor_spectrogram[:,:,idx].shape[0]), stft_tensor_spectrogram[:,:,idx],cmap=cmap)
    axs[0].set_title('STFT')
    axs[1].set_ylabel("Constant-Q bandwidths")
    axs[1].pcolormesh(np.arange(cqt_tensor_spectrogram[:,:,idx].shape[1]), np.arange(cqt_tensor_spectrogram[:,:,idx].shape[0]), cqt_tensor_spectrogram[:,:,idx],cmap=cmap)
    axs[1].set_title('CQT')
    axs[2].set_ylabel("Semi-tone scale")
    axs[2].pcolormesh(np.arange(pcp_tensor_spectrogram[:,:,idx].shape[1]), np.arange(pcp_tensor_spectrogram[:,:,idx].shape[0]), pcp_tensor_spectrogram[:,:,idx],cmap=cmap)
    axs[2].set_title('PCP/Chromas')
    for i in range(0,3):
        axs[i].set_xlabel("Frame index (barwise)")
    plt.show()
    
def nice_plot_factors(factors):
    shape = factors[0].shape
    plt.figure(figsize=(shape[1]/10,shape[0]/10))
    plt.pcolormesh(np.arange(shape[1]), np.arange(shape[0]), factors[0],
                   cmap=cm.gray)
    plt.title("W matrix (muscial content)")
    plt.xlabel("Atoms")
    plt.ylabel("Constant-Q bandwidths")
    #plt.gca().invert_yaxis()
    plt.show()
    
    shape = factors[1].shape
    plt.figure(figsize=(shape[0]/10,shape[1]/10))
    plt.pcolormesh(np.arange(shape[0]), np.arange(shape[1]), factors[1].T,
                   cmap=cm.gray)
    plt.title("H matrix: time at barscale (rythmic content)")
    plt.xlabel("Position in the bar (in frame indexes)")
    plt.ylabel("Atoms")
    #plt.gca().invert_yaxis()
    plt.show()
    
    shape = factors[2].shape
    plt.figure(figsize=(shape[0]/10,shape[1]/10))
    plt.pcolormesh(np.arange(shape[0]), np.arange(shape[1]), factors[2].T,
                   cmap=cm.gray)
    plt.title("Q matrix: Bar content feature")
    plt.xlabel("Index of the bar")
    plt.ylabel("Musical pattern index")
    #plt.gca().invert_yaxis()
    plt.show()
    
def nice_plot_autosimilarities(factor, tensor_spectrogram, annotations_frontiers_barwise):
    unfolded = tl.unfold(tensor_spectrogram, 2)
    autosimil = as_seg.get_autosimilarity(factor, transpose = True, normalize = False)
    normalized_autosimil = as_seg.get_autosimilarity(factor, transpose = True, normalize = True)
    signal_autosimil = as_seg.get_autosimilarity(unfolded, transpose = True, normalize = False)
    cmap = cm.Greys

    fig, axs = plt.subplots(1, 3, figsize=(15,5), sharey=False)#, sharex = True)
    axs[0].set_ylabel("Bar index")
    axs[0].pcolormesh(np.arange(signal_autosimil.shape[0]), np.arange(signal_autosimil.shape[0]), signal_autosimil, cmap = cmap)
    axs[0].set_title('Autosimilarity of the signal, barwise')
    axs[1].pcolormesh(np.arange(autosimil.shape[0]), np.arange(autosimil.shape[0]), autosimil, cmap = cmap)
    axs[1].set_title('Autosimilarity of Q matrix')
    axs[2].pcolormesh(np.arange(normalized_autosimil.shape[0]), np.arange(normalized_autosimil.shape[0]), normalized_autosimil, cmap = cmap)
    axs[2].set_title('Normalized Autosimilarity of Q matrix')
    for i in range(0,3):
        axs[i].set_xlabel("Bar index")
        axs[i].invert_yaxis()
        
    for x in annotations_frontiers_barwise[:-1]:
        for i in range(0,3):
            axs[i].plot([x,x], [0,signal_autosimil.shape[0] - 1], '-', linewidth=1, color = "blue")
            axs[i].plot([0,signal_autosimil.shape[0] - 1], [x,x], '-', linewidth=1, color = "blue")
    plt.show()
    
def compute_novelty_from_c_factor(factor):
    autosimilarity = as_seg.get_autosimilarity(factor, transpose = True, normalize = True)
    novelty = as_seg.novelty_computation(autosimilarity, 16)
    nov_ends = as_seg.select_highest_peaks_thresholded_indexes(novelty, percentage = 0.22)
    nov_ends.append(len(autosimilarity) - 1)
    
    max_pond_novelty = as_seg.values_as_slop(novelty, choice_func = max)
    max_pond_ends = as_seg.select_highest_peaks_thresholded_indexes(max_pond_novelty, percentage = 0.08)
    max_pond_ends.append(len(autosimilarity) - 1)
    
    min_pond_novelty = as_seg.values_as_slop(novelty, choice_func = min)
    min_pond_ends = as_seg.select_highest_peaks_thresholded_indexes(min_pond_novelty, percentage = 0.08)
    min_pond_ends.append(len(autosimilarity) - 1)
    
    mean_pond_novelty = as_seg.values_as_slop(novelty, choice_func = as_seg.mean)
    mean_pond_ends = as_seg.select_highest_peaks_thresholded_indexes(mean_pond_novelty, percentage = 0.08)
    mean_pond_ends.append(len(autosimilarity) - 1)
    return novelty, nov_ends, max_pond_novelty, max_pond_ends, min_pond_novelty, min_pond_ends, mean_pond_novelty, mean_pond_ends
    
def nice_plot_novelty_measures(stft_factor, cqt_factor, pcp_factor, annotations):  
    
    stft_res = compute_novelty_from_c_factor(stft_factor)
    cqt_res = compute_novelty_from_c_factor(cqt_factor)
    pcp_res = compute_novelty_from_c_factor(pcp_factor)
    
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].plot(np.arange(len(stft_res[0])),stft_res[0], color = "black")
    axs[0].set_title("Novelty on STFT")
    axs[1].plot(np.arange(len(cqt_res[0])),cqt_res[0], color = "black")
    axs[1].set_title("Novelty on CQT")
    axs[2].plot(np.arange(len(pcp_res[0])),pcp_res[0], color = "black")
    axs[2].set_title("Novelty on PCP")
    
    for i in range(3):
        axs[i].set_xlabel("Bar indexes")
        
    axs[0].set_ylabel("Novelty measure")
    
    for x in annotations:
        axs[0].plot([x,x], [0,np.amax(stft_res[0])], '-', linewidth=1, color = "#8080FF")
        axs[1].plot([x,x], [0,np.amax(cqt_res[0])], '-', linewidth=1, color = "#8080FF")
        axs[2].plot([x,x], [0,np.amax(pcp_res[0])], '-', linewidth=1, color = "#8080FF")
    for x in stft_res[1]:
        if x in annotations:
            axs[0].plot([x,x], [0,np.amax(stft_res[0])], '-', linewidth=1, color = "green")
        else:
            axs[0].plot([x,x], [0,np.amax(stft_res[0])], '-', linewidth=1, color = "orange")

    for x in cqt_res[1]:
        if x in annotations:
            axs[1].plot([x,x], [0,np.amax(cqt_res[0])], '-', linewidth=1, color = "green")
        else:
            axs[1].plot([x,x], [0,np.amax(cqt_res[0])], '-', linewidth=1, color = "orange")
    
    for x in pcp_res[1]:
        if x in annotations:
            axs[2].plot([x,x], [0,np.amax(pcp_res[0])], '-', linewidth=1, color = "green")
        else:
            axs[2].plot([x,x], [0,np.amax(pcp_res[0])], '-', linewidth=1, color = "orange")

    plt.show()
    
    
def nice_plot_novelty_peaks(factor, annotations):
    novelty_results = compute_novelty_from_c_factor(factor)
    
    fig, axs = plt.subplots(2, 2, figsize=(15,15))
    axs[0,0].plot(np.arange(len(novelty_results[0])),novelty_results[0], color = "black")
    axs[0,0].set_title("Novelty and frontiers on raw peaks.")
    axs[0,1].plot(np.arange(len(novelty_results[2])),novelty_results[2], color = "black")
    axs[0,1].set_title("Novelty and frontiers on peaks as maximal gap.")
    axs[1,0].plot(np.arange(len(novelty_results[4])),novelty_results[4], color = "black")
    axs[1,0].set_title("Novelty and frontiers on peaks as minimal gap.")
    axs[1,1].plot(np.arange(len(novelty_results[6])),novelty_results[6], color = "black")
    axs[1,1].set_title("Novelty and frontiers on peaks as mean of both gap.")
    
    for i in range(2):
        axs[i,0].set_ylabel("Novelty measure")
        axs[1,i].set_xlabel("Bar indexes")
    
    for x in annotations:
        for i in range(4):
            axs[i//2,i%2].plot([x,x], [0,np.amax(novelty_results[2*i])], '-', linewidth=1, color = "#8080FF")
    for x in novelty_results[1]:
        if x in annotations:
            axs[0,0].plot([x,x], [0,np.amax(novelty_results[0])], '-', linewidth=1, color = "green")
        else:
            axs[0,0].plot([x,x], [0,np.amax(novelty_results[0])], '-', linewidth=1, color = "orange")

    for x in novelty_results[3]:
        if x in annotations:
            axs[0,1].plot([x,x], [0,np.amax(novelty_results[2])], '-', linewidth=1, color = "green")
        else:
            axs[0,1].plot([x,x], [0,np.amax(novelty_results[2])], '-', linewidth=1, color = "orange")
    for x in novelty_results[5]:
        if x in annotations:
            axs[1,0].plot([x,x], [0,np.amax(novelty_results[4])], '-', linewidth=1, color = "green")
        else:
            axs[1,0].plot([x,x], [0,np.amax(novelty_results[4])], '-', linewidth=1, color = "orange")
    for x in novelty_results[7]:
        if x in annotations:
            axs[1,1].plot([x,x], [0,np.amax(novelty_results[6])], '-', linewidth=1, color = "green")
        else:
            axs[1,1].plot([x,x], [0,np.amax(novelty_results[6])], '-', linewidth=1, color = "orange")

    plt.show()
    
def nice_plot_convolution_measures(stft_factor, cqt_factor, pcp_factor, annotations):  
    stft_autosimilarity = as_seg.get_autosimilarity(stft_factor, transpose = True, normalize = True)
    stft_ends = dm.segments_to_frontiers(as_seg.dynamic_convolution_computation(stft_autosimilarity, mix = 1)[0])
    
    cqt_autosimilarity = as_seg.get_autosimilarity(cqt_factor, transpose = True, normalize = True)
    cqt_ends = dm.segments_to_frontiers(as_seg.dynamic_convolution_computation(cqt_autosimilarity, mix = 1)[0])
    
    pcp_autosimilarity = as_seg.get_autosimilarity(pcp_factor, transpose = True, normalize = True)
    pcp_ends = dm.segments_to_frontiers(as_seg.dynamic_convolution_computation(pcp_autosimilarity, mix = 1)[0])
    
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].pcolormesh(np.arange(stft_autosimilarity.shape[1]), np.arange(stft_autosimilarity.shape[0]), stft_autosimilarity, cmap=cm.gray)
    axs[0].set_title("Convolution on STFT")
    axs[1].pcolormesh(np.arange(cqt_autosimilarity.shape[1]), np.arange(cqt_autosimilarity.shape[0]), cqt_autosimilarity, cmap=cm.gray)
    axs[1].set_title("Convolution on CQT")
    axs[2].pcolormesh(np.arange(pcp_autosimilarity.shape[1]), np.arange(pcp_autosimilarity.shape[0]), pcp_autosimilarity, cmap=cm.gray)
    axs[2].set_title("Convolution on PCP")
    for i in range(3):
        axs[i].invert_yaxis()
        axs[i].set_xlabel("Bar indexes")
    axs[0].set_ylabel("Bar indexes, with frontiers as vertial bars.")
    
    for x in annotations:
        for i in range(3):
            axs[i].plot([x,x], [0,len(stft_factor) - 1], '-', linewidth=1, color = "#8080FF")
    for x in stft_ends:
        if x in annotations:
            axs[0].plot([x,x], [0,len(stft_factor) - 1], '-', linewidth=1, color = "green")
        else:
            axs[0].plot([x,x], [0,len(stft_factor) - 1], '-', linewidth=1, color = "orange")
    for x in cqt_ends:
        if x in annotations:
            axs[1].plot([x,x], [0,len(stft_factor) - 1], '-', linewidth=1, color = "green")
        else:
            axs[1].plot([x,x], [0,len(stft_factor) - 1], '-', linewidth=1, color = "orange")
    for x in pcp_ends:
        if x in annotations:
            axs[2].plot([x,x], [0,len(stft_factor) - 1], '-', linewidth=1, color = "green")
        else:
            axs[2].plot([x,x], [0,len(stft_factor) - 1], '-', linewidth=1, color = "orange")
    plt.show()

def nice_plot_convolution_frontiers(factor, annotations):
    raise err.OutdatedBehaviorException("Uses convolution with novelty measure, which is not supported anymore. To track and correct.")
    autosimilarity = as_seg.get_autosimilarity(factor, transpose = True, normalize = True)
    conv_ends = dm.segments_to_frontiers(as_seg.dynamic_convolution_computation(autosimilarity, mix = 1)[0])
    conv_nov_ends = dm.segments_to_frontiers(as_seg.dynamic_convolution_computation(autosimilarity, mix = 0.5)[0])

    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].pcolormesh(np.arange(autosimilarity.shape[1]), np.arange(autosimilarity.shape[0]), autosimilarity, cmap=cm.gray)
    axs[0].set_title("Mesure de convolution")
    axs[1].pcolormesh(np.arange(autosimilarity.shape[1]), np.arange(autosimilarity.shape[0]), autosimilarity, cmap=cm.gray)
    axs[1].set_title("Mesure de convolution avec nouveauté")

    for i in range(2):
        axs[i].invert_yaxis()
        axs[i].set_xlabel("Nombre de mesures")
    axs[0].set_ylabel("Nombre de mesures")
    
    for x in annotations:
        for i in range(2):
            axs[i].plot([x,x], [0,len(factor) - 1], '-', linewidth=1, color = "#8080FF")
    for x in conv_ends:
        if x in annotations:
            axs[0].plot([x,x], [0,len(factor) - 1], '-', linewidth=1, color = "green")
        else:
            axs[0].plot([x,x], [0,len(factor) - 1], '-', linewidth=1, color = "orange")
    for x in conv_nov_ends:
        if x in annotations:
            axs[1].plot([x,x], [0,len(factor) - 1], '-', linewidth=1, color = "green")
        else:
            axs[1].plot([x,x], [0,len(factor) - 1], '-', linewidth=1, color = "orange")
    plt.show()
    
def print_dataframe_results_novelty(stft_factor, cqt_factor, pcp_factor, bars, references_segments):
    data = ["STFT", "CQT", "PCP"]
    tolerance = ["0.5 seconds","3 seconds"]
    
    lines = np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'])
    types = []
    tol = []
    for i in data:
        for j in tolerance:
            types.append(i)
            tol.append(j)
    col = [np.array(types), np.array(tol)]
    
    arr = []
    for tab in [stft_factor, cqt_factor, pcp_factor]:
        nov_ends = compute_novelty_from_c_factor(tab)[1]
        nov_seg_time = dm.segments_from_bar_to_time(dm.frontiers_to_segments(nov_ends), bars)
        for window_len in [0.5, 3]:
            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, nov_seg_time, window_length = window_len)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, nov_seg_time, window_length = window_len)
            arr.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
    return pd.DataFrame(np.array(arr), index=col, columns=lines)

def print_dataframe_results_novelty_peaks(factor, bars, references_segments):
    data = ["Raw peaks", "Maximal gap", "Minimal gap", "Mean both gaps"]
    tolerance = ["0.5 seconds","3 seconds"]
    
    lines = np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'])
    types = []
    tol = []
    for i in data:
        for j in tolerance:
            types.append(i)
            tol.append(j)
    col = [np.array(types), np.array(tol)]
    
    arr = []
    _, ends, _, max_ends, _, min_ends, _, mean_ends, = compute_novelty_from_c_factor(factor)
    for end in [ends, max_ends, min_ends, mean_ends]:
        nov_seg_time = dm.segments_from_bar_to_time(dm.frontiers_to_segments(end), bars)
        for window_len in [0.5, 3]:
            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, nov_seg_time, window_length = window_len)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, nov_seg_time, window_length = window_len)
            arr.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
    return pd.DataFrame(np.array(arr), index=col, columns=lines)

def print_dataframe_results_convolution_features(stft_factor, cqt_factor, pcp_factor, bars, references_segments):
    data = ["STFT", "CQT", "PCP"]
    tolerance = ["0.5 seconds","3 seconds"]
    
    lines = np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'])
    types = []
    tol = []
    for i in data:
        for j in tolerance:
            types.append(i)
            tol.append(j)
    col = [np.array(types), np.array(tol)]
    
    arr = []
    for tab in [stft_factor, cqt_factor, pcp_factor]:
        autosimilarity = as_seg.get_autosimilarity(tab, transpose = True, normalize = True)
        segments = as_seg.dynamic_convolution_computation(autosimilarity, mix = 1)[0]
        segments_in_time = dm.segments_from_bar_to_time(segments, bars)
        for window_len in [0.5, 3]:
            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = window_len)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = window_len)
            arr.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
    return pd.DataFrame(np.array(arr), index=col, columns=lines)

def print_dataframe_results_convolution_pcp(factor, bars, references_segments):
    raise err.OutdatedBehaviorException("Uses convolution with novelty measure, which is not supported anymore. To track and correct.")
    data = ["Convolution", "Convolution avec nouveauté"]
    tolerance = ["0.5 seconds","3 seconds"]
    
    lines = np.array(['Vrai Positifs','Faux Positifs','Faux Négatifs','Precision', 'Rappel', 'F mesure'])
    types = []
    tol = []
    for i in data:
        for j in tolerance:
            types.append(i)
            tol.append(j)
    col = [np.array(types), np.array(tol)]
    
    arr = []
    autosimilarity = as_seg.get_autosimilarity(factor, transpose = True, normalize = True)
    for mix in [1, 0.5]:
        segments = as_seg.dynamic_convolution_computation(autosimilarity, mix = mix)[0]
        segments_in_time = dm.segments_from_bar_to_time(segments, bars)
        for window_len in [0.5, 3]:
            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = window_len)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = window_len)
            arr.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
    return pd.DataFrame(np.array(arr), index=col, columns=lines)

def convolution_parameter_on_all_rwc(param_range, annotations_type = "MIREX10", subdivision = subdivision_default, penalty_func = "modulo4", convolution_type = "full", persisted_path = default_persisted_path):
    folder = rwc_folder_path

    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    paths = scr.load_RWC_dataset(folder, annotations_type)

    ranks = [12,32,32]
    feature = "pcp"
    zero_five = []
    three = []
    for song_and_annotations in paths:
        song_name = song_and_annotations[0].replace(".wav","")
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(folder, song_name), feature, hop_length)
        
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)

        q_factor = scr.NTD_decomp_as_script(persisted_path, "_{}_{}_{}_{}".format(song_name, "pcp", "chromas", subdivision), tensor_spectrogram, [12,32,32], init = "chromas")[1][2]

        autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)
        this_song = []
        for param in param_range:
            segments = as_seg.dynamic_convolution_computation(autosimilarity, mix = 1, penalty_weight = param, penalty_func = penalty_func, convolution_type = convolution_type)[0] #, fixed_ponderation = fixed_ponderation
            segments_in_time = dm.segments_from_bar_to_time(segments, bars)
            
            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            this_song.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
            
        zero_five.append(this_song)
    all_lines = []
    for param in range(len(param_range)):
        line = []
        for i in range(6):
            line.append(np.mean(np.array(zero_five)[:,param,i]))
        all_lines.append(line)
    dataframe = pd.DataFrame(np.array(all_lines), index = param_range, columns=np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure']))
    display(dataframe.style.bar(subset=["F measure"], color='#5fba7d'))

def run_and_show_results_on_rwc():
    arr = []
    for feature in ["stft", "cqt", "pcp"]:
        zero_five, three = run_on_all_rwc(feature)
        zero_five_means = [np.mean(np.array(zero_five)[:,i]) for i in range(6)]
        arr.append(zero_five_means)
        three_means = [np.mean(np.array(three)[:,i]) for i in range(6)]
        arr.append(three_means)
        
    data = ["STFT", "CQT", "PCP"]
    tolerance = ["0.5 seconds","3 seconds"]
    
    lines = np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'])
    types = []
    tol = []
    for i in data:
        for j in tolerance:
            types.append(i)
            tol.append(j)
    col = [np.array(types), np.array(tol)]    
    return pd.DataFrame(np.array(arr), index=col, columns=lines)


def cross_validation_for_param(learning_dataset, testing_dataset, measure, annotations_type = "MIREX10",
                               ker_range = [8,12,16,20,24],percentage_range = range(2,30), subdivision = subdivision_default, persisted_path = default_persisted_path):
    all_res = []
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)

    ranks = [12,32,32]
    feature="pcp"
    hop_length = 32
    
    for song_and_annotations in learning_dataset:
        current_res = []
        #print(song_and_annotations[0])
        song_name = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(learning_dataset, song_name), feature, hop_length)

        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)  
        persisted_arguments = "_{}_{}_{}_{}".format(song_name, feature, "chromas", subdivision)
    
        q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, [12,32,32], init = "chromas")[1][2]

        autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)
        
        for ker_size in ker_range:
            novelty = as_seg.novelty_computation(autosimilarity, ker_size)
            for percentage in percentage_range:
                if measure == "novelty":
                    ends = as_seg.select_highest_peaks_thresholded_indexes(novelty, percentage = 0.01 * percentage)
                    ends.append(len(autosimilarity) - 1)
                    current_res.append(score_from_frontiers(ends, bars, references_segments))
                elif measure == "max":
                    max_pond_novelty = as_seg.values_as_slop(novelty, choice_func = max)
                    max_ends = as_seg.select_highest_peaks_thresholded_indexes(max_pond_novelty, percentage = 0.01 * percentage)
                    max_ends.append(len(autosimilarity) - 1)
                    current_res.append(score_from_frontiers(max_ends, bars, references_segments))
                elif measure == "min":                
                    min_pond_novelty = as_seg.values_as_slop(novelty, choice_func = min)
                    min_ends = as_seg.select_highest_peaks_thresholded_indexes(min_pond_novelty, percentage = 0.01 * percentage)
                    min_ends.append(len(autosimilarity) - 1)
                    current_res.append(score_from_frontiers(min_ends, bars, references_segments))
                elif measure == "mean":               
                    mean_pond_novelty = as_seg.values_as_slop(novelty, choice_func = as_seg.mean)
                    mean_ends = as_seg.select_highest_peaks_thresholded_indexes(mean_pond_novelty, percentage = 0.01 * percentage)
                    mean_ends.append(len(autosimilarity) - 1)
                    current_res.append(score_from_frontiers(mean_ends, bars, references_segments))

        all_res.append(current_res)
    
    ker, perc, idx_max = find_best_couple_from_results(all_res, ker_range, percentage_range)
    line = [ker,perc]
    for i in range(6):
        line.append(np.mean(np.array(all_res)[:,idx_max,i]))
    dataframe_param = pd.DataFrame(np.array(line), columns = ["Best couple:"], index=np.array(['Kernel size', 'Percentage','True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure']))
    display(dataframe_param.T)  
    tp,fp,fn,p,r,f = results_from_this_param(testing_dataset, measure, ranks, ker, perc, references_segments, annotations_type)
    dataframe = pd.DataFrame(np.array([tp,fp,fn,p,r,f]), columns = ["On the test set:"], index=np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure']))
    display(dataframe.T)  
    return all_res

def print_dataframe_results_of_parameters(results, kernel_range, percentage_range):
    data = ["Size of the kernel: {}".format(i) for i in kernel_range]
    metrics = ['Prec', 'Rap', 'Fmes']
    col = np.array([i for i in percentage_range])
    types = []
    tol = []
    for i in data:
        for j in metrics:
            types.append(i)
            tol.append(j)
    lines = [np.array(types), np.array(tol)]
    
    res = np.array(results)
    arr = []
    for ker_idx in range(len(kernel_range)):
        for param in range(3,6):
            line = []
            for perc_idx in range(len(percentage_range)):
                line.append(round(np.mean(res[:,ker_idx * len(percentage_range) + perc_idx, param]),4))
            arr.append(line)
    
    df = pd.DataFrame(np.array(arr), index=lines, columns=col)  
    df.columns.name = 'Percentage val'
    return df.T.style.background_gradient(cmap='Greys')

def find_best_couple_from_results(results, ker_range, percentage_range):
    all_mean = []
    res = np.array(results)
    for i in range(res.shape[1]):
        all_mean.append(np.mean(res[:,i,5]))
    idx_max = np.argmax(all_mean)
    return ker_range[idx_max//len(percentage_range)], percentage_range[idx_max%len(percentage_range)], idx_max

def score_from_frontiers(ends, bars, references_segments):
    segments = dm.frontiers_to_segments(ends)
    time_segments = dm.segments_from_bar_to_time(segments, bars)
    tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, time_segments, window_length = 0.5)
    prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, time_segments, window_length = 0.5)
    return [tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)]

def results_from_this_param(dataset, measure, ranks, ker_size, percentage, references_segments, annotations_type, subdivision = subdivision_default, persisted_path = default_persisted_path):
    results = []
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    subdivision = subdivision_default
    for song_and_annotations in dataset:
        current_res = []
        #print(song_and_annotations[0])
        song_name = song_and_annotations[0].replace(".wav","")
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
    
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(dataset, song_name), "pcp", hop_length)

        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)  
        persisted_arguments = "_{}_{}_{}_{}".format(song_name, "pcp", "chromas", subdivision)
    
        q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, [12,32,32], init = "chromas")[1][2]

        autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)
        
        novelty = as_seg.novelty_computation(autosimilarity, ker_size)
        if measure == "novelty":
            ends = as_seg.select_highest_peaks_thresholded_indexes(novelty, percentage = 0.01 * percentage)
            ends.append(len(autosimilarity) - 1)
            results.append(score_from_frontiers(ends, bars, references_segments))

        if measure == "max":
            max_pond_novelty = as_seg.values_as_slop(novelty, choice_func = max)
            max_ends = as_seg.select_highest_peaks_thresholded_indexes(max_pond_novelty, percentage = 0.01 * percentage)
            max_ends.append(len(autosimilarity) - 1)
            results.append(score_from_frontiers(max_ends, bars, references_segments))
        
        if measure == "min":
            min_pond_novelty = as_seg.values_as_slop(novelty, choice_func = min)
            min_ends = as_seg.select_highest_peaks_thresholded_indexes(min_pond_novelty, percentage = 0.01 * percentage)
            min_ends.append(len(autosimilarity) - 1)
            results.append(score_from_frontiers(min_ends, bars, references_segments))
        
        if measure == "mean":
            mean_pond_novelty = as_seg.values_as_slop(novelty, choice_func = as_seg.mean)
            mean_ends = as_seg.select_highest_peaks_thresholded_indexes(mean_pond_novelty, percentage = 0.01 * percentage)
            mean_ends.append(len(autosimilarity) - 1)
            results.append(score_from_frontiers(mean_ends, bars, references_segments))
    
    res = np.array(results)
    return (np.mean(res[:,0]),np.mean(res[:,1]),np.mean(res[:,2]),round(np.mean(res[:,3]),4),round(np.mean(res[:,4]),4),round(np.mean(res[:,5]),4))
    
def plot_3d_parameters_study(results, ker_range, percentage_range):
    kernel = []
    percentages = []
    for i in ker_range:
        for j in percentage_range:
            kernel.append(i)
            percentages.append(j)

    tab = []
    res = np.array(results)
    for ker_idx in range(len(ker_range)):
        percs = []
        for perc_idx in range(len(percentage_range)):
            percs.append(round(np.mean(res[:,ker_idx * len(percentage_range) + perc_idx, 5]),4))
        tab.append(percs)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_wireframe(X=np.array(kernel).reshape((len(ker_range),len(percentage_range))), 
                    Y=np.array(percentages).reshape((len(ker_range),len(percentage_range))), 
                    Z=np.array(tab))
    ax.set_xlabel('Kernel size')
    ax.set_ylabel('Percentage')

    plt.show()
    
    
def compare_chromas_tucker_decomp(song_number, ranks, subdivision = subdivision_default, persisted_path = default_persisted_path):

    bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(rwc_folder_path, song_number), "pcp", hop_length)
        
    tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
            
    tucker_persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "tucker", subdivision)
    tk_core, tk_factors = scr.NTD_decomp_as_script(persisted_path, tucker_persisted_arguments, tensor_spectrogram, ranks, init = "tucker")
    
    chromas_persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision)
    chr_core, chr_factors = scr.NTD_decomp_as_script(persisted_path, chromas_persisted_arguments, tensor_spectrogram, ranks, init = "chromas")
    
    plot_me_two_spec(chr_factors[0], tk_factors[0],
                     title_1 = "$W$, fixed to identity (Id12)", title_2 = "$W$, treated as a factor",
                     xlabel = "Musical atoms", ylabel = "Semi-tone scale", cmap = cm.Greys)
    
    plot_me_two_spec(chr_factors[1].T, tk_factors[1].T,
                     title_1 = "$H^T$, when $W =$ Id12", title_2 = "$H^T$, when $W$ is a factor",
                     xlabel = "Inner-bar time (number of frames)", ylabel = "Rhytmic atoms", cmap = cm.Greys)
    
    plot_me_two_spec(chr_factors[2].T, tk_factors[2].T,
                     title_1 = "$Q^T$, when $W =$ Id12", title_2 = "$Q^T$, when $W$ is a factor",
                     xlabel = "Bars", ylabel = "Musical patterns", cmap = cm.Greys)
    
    chr_autosimilarity = as_seg.get_autosimilarity(chr_factors[2], transpose = True, normalize = True)
    tk_autosimilarity = as_seg.get_autosimilarity(tk_factors[2], transpose = True, normalize = True)

    plot_me_two_spec(chr_autosimilarity, tk_autosimilarity,
                     title_1 = "Autosimilarity of $Q$, when $W =$ Id12", title_2 = "Autosimilarity of $Q$, when $W$ is a factor",
                     xlabel = "Bars", ylabel = "Bars", cmap = cm.Greys)

def compare_chromas_tucker_dataset(dataset_name, ranks, annotations_type = "MIREX10", subdivision = subdivision_default, persisted_path = default_persisted_path):
    dataset = "{}\\{}".format(root_rwc_folder, dataset_name)
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)

    paths = scr.load_RWC_dataset(dataset, annotations_type)

    for song_and_annotations in paths:
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        #references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(dataset, song_number), "pcp", hop_length)
        barwise_annot = dm.frontiers_from_time_to_bar(np.array(annotations)[:,1], bars)
        
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        
        tucker_persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "tucker", subdivision)
        tk_q_factor = scr.NTD_decomp_as_script(persisted_path, tucker_persisted_arguments, tensor_spectrogram, ranks, init = "tucker")[1][2]
        
        chromas_persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision)
        chr_q_factor = scr.NTD_decomp_as_script(persisted_path, chromas_persisted_arguments, tensor_spectrogram, ranks, init = "chromas")[1][2]

        tk_autosimilarity = as_seg.get_autosimilarity(tk_q_factor, transpose = True, normalize = True)
        chr_autosimilarity = as_seg.get_autosimilarity(chr_q_factor, transpose = True, normalize = True)

        chr_q_permut = permutate_factor(chr_q_factor)
        tk_q_permut = permutate_factor(tk_q_factor)

        plot_me_two_spec(chr_q_factor.T[chr_q_permut],tk_q_factor.T[tk_q_permut],
                         title_1 = "Permutation of Q on song {}, chromas decomposition".format(song_and_annotations[0]), title_2 = "Permutation of Q on song {}, tucker decomposition".format(song_and_annotations[0]),
                         xlabel = "Bars", ylabel = "Atoms",cmap = cm.Greys, annotations_barwise = barwise_annot)
        
        plot_me_two_spec(chr_autosimilarity, tk_autosimilarity,
                     title_1 = "Autosimilarity of Q on song {}, chromas decomposition".format(song_and_annotations[0]), title_2 = "Autosimilarity of Q on song {}, tucker decomposition".format(song_and_annotations[0]),
                     xlabel = "Bars", ylabel = "Bars", cmap = cm.Greys, annotations_barwise = barwise_annot)
    
def plot_me_two_spec(spec_1, spec_2, title_1 = None, title_2 = None, xlabel = None, ylabel = None, cmap = cm.gray,
                     annotations_barwise = None):
    fig, axs = plt.subplots(1, 2, figsize=(14,7))
    padded_spec_1 = pad_factor(spec_1)
    padded_spec_2 = pad_factor(spec_2)
    axs[0].pcolormesh(np.arange(padded_spec_1.shape[1]), np.arange(padded_spec_1.shape[0]), padded_spec_1, cmap=cmap)
    axs[0].set_title(title_1)
    axs[0].set_ylabel(ylabel)
    axs[1].pcolormesh(np.arange(padded_spec_2.shape[1]), np.arange(padded_spec_2.shape[0]), padded_spec_2, cmap=cmap)
    axs[1].set_title(title_2)

    for i in range(2):
        axs[i].invert_yaxis()
        axs[i].set_xlabel(xlabel)
    
    if annotations_barwise != None:
        for x in annotations_barwise:
            for i in range(2):
                if spec_1.shape[0] == spec_1.shape[1]:
                    axs[i].plot([x,x], [0,spec_1.shape[0]], '-', linewidth=1, color = "#8080FF")
                    axs[i].plot([0,spec_1.shape[0]], [x,x], '-', linewidth=1, color = "#8080FF")
                else:
                    axs[i].plot([x,x], [0,spec_1.shape[0]], '-', linewidth=1, color = "#8080FF")

    plt.show()
    
def compare_autosimil_ranks(dataset_name, ranks_rhythm, ranks_patterns, cmap = cm.Greys, W = "chromas", annotations_type = "MIREX10", penalty_weight = 0, subdivision = subdivision_default, persisted_path = default_persisted_path):
    dataset = "{}\\{}".format(root_rwc_folder, dataset_name)
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)

    paths = scr.load_RWC_dataset(dataset, annotations_type)
    zero_five = []
    three = []
    
    rhythms = []
    patterns = []
    for i in ranks_patterns:
        for j in ranks_rhythm:
            patterns.append("Rank of Q:" + str(i))
            rhythms.append("Rank of H:" + str(j))
    col = [np.array(patterns), np.array(rhythms)]
    
    for song_and_annotations in paths:
        printmd('**Current song: {}**'.format(song_and_annotations[0]))
        zero_five_song = []
        three_song = []
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(dataset, song_number), "pcp", hop_length)
        
        barwise_annot = dm.frontiers_from_time_to_bar(np.array(annotations)[:,1], bars)
        
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        for rank_pattern in ranks_patterns:
            nb_ranks_rhythm = len(ranks_rhythm)
            fig, axs = plt.subplots(1, nb_ranks_rhythm, figsize=(20,int(20/nb_ranks_rhythm)))
            axs[0].set_ylabel("Measures")
            for i, rank_rhythm in enumerate(ranks_rhythm):
                ranks = [12, rank_rhythm, rank_pattern]
                persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", W, subdivision)
                chr_q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = W)[1][2]
                chr_autosimilarity = as_seg.get_autosimilarity(chr_q_factor, transpose = True, normalize = True)
                chr_autosimilarity = pad_factor(chr_autosimilarity)
                axs[i].pcolormesh(np.arange(chr_autosimilarity.shape[1]), np.arange(chr_autosimilarity.shape[0]), chr_autosimilarity, cmap=cmap)
                axs[i].set_title("Rank of H: {}, rank of Q: {}".format(rank_rhythm, rank_pattern))
                axs[i].invert_yaxis()
                
                for x in barwise_annot:
                    axs[i].plot([x,x], [0,len(chr_autosimilarity) - 1], '-', linewidth=1, color = "#8080FF")
                    axs[i].plot([0,len(chr_autosimilarity) - 1], [x,x], '-', linewidth=1, color = "#8080FF")
                
                segments = as_seg.dynamic_convolution_computation(chr_autosimilarity, mix = 1, min_size = 1, penalty_weight = penalty_weight)[0]
                frontiers = dm.segments_to_frontiers(segments)
                
                for x in frontiers:
                    if x in barwise_annot:
                        axs[i].plot([x,x], [0,len(chr_autosimilarity) - 1], '-', linewidth=1, color = "green")
                    else:
                        axs[i].plot([x,x], [0,len(chr_autosimilarity) - 1], '-', linewidth=1, color = "orange")
                        
                segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                
                tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                zero_five_song.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
                axs[i].set_xlabel("0.5 seconds score:\nTP: {}, FP: {}, FN: {}\nP: {}, R: {}, F1: {}".format(int(tp),int(fp),int(fn),round(prec,4),round(rap,4),round(f_mes,4)))

                tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
                prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
                three_song.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
            plt.show()
            
        dataframe_zero_five = pd.DataFrame(np.array(zero_five_song), columns = [np.array(["0.5 seconds for song: {}".format(song_number) for i in range(6)]),np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'])], index = col)
        display(dataframe_zero_five)
        
        dataframe_three = pd.DataFrame(np.array(three_song), columns = [np.array(["3 seconds score for song: {}".format(song_number) for i in range(6)]),np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'])], index = col)
        display(dataframe_three) 
        
        zero_five.append(zero_five_song)
        three.append(three_song)

    return zero_five, three

def printmd(string):
    display(Markdown(string))
    
def return_worst_songs(results, k):
    to_return = []
    paths = scr.load_RWC_dataset(rwc_folder_path, "MIREX10")
    f_mes = return_optimal_fmes(results)
    tabs = np.argpartition(f_mes, k)[:k]
    for i in range(k):
        to_return.append((np.array(paths)[tabs[i],0],f_mes[tabs[i]]))
    return to_return
    
    
def compute_ranks_RWC(ranks_rhythm, ranks_patterns, W = "chromas", annotations_type = "MIREX10", penalty_weight = 1, subdivision = subdivision_default, penalty_func = "modulo8", convolution_type = "full", persisted_path = default_persisted_path):
    dataset = rwc_folder_path
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)

    paths = scr.load_RWC_dataset(dataset, annotations_type)

    zero_five = []
    three = []
    
    rhythms = []
    patterns = []
    for i in ranks_patterns:
        for j in ranks_rhythm:
            patterns.append("Rang Q:" + str(i))
            rhythms.append("Rang H:" + str(j))
    col = [np.array(patterns), np.array(rhythms)]
    
    for song_and_annotations in paths:
        #printmd('**Chanson courante: {}**'.format(song_and_annotations[0]))
        zero_five_song = []
        three_song = []
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(dataset, song_number), "pcp", hop_length)
                
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        for rank_pattern in ranks_patterns:
            for i, rank_rhythm in enumerate(ranks_rhythm):
                ranks = [12, rank_rhythm, rank_pattern]
                persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", W, subdivision)
                chr_q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = W)[1][2]
                chr_autosimilarity = as_seg.get_autosimilarity(chr_q_factor, transpose = True, normalize = True)

                segments = as_seg.dynamic_convolution_computation(chr_autosimilarity, mix = 1, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
                segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                
                tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                zero_five_song.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])

                tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
                prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
                three_song.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
        
        zero_five.append(zero_five_song)
        three.append(three_song)
    
    zero_five_all_mean = []
    three_all_mean = []
    for ranks_couple in range(len(zero_five[0])):
        zero_five_line = []
        three_line = []
        for i in range(6):
            zero_five_line.append(np.mean(np.array(zero_five)[:,ranks_couple,i]))
            three_line.append(np.mean(np.array(three)[:,ranks_couple,i]))
        zero_five_all_mean.append(zero_five_line)
        three_all_mean.append(three_line)
            
    dataframe_zero_five = pd.DataFrame(np.array(zero_five_all_mean), columns = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], index = col)
    dataframe_zero_five.columns.name = "0.5 seconds results"
    display(dataframe_zero_five.style.bar(subset=["F measure"], color='#5fba7d'))
    
    dataframe_three = pd.DataFrame(np.array(three_all_mean), columns = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], index = col)
    dataframe_three.columns.name = "3 seconds results"
    display(dataframe_three.style.bar(subset=["F measure"], color='#5fba7d'))
    return zero_five, three

def return_optimal_ranks(results):
    argmaxs = []
    for song in range(len(results)):
        all_f = np.array(results)[song,:,5]
        argmaxs.append(np.argmax(all_f))
    return argmaxs

def return_optimal_fmes(results):
    fmes_maxs = []
    for song in range(len(results)):
        all_f = np.array(results)[song,:,5]
        fmes_maxs.append(np.amax(all_f))
    return fmes_maxs

def best_f_one_score_rank(results):
   # Best F1 Score on entire RWC
    best_NTD = []
    array = np.array(results)
    argmaxs = return_optimal_ranks(results)
    for i in range(len(results)):
        best_NTD.append(array[i,argmaxs[i],:])
    tab = [np.mean(np.array(best_NTD)[:,i]) for i in range(6)]
    best_dataframe = pd.DataFrame(np.array(tab), index = [np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'])], columns = ["Maximizing the f measure on each song:"])
    display(best_dataframe.T)
    return tab

# Plot both side by side.
def plot_3d_ranks_study(results, ranks_rhythm, ranks_pattern):
    rhythms = []
    patterns = []
    for i in ranks_pattern:
        for j in ranks_rhythm:
            patterns.append(i)
            rhythms.append(j)
            
    tab = np.zeros((len(ranks_pattern), len(ranks_rhythm)))
    res = np.array(results)
    argmaxs = return_optimal_ranks(results)

    for song in range(len(results)):
        tab[argmaxs[song]//len(ranks_rhythm), argmaxs[song]%len(ranks_rhythm)] += 1

    min_gap_pattern = min(min([ranks_rhythm[i] - ranks_rhythm[i-1] for i in range(1, len(ranks_rhythm))]), min([ranks_pattern[i] - ranks_pattern[i-1] for i in range(1, len(ranks_pattern))]))

    #fig = plt.figure()
    #ax = Axes3D(fig)

    # Create a figure for plotting the data as a 3D histogram.
    #fig = plt.figure()
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_wireframe(X=np.array(patterns).reshape((len(ranks_pattern),len(ranks_rhythm))), 
                    Y=np.array(rhythms).reshape((len(ranks_pattern), len(ranks_rhythm))), 
                    Z=np.array(tab))
    ax.set_xlabel('Ranks of $Q$')
    ax.set_ylabel('Ranks of $H$')

    #plt.show()
    ax = fig.add_subplot(1, 2,2, projection='3d')

    data_array = np.array(tab)
    

    
    # Create an X-Y mesh of the same dimension as the 2D data. You can
    # think of this as the floor of the plot.
    x_data, y_data = np.meshgrid( np.array(ranks_pattern),
                                 np.array(ranks_rhythm) )
    
    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays:
    # x_data, y_data, z_data. The following call boils down to picking
    # one entry from each array and plotting a bar to from
    # (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.T.flatten()
    ax.bar3d( x_data,
              y_data,
              np.zeros(len(z_data)),
              min_gap_pattern - 1, min_gap_pattern - 1, z_data )
    ax.set_xlabel('Ranks of $Q$')
    ax.set_ylabel('Ranks of $H$')

    # Finally, display the plot.
    plt.show()
    
def plot_f_mes_histogram(results):
    plt.figure(figsize=(10,5))
    plt.title("Distribution histogram of the best f measures (with oracle ranks).")
    plt.xlabel("Best f measure (with 0.05 step)")
    plt.ylabel("Number of songs in this bin")
    plt.hist(return_optimal_fmes(results), bins = [i/100 for i in range(0,100, 5)])

def focus_on_bar(core, factors, bar, cmap = cm.Greys):
    h_permut = permutate_factor(factors[1])
    q_permut = permutate_factor(factors[2])
    permutated_factors = copy.deepcopy(factors[:2])
    permutated_factors[1] = (factors[1][:,h_permut])
    plot_permuted_factor(factors[2], title = "$Q$ matrix")
    printmd("**Focus on bar {}:**".format(bar))

    patt_idx = np.argmax(factors[2].T[:,2])
    plot_me_this_spectrogram(core[:,h_permut,patt_idx], title = "Slice {} of the core (most important pattern for this bar)".format(q_permut.index(patt_idx)), x_axis="Rhythmic atoms", y_axis="Chromas indexes")#, title = "Core, slice {} (slice {} in original decomposition order)".format(i, idx), x_axis = "Time atoms", y_axis = "Freq Atoms", cmap = cm.Greys)
    pattern = tl.tenalg.multi_mode_dot(core[:,h_permut,patt_idx], permutated_factors, transpose = False)
    plot_me_this_spectrogram(pattern, title = "Previous musical pattern, decomposed", x_axis="Time, in frames indexes, in the bar", y_axis="Chromas indexes")
    
    pattern = np.zeros((factors[0].shape[0], factors[1].shape[0]))
    for idx, val in enumerate(factors[2].T[q_permut,bar]):
        pattern += tl.tenalg.multi_mode_dot(val * core[:,h_permut,q_permut[idx]], permutated_factors, transpose = False)
    plot_me_this_spectrogram(pattern, title = "Entire bar {}".format(bar), x_axis="Time, in frame indexes, in the bar", y_axis="Chromas indexes")

def study_impact_line_jeremy(factors, core, mode, norm = 1):
    other_modes = [0,1,2]
    other_modes.pop(mode)
    a = factors[mode]@((tl.unfold(core, mode)@np.kron(factors[other_modes[0]], factors[other_modes[1]]).T))
    norm_tensor = tl.norm(tl.tenalg.multi_mode_dot(core, factors, transpose = False), norm)
    return [np.linalg.norm(i,norm)/norm_tensor for i in a]

def study_impact_column_jeremy(factors, core, mode, norm = 1):
    norm_tensor = tl.norm(tl.tenalg.multi_mode_dot(core, factors, transpose = False), norm)
    permutation = permutate_factor(factors[mode])
    other_modes = [0,1,2]
    other_modes.pop(mode)
    U = factors[mode]
    V = tl.unfold(core, mode)@np.kron(factors[other_modes[0]], factors[other_modes[1]]).T
    return [tl.norm(np.dot(np.reshape(U[:,i], (U.shape[0],1)), np.reshape(V[i,:], (1,V.shape[1]))), norm)/norm_tensor for i in permutation]

def study_impact_column_axel(fac, core, factor_number, columns, transpose = False, plotting = False, norm = 1):
    permutation = permutate_factor(fac[factor_number])
    permutated_factor = fac[factor_number][:,permutation]
    cropped_factors = copy.deepcopy(fac)
    cropped_factors[factor_number] = permutated_factor

    cropped_factors[factor_number] = np.zeros_like(fac[factor_number])
    if transpose:
        cropped_factors[factor_number][columns] = permutated_factor[columns]
    else:
        cropped_factors[factor_number][:,columns] = permutated_factor[:,columns]
    if plotting:
        plot_permuted_factor(cropped_factors[factor_number])
    if factor_number == 0:
        permutated_core = core[permutation,:,:]
    if factor_number == 1:
        permutated_core = core[:,permutation,:]
    if factor_number == 2:
        permutated_core = core[:,:,permutation]
    return tl.norm(tl.tenalg.multi_mode_dot(permutated_core, cropped_factors, transpose = False), 1) / tl.norm(tl.tenalg.multi_mode_dot(core, fac, transpose = False), 1)

def compare_decomp_on_spectrograms(dataset_name, cmap = cm.Greys, penalty_weight = 0, fmin = 98, subdivision = subdivision_default, persisted_path = default_persisted_path):
    raise err.OutdatedBehaviorException("Comparison with an old version of the spectrogram, which is not supported anymore.")
    dataset = "{}\\{}".format(root_rwc_folder, dataset_name)
    annotations_type = "MIREX10"
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)

    paths = scr.load_RWC_dataset(dataset, annotations_type)
    old_zero_five = []
    old_three = []
    other_zero_five = []
    other_three = []
    
    hop_length = 32
    
    for song_and_annotations in paths:
        printmd('**Current song: {}**'.format(song_and_annotations[0]))
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars = scr.load_or_save_bars(persisted_path, "{}\\{}.wav".format(dataset, song_number))
        annotations_frontiers_barwise = dm.frontiers_from_time_to_bar(references_segments[:,1], bars)

                
        old_spectrogram = scr.load_or_save_spectrogram(default_persisted_path, dataset + "\\" + song_and_annotations[0], "old_pcp", hop_length, fmin)
        other_spectrogram = scr.load_or_save_spectrogram(default_persisted_path, dataset + "\\" + song_and_annotations[0], "pcp", hop_length, fmin)
        
        fig, axs = plt.subplots(1, 2, figsize=(16,4))
        padded_spec_1 = pad_factor(old_spectrogram)
        padded_spec_2 = pad_factor(other_spectrogram)
        axs[0].pcolormesh(np.arange(padded_spec_1.shape[1]), np.arange(padded_spec_1.shape[0]), padded_spec_1, cmap=cmap)
        axs[0].set_title("Ancienne version du spectrogramme")
        axs[0].set_ylabel("Chromas")
        axs[1].pcolormesh(np.arange(padded_spec_2.shape[1]), np.arange(padded_spec_2.shape[0]), padded_spec_2, cmap=cmap)
        axs[1].set_title("Version de groupe du spectrogramme")
        axs[1].set_xlabel("hop_length: {}, fmin: {}, win_len_smooth: 82, norm: inf, n_octaves: 6 bins_per_chroma: 3".format(hop_length, fmin))
        plt.show()
        
        hop_length_seconds = hop_length/sampling_rate

        old_tensor_spectrogram = tf.tensorize_barwise(old_spectrogram, bars, hop_length_seconds, subdivision)
        other_tensor_spectrogram = tf.tensorize_barwise(other_spectrogram, bars, hop_length_seconds, subdivision)
        
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        padded_spec_1 = pad_factor(as_seg.get_autosimilarity(tl.unfold(old_tensor_spectrogram, 2), transpose = True, normalize = True))
        padded_spec_2 = pad_factor(as_seg.get_autosimilarity(tl.unfold(other_tensor_spectrogram, 2), transpose = True, normalize = True))
        axs[0].pcolormesh(np.arange(padded_spec_1.shape[1]), np.arange(padded_spec_1.shape[0]), padded_spec_1, cmap=cmap)
        axs[0].set_title("Autosimilarité de l'ancienne version du spectrogramme")
        axs[0].set_ylabel("Mesures")
        axs[0].invert_yaxis()

        axs[1].pcolormesh(np.arange(padded_spec_2.shape[1]), np.arange(padded_spec_2.shape[0]), padded_spec_2, cmap=cmap)
        axs[1].set_title("Autosimilarité de  de la version de groupe du spectrogramme")
        axs[1].invert_yaxis()

        plt.show()

        old_chr_q_factor = scr.NTD_decomp_as_script(persisted_path, "_{}_{}_{}_{}".format(song_number, "old_pcp", "chromas", subdivision), old_tensor_spectrogram, [12,32,32], init = "chromas")[1][2]

        old_chr_autosimilarity = as_seg.get_autosimilarity(old_chr_q_factor, transpose = True, normalize = True)
        old_segments = as_seg.dynamic_convolution_computation(old_chr_autosimilarity, mix = 1)[0]#, penalty_weight = penalty_weight)[0]
        old_segments_in_time = dm.segments_from_bar_to_time(old_segments, bars)
                
        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, old_segments_in_time, window_length = 0.5)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, old_segments_in_time, window_length = 0.5)
        old_zero_five.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])

        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, old_segments_in_time, window_length = 3)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, old_segments_in_time, window_length = 3)
        old_three.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
        
        other_chr_q_factor = scr.NTD_decomp_as_script(persisted_path, "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision), other_tensor_spectrogram, [12,32,32], init = "chromas")[1][2]

        other_chr_autosimilarity = as_seg.get_autosimilarity(other_chr_q_factor, transpose = True, normalize = True)
        other_segments = as_seg.dynamic_convolution_computation(other_chr_autosimilarity, mix = 1)[0]#, penalty_weight = penalty_weight)[0]
        other_segments_in_time = dm.segments_from_bar_to_time(other_segments, bars)
                
        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, other_segments_in_time, window_length = 0.5)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, other_segments_in_time, window_length = 0.5)
        other_zero_five.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])

        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, other_segments_in_time, window_length = 3)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, other_segments_in_time, window_length = 3)
        other_three.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
        
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        padded_spec_1 = pad_factor(as_seg.get_autosimilarity(old_chr_q_factor, transpose = True, normalize = True))
        padded_spec_2 = pad_factor(as_seg.get_autosimilarity(other_chr_q_factor, transpose = True, normalize = True))
        axs[0].pcolormesh(np.arange(padded_spec_1.shape[1]), np.arange(padded_spec_1.shape[0]), padded_spec_1, cmap=cmap)
        axs[0].set_title("Autosimilarité de Q calculé sur l'ancienne version du spectrogramme")
        axs[0].invert_yaxis()
        axs[0].set_ylabel("Mesures")
        axs[1].pcolormesh(np.arange(padded_spec_2.shape[1]), np.arange(padded_spec_2.shape[0]), padded_spec_2, cmap=cmap)
        axs[1].set_title("Autosimilarité de  Q calculé sur la version de groupe du spectrogramme")
        axs[1].invert_yaxis()
        for x in annotations_frontiers_barwise[:-1]:
            for i in range(0,2):
                axs[i].plot([x,x], [0,padded_spec_1.shape[0] - 1], '-', linewidth=1, color = "blue")
                axs[i].plot([0,padded_spec_1.shape[0] - 1], [x,x], '-', linewidth=1, color = "blue")       
        plt.show()

    results = [old_zero_five, old_three, other_zero_five, other_three]

    spectrogramme = []
    tolerance = []
    for i in ['Ancien spectrogramme', 'Nouveau spectrogramme']:
        for j in ['0.5 seconde', '3 secondes']:
            spectrogramme.append(i)
            tolerance.append(j)
    col = [np.array(spectrogramme), np.array(tolerance)]
    lines = []
    for i in range(4):
        line = []
        for j in range(6):
            line.append(np.mean(np.array(results)[i,:,j]))
        lines.append(line)
    
    dataframe_zero_five = pd.DataFrame(np.array(lines), columns = ['Vrai Positifs','Faux Positifs','Faux Négatifs','Precision', 'Rappel', 'F mesure'], index = col)
    return dataframe_zero_five

def compare_subdivisions(dataset_name, subdivisions, fmin = 98, persisted_path = default_persisted_path):
    dataset = "{}\\{}".format(root_rwc_folder, dataset_name)
    annotations_type = "MIREX10"
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)

    paths = scr.load_RWC_dataset(dataset, annotations_type)
    zero_five = []
    three = []
    
    hop_length = 32
    
    for song_and_annotations in paths:
        printmd('**Current song: {}**'.format(song_and_annotations[0]))

        line_zero_five = []
        line_three = []
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars = scr.load_or_save_bars(persisted_path, "{}\\{}.wav".format(dataset, song_number))
        annotations_frontiers_barwise = dm.frontiers_from_time_to_bar(references_segments[:,1], bars)
        spectrogram = scr.load_or_save_spectrogram(default_persisted_path, dataset + "\\" + song_and_annotations[0], "pcp", hop_length, fmin)
        
        hop_length_seconds = hop_length/sampling_rate
        
        fig, axs = plt.subplots(1, len(subdivisions), figsize=(5 * len(subdivisions),5))


        for idx, subdivision in enumerate(subdivisions):
            tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
            
            chr_q_factor = scr.NTD_decomp_as_script(persisted_path, "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision), tensor_spectrogram, [12,32,32], init = "chromas")[1][2]
    
            autosimilarity = as_seg.get_autosimilarity(chr_q_factor, transpose = True, normalize = True)
            
            padded_spec_1 = pad_factor(autosimilarity)
            axs[idx].pcolormesh(np.arange(padded_spec_1.shape[1]), np.arange(padded_spec_1.shape[0]), padded_spec_1, cmap = cm.Greys)
            axs[idx].set_title("Autosimilarity of $Q$ with subdivision {}".format(subdivision))
            axs[idx].invert_yaxis()
            
            segments = as_seg.dynamic_convolution_computation(autosimilarity, mix = 1)[0]#, penalty_weight = penalty_weight)[0]
            segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                    
            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            line_zero_five.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
            line_three.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
        
        for x in annotations_frontiers_barwise[:-1]:
            for i in range(0,len(subdivisions)):
                axs[i].plot([x,x], [0,padded_spec_1.shape[0] - 1], '-', linewidth=1, color = "blue")
                axs[i].plot([0,padded_spec_1.shape[0] - 1], [x,x], '-', linewidth=1, color = "blue")       
        plt.show()
        
        zero_five.append(line_zero_five)
        three.append(line_three)

    subdiv = []
    tolerance = []
    for i in subdivisions:
        for j in ['0.5 seconds', '3 seconds']:
            subdiv.append("Subdivision: {}".format(i))
            tolerance.append(j)
    col = [np.array(subdiv), np.array(tolerance)]
    lines = []
    for i in range(3):
        line = []
        for j in range(6):
            line.append(np.mean(np.array(zero_five)[:,i,j]))
        lines.append(line)
        line = []
        for j in range(6):
            line.append(np.mean(np.array(three)[:,i,j]))
        lines.append(line)
    
    dataframe_zero_five = pd.DataFrame(np.array(lines), columns = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], index = col)
    return dataframe_zero_five

def compute_ranks_RWC_with_lookaround(ranks_rhythm, ranks_patterns, W = "chromas", annotations_type = "MIREX10", penalty_weight = 1, subdivision = subdivision_default, persisted_path = default_persisted_path):
    dataset = rwc_folder_path
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)

    paths = scr.load_RWC_dataset(dataset, annotations_type)

    zero_five = []
    three = []
    zero_five_lookaround = []
    optimal_q_ranks = []
    
    rhythms = []
    patterns = []
    for i in ranks_patterns:
        for j in ranks_rhythm:
            patterns.append("Rank of Q:" + str(i))
            rhythms.append("Rank of H:" + str(j))
    col = [np.array(patterns), np.array(rhythms)]
    
    for song_and_annotations in paths:
        f_mes_max = 0
        best_couple = (0,0)
        #printmd('**Chanson courante: {}**'.format(song_and_annotations[0]))
        zero_five_song = []
        three_song = []
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(dataset, song_number), "pcp", hop_length)
                
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        for rank_pattern in ranks_patterns:
            for i, rank_rhythm in enumerate(ranks_rhythm):
                ranks = [12, rank_rhythm, rank_pattern]
                persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", W, subdivision)
                chr_q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = W)[1][2]
                chr_autosimilarity = as_seg.get_autosimilarity(chr_q_factor, transpose = True, normalize = True)

                segments = as_seg.dynamic_convolution_computation(chr_autosimilarity, mix = 1, penalty_weight = 1)[0]                
                segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                
                tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                zero_five_song.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
                if f_mes == f_mes_max:
                    best_couple = (rank_rhythm, rank_pattern)
                    f_mes_max = f_mes
                    best_results = (prec,rap,f_mes)

                tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
                prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
                three_song.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
        zero_five.append(zero_five_song)
        three.append(three_song)
        the_rank = best_couple[1]
        for near_q_rank in range(best_couple[1] - 2,best_couple[1] + 3):
            near_q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, [12,best_couple[0], near_q_rank], init = W)[1][2]
            near_autosimilarity = as_seg.get_autosimilarity(near_q_factor, transpose = True, normalize = True)
            segments = as_seg.dynamic_convolution_computation(near_autosimilarity, mix = 1, penalty_weight = 1)[0]                
            segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                
            p, r, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)

            if f_mes > f_mes_max:
                f_mes_max = f_mes
                best_results = (p,r,f_mes)
                the_rank = near_q_rank
        zero_five_lookaround.append(best_results)
        optimal_q_ranks.append(the_rank)
        
    zero_five_all_mean = []
    three_all_mean = []
    for ranks_couple in range(len(zero_five[0])):
        zero_five_line = []
        three_line = []
        for i in range(6):
            zero_five_line.append(np.mean(np.array(zero_five)[:,ranks_couple,i]))
            three_line.append(np.mean(np.array(three)[:,ranks_couple,i]))
        zero_five_all_mean.append(zero_five_line)
        three_all_mean.append(three_line)
            
    dataframe_zero_five = pd.DataFrame(np.array(zero_five_all_mean), columns = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], index = col)
    dataframe_zero_five.columns.name = "0.5 seconds results"
    display(dataframe_zero_five.style.bar(subset=["F measure"], color='#5fba7d'))
    
    dataframe_three = pd.DataFrame(np.array(three_all_mean), columns = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], index = col)
    dataframe_three.columns.name = "3 seconds results"
    display(dataframe_three.style.bar(subset=["F measure"], color='#5fba7d'))
    
    dataframe_best = pd.DataFrame([np.mean(np.array(zero_five_lookaround)[:,i]) for i in range(3)], index = ['Precision', 'Recall', 'F measure'], columns = ["Best results at 0.5 seconds when we look around the best in the grid for $H$ and $Q$."])
    display(dataframe_best.T)
    
    plt.hist(optimal_q_ranks, bins = range(10,50), rwidth = 0.8)
    plt.show()
    
    return zero_five, three, zero_five_lookaround, optimal_q_ranks


def convolution_parameter_on_subset(param_range, dataset = "subset_analysis", annotations_type = "MIREX10", subdivision = subdivision_default, penalty_func = "modulo4", persisted_path = default_persisted_path):
    folder = "{}\\{}".format(root_rwc_folder, dataset)
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    
    paths = scr.load_RWC_dataset(folder, annotations_type)

    ranks = [12,32,32]
    feature = "pcp"
    zero_five = []
    three = []
    for song_and_annotations in paths:
        song_name = song_and_annotations[0].replace(".wav","")
        printmd('# Current song: {}.wav'.format(song_name))
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]

        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(folder, song_name), feature, hop_length)
        annotations_frontiers_barwise = dm.frontiers_from_time_to_bar(references_segments[:,1], bars)

        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)

        q_factor = scr.NTD_decomp_as_script(persisted_path, "_{}_{}_{}_{}".format(song_name, "pcp", "chromas", subdivision), tensor_spectrogram, [12,32,32], init = "chromas")[1][2]

        autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)
        this_song = []
        for param in param_range:
            segments = as_seg.dynamic_convolution_computation(autosimilarity, mix = 1, penalty_weight = param, penalty_func = penalty_func)[0] #, fixed_ponderation = fixed_ponderation
            segments_in_time = dm.segments_from_bar_to_time(segments, bars)
            plot_spec_with_annotations_and_prediction(autosimilarity, annotations_frontiers_barwise, dm.segments_to_frontiers(segments), title = "Autosimilarity of $Q$, with lambda {}".format(param))

            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            this_song.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
            
        zero_five.append(this_song)
    all_lines = []
    for param in range(len(param_range)):
        line = []
        for i in range(6):
            line.append(np.mean(np.array(zero_five)[:,param,i]))
        all_lines.append(line)
    dataframe = pd.DataFrame(np.array(all_lines), index = param_range, columns=np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure']))
    display(dataframe.style.bar(subset=["F measure"], color='#5fba7d'))

def repartition_bar_lengths_RWCPop():
    dataset = rwc_folder_path
    annotations_folder = "{}\\MIREX10".format(annotations_folder_path)
    paths = scr.load_RWC_dataset(dataset, "MIREX10")
    persisted_path = default_persisted_path
    lengths = []
    
    for song_and_annotations in paths:
        song_number = song_and_annotations[0].replace(".wav","")
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, "MIREX10")
        bars = scr.load_or_save_bars(persisted_path, "{}\\{}.wav".format(dataset, song_number))
        barwise_annot = dm.frontiers_from_time_to_bar(np.array(annotations)[:,1], bars)
        for i in range(len(barwise_annot) - 1):
            lengths.append(barwise_annot[i+1] - barwise_annot[i])
    print("Number of segments: {}".format(len(lengths)))
    plt.hist(lengths, bins = range(1,25), density = True, cumulative = False)
    plt.xlabel("Size of the segment")
    plt.ylabel("Proportion")
    plt.title("Distribution histogram of segment' sizes in MIREX 10 annotation")
    plt.show()

   
def compute_ranks_with_param_RWC(ranks_rhythm, ranks_patterns, penalty_range, annotations_type = "MIREX10", subdivision = subdivision_default, penalty_func = "modulo8", convolution_type = "full", persisted_path = default_persisted_path):
    dataset = rwc_folder_path
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)

    paths = scr.load_RWC_dataset(dataset, annotations_type)

    zero_five = []
    three = []
    
    for song_and_annotations in paths:
        printmd('**Current song: {}**'.format(song_and_annotations[0]))
        zero_five_song = []
        three_song = []
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(dataset, song_number), "pcp", hop_length)
                
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        for rank_pattern in ranks_patterns:
            for i, rank_rhythm in enumerate(ranks_rhythm):
                ranks = [12, rank_rhythm, rank_pattern]
                persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision)
                chr_q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "chromas")[1][2]
                chr_autosimilarity = as_seg.get_autosimilarity(chr_q_factor, transpose = True, normalize = True)
                penalty_score_zero_five = []
                penalty_score_three = []
                for penalty in penalty_range:
                    segments = as_seg.dynamic_convolution_computation(chr_autosimilarity, mix = 1, penalty_weight = penalty, max_size = 24, penalty_func = penalty_func, convolution_type = convolution_type)[0]
                    segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                    
                    tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                    prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                    penalty_score_zero_five.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
                    tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
                    prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
                    penalty_score_three.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
                    
                zero_five_song.append(penalty_score_zero_five)
                three_song.append(penalty_score_three)

        zero_five.append(zero_five_song)
        three.append(three_song)
    
    rhythms = []
    patterns = []
    for i in ranks_patterns:
        for j in ranks_rhythm:
            patterns.append("Rank of Q:" + str(i))
            rhythms.append("Rank of H:" + str(j))
    
    zero_five_all_mean = []
    three_all_mean = []
    best_params = []
    for ranks_couple in range(len(zero_five[0])):
        zero_five_line = []
        three_line = []
        param_means = []
        for param in range(len(penalty_range)):
            param_means.append(np.mean(np.array(zero_five)[:,ranks_couple,param,5]))
        best_param = np.argmax(param_means)
        zero_five_line.append(penalty_range[best_param])
        for i in range(6):
            zero_five_line.append(np.mean(np.array(zero_five)[:,ranks_couple,best_param,i]))
            three_line.append(np.mean(np.array(three)[:,ranks_couple,best_param,i]))
        zero_five_all_mean.append(zero_five_line)
        three_all_mean.append(three_line)
    col = [np.array(patterns), np.array(rhythms)]
    
    dataframe_zero_five = pd.DataFrame(np.array(zero_five_all_mean), columns = ['Best penalty weight', 'True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], index = col)
    dataframe_zero_five.columns.name = "0.5 seconds results"
    display(dataframe_zero_five.style.bar(subset=["F measure"], color='#5fba7d'))
    
    dataframe_three = pd.DataFrame(np.array(three_all_mean), columns = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], index = col)
    dataframe_three.columns.name = "3 seconds results"
    display(dataframe_three.style.bar(subset=["F measure"], color='#5fba7d'))
    return zero_five, three

    
def script_loop_ranks():
    dataset = rwc_folder_path
    paths = scr.load_RWC_dataset(dataset, "MIREX10")
    persisted_path = default_persisted_path
    ranks_rhythm = [16,24,32,40]
    ranks_pattern = [16,24,32,40]
    for song_and_annotations in paths:
        print(song_and_annotations[0])
        song_number = song_and_annotations[0].replace(".wav","")
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(dataset, song_number), "pcp", hop_length)

        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, 96)
        for rank_pattern in ranks_pattern:
            for rank_rhythm in ranks_rhythm:
                ranks = [12, rank_rhythm, rank_pattern]
                persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "tucker", 96)
                scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "tucker")[1][2]
                
# if __name__ == '__main__':
#     script_loop_ranks()

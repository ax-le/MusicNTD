# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:46:36 2020

@author: amarmore
"""

### Module defining the functions used for the tests in the paper.

from IPython.display import display, Markdown

import numpy as np
import pandas as pd
pd.set_option('precision', 4)
import tensorly as tl
import warnings
import math

import musicntd.autosimilarity_segmentation as as_seg
import musicntd.data_manipulation as dm
import musicntd.tensor_factory as tf
import musicntd.scripts.overall_scripts as scr
from musicntd.model.current_plot import *
import musicntd.scripts.default_path as paths
import musicntd.model.errors as err

def final_results_fixed_conditions(dataset, feature, ranks, penalty_weight, init = "tucker", update_rule = "hals", beta = None, n_iter_max = 1000, annotations_type = "MIREX10", subdivision = 96, penalty_func = "modulo8", convolution_type = "eight_bands", legend = "in unknown conditions."):
    """
    Segmentation results when ranks and penalty_weight are fixed before computation.
    """
    annotations_folder = "{}/{}".format(paths.path_annotation_rwc, annotations_type)
    if dataset == "full":
        dataset_path = paths.path_entire_rwc
    elif dataset == "odd_songs":
        dataset_path = paths.path_odd_songs_rwc
    elif dataset == "even_songs":
        dataset_path = paths.path_even_songs_rwc
    elif dataset == "debug":
        dataset_path = paths.path_debug_rwc
    else:
        raise err.InvalidArgumentValueException(f"Dataset type not understood: {dataset}") from None

    list_songs = scr.load_RWC_dataset(dataset_path, annotations_type)
    hop_length = 32
    hop_length_seconds = 32/44100
    zero_five_results = []
    three_results = []
    deviation = []
    
    for song_and_annotations in list_songs:
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}/{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(paths.path_data_persisted_rwc, f"{dataset_path}/{song_number}", feature, hop_length)
                
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        if update_rule == "hals":
            persisted_arguments = f"_{song_number}_{feature}_{init}_{subdivision}"
        elif update_rule == "mu":
            persisted_arguments = f"mu_slow_{song_number}_beta{beta}_{feature}_{init}_{subdivision}_n_iter_max{n_iter_max}"
        else:
            raise err.InvalidArgumentValueException(f"Update rule type not understood: {update_rule}")
            
        q_factor = scr.NTD_decomp_as_script(paths.path_data_persisted_rwc, persisted_arguments, tensor_spectrogram, ranks, init = init, update_rule = update_rule, beta = beta)[1][2]
        autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)
                    
        segments = as_seg.dynamic_convolution_computation(autosimilarity, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
        segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                    
        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
        zero_five_results.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
        three_results.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
        
        r_to_e, e_to_r = dm.compute_median_deviation_of_segmentation(references_segments, segments_in_time)
        deviation.append([r_to_e, e_to_r])
    
    results_at_zero_five = np.array([np.mean(np.array(zero_five_results)[:,i]) for i in range(6)])
    dataframe_zero_five = pd.DataFrame(results_at_zero_five, index = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], columns = [f"Results of {feature} with 0.5 seconds tolerance window {legend}"])
    display(dataframe_zero_five.T)
    
    results_at_three = np.array([np.mean(np.array(three_results)[:,i]) for i in range(6)])
    dataframe_three = pd.DataFrame(results_at_three, index = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], columns = [f"Results of {feature} with 3 seconds tolerance window {legend}"])
    display(dataframe_three.T)
    
    # mean_deviation = np.array([np.mean(np.array(deviation)[:,i]) for i in range(2)])
    # dataframe_deviation = pd.DataFrame(mean_deviation, index = ['Reference to Estimation mean deviation','Estimation to Reference mean deviation'], columns = ["Mean deviation between estimations and references{}".format(legend)])
    # display(dataframe_deviation.T)
    
    return results_at_zero_five, results_at_three

def several_ranks_with_cross_validation_of_param_RWC(learning_dataset, testing_dataset, feature, ranks_frequency, ranks_rhythm, ranks_pattern, penalty_range, init = "tucker", update_rule = "hals", beta = None, n_iter_max = 1000, annotations_type = "MIREX10", penalty_func = "modulo8", convolution_type = "eight_bands"):
    """
    Segmentation results when ranks and penalty parameter are fitted by cross validation.
    Results are shown for the test dataset.
    """
    if learning_dataset == "odd_songs":
        learning_dataset_path = paths.path_odd_songs_rwc
    elif learning_dataset == "even_songs":
        learning_dataset_path = paths.path_even_songs_rwc
    elif learning_dataset == "debug":
        learning_dataset_path = paths.path_debug_rwc
    else:
        raise err.InvalidArgumentValueException(f"Dataset type not understood: {learning_dataset}") from None
        
    if testing_dataset == "odd_songs":
        testing_dataset_path = paths.path_odd_songs_rwc
    elif testing_dataset == "even_songs":
        testing_dataset_path = paths.path_even_songs_rwc
    elif testing_dataset == "debug":
        testing_dataset_path = paths.path_debug_rwc
    else:
        raise err.InvalidArgumentValueException(f"Dataset type not understood: {testing_dataset_path}") from None
        
    if learning_dataset == testing_dataset:
        warnings.warn("Careful: using the same dataset as test and learn, normal?")
        
    annotations_folder = "{}/{}".format(paths.path_annotation_rwc, annotations_type)
    hop_length = 32
    hop_length_seconds = 32/44100
    subdivision = 96
    
    learning_dataset_songs = scr.load_RWC_dataset(learning_dataset_path, annotations_type)

    zero_five = -math.inf * np.ones((len(learning_dataset_songs), len(ranks_frequency), len(ranks_rhythm), len(ranks_pattern), len(penalty_range), 1))
    three = -math.inf * np.ones((len(learning_dataset_songs), len(ranks_frequency), len(ranks_rhythm), len(ranks_pattern), len(penalty_range), 1))
    
    for song_idx, song_and_annotations in enumerate(learning_dataset_songs):
        #printmd('**Chanson courante: {}**'.format(song_and_annotations[0]))
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}/{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(paths.path_data_persisted_rwc, "{}/{}".format(learning_dataset_path, song_number), feature, hop_length)
                
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        
        for w, rank_W in enumerate(ranks_frequency):
            for h, rank_h in enumerate(ranks_rhythm):
                for q, rank_q in enumerate(ranks_pattern):
                    ranks = [rank_W, rank_h, rank_q]  
                    if update_rule == "hals":
                        persisted_arguments = f"_{song_number}_{feature}_{init}_{subdivision}"
                    elif update_rule == "mu":
                        persisted_arguments = f"mu_slow_{song_number}_beta{beta}_{feature}_{init}_{subdivision}_n_iter_max{n_iter_max}"
                    else:
                        raise err.InvalidArgumentValueException(f"Update rule type not understood: {update_rule}")
                        
                    q_factor = scr.NTD_decomp_as_script(paths.path_data_persisted_rwc, persisted_arguments, tensor_spectrogram, ranks, init = init, update_rule = update_rule, beta = beta)[1][2]

                    autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)

                    for p, penalty in enumerate(penalty_range):
                        segments = as_seg.dynamic_convolution_computation(autosimilarity, penalty_weight = penalty, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
                        segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                        
                        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                        zero_five[song_idx, w, h, q, p] = round(f_mes,4)
        
                        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
                        three[song_idx, w, h, q, p] = round(f_mes,4)

    best_mean = 0
    best_params = []
    for w, rank_W in enumerate(ranks_frequency):
        for h, rank_h in enumerate(ranks_rhythm):
            for q, rank_q in enumerate(ranks_pattern):
                for p, penalty in enumerate(penalty_range):     
                    this_avg = np.mean(zero_five[:, w, h, q, p])
                    if this_avg > best_mean:
                        best_mean = this_avg
                        best_params = [rank_W, rank_h, rank_q, penalty]
                        
    display(pd.DataFrame(np.array([best_params[0],best_params[1],best_params[2], best_params[3]]), index = ['Best rank for $W$', 'Best rank for $H$','Best rank for $Q$','Best lambda: ponderation parameter.'], columns = ["Learned parameters"]).T)
    
    learned_ranks = [best_params[0],best_params[1],best_params[2]]
    results_at_zero_five, results_at_three = final_results_fixed_conditions(testing_dataset, feature, learned_ranks, best_params[3], init = init, update_rule = update_rule, beta = beta, n_iter_max = n_iter_max, annotations_type = annotations_type, penalty_func = penalty_func, legend = "on test dataset.", convolution_type = convolution_type)
    
    return best_params, results_at_zero_five, results_at_three


def fixed_conditions_feature(dataset, feature, penalty_weight, annotations_type = "MIREX10", subdivision = 96, penalty_func = "modulo8", legend = ", on test dataset.", convolution_type = "eight_bands"):
    """
    Segmentation results when segmenting directly the signal, where the penalty_weight is fixed before computation.
    """
    if dataset == "full":
        dataset_path = paths.path_entire_rwc
    elif dataset == "odd_songs":
        dataset_path = paths.path_odd_songs_rwc
    elif dataset == "even_songs":
        dataset_path = paths.path_even_songs_rwc
    elif dataset == "debug":
        dataset_path = paths.path_debug_rwc
    else:
        raise err.InvalidArgumentValueException(f"Dataset type not understood: {dataset}") from None

    list_songs = scr.load_RWC_dataset(dataset_path, annotations_type)
    annotations_folder = "{}/{}".format(paths.path_annotation_rwc, annotations_type)
    hop_length = 32
    hop_length_seconds = 32/44100
    zero_five = []
    three = []
    
    for song_and_annotations in list_songs:
        song_name = song_and_annotations[0].replace(".wav","")
        annot_path = "{}/{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(paths.path_data_persisted_rwc, "{}/{}".format(dataset_path, song_name), feature, hop_length)
        
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        
        unfolded = tl.unfold(tensor_spectrogram, 2)

        autosimilarity = as_seg.get_autosimilarity(unfolded, transpose = True, normalize = True)
        
        segments = as_seg.dynamic_convolution_computation(autosimilarity, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]
        segments_in_time = dm.segments_from_bar_to_time(segments, bars)
        
        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
        zero_five.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
        
        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
        three.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
            
    final_res_sig_zero_five = np.array([np.mean(np.array(zero_five)[:,i]) for i in range(6)])
    dataframe = pd.DataFrame(final_res_sig_zero_five, columns = ["Results with 0.5 seconds tolerance window{}".format(legend)], index=np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure']))
    display(dataframe.T)
    
    final_res_sig_three = np.array([np.mean(np.array(three)[:,i]) for i in range(6)])
    dataframe = pd.DataFrame(final_res_sig_three, columns = ["Results with 3 seconds tolerance window{}".format(legend)], index=np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure']))
    display(dataframe.T)
    
    return final_res_sig_zero_five, final_res_sig_three


# %% Old code, which can still be useful
def oracle_ranks(dataset, ranks_rhythm, ranks_pattern, penalty_weight, annotations_type = "MIREX10", subdivision = 96, penalty_func = "modulo8", convolution_type = "eight_bands"):
    """
    Segmentation results in the oracle condition, 
    where the rank resulting in the best f measure is chosen (among a range) for each song.
    Results are computed with a fixed penalty weight.
    """
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    hop_length = 32
    hop_length_seconds = 32/44100
    
    paths = scr.load_RWC_dataset(dataset, annotations_type)

    zero_five = []
    three = []
    
    for song_and_annotations in paths:
        #printmd('**Chanson courante: {}**'.format(song_and_annotations[0]))
        zero_five_rank = []
        three_rank = []
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(dataset, song_number), "pcp", hop_length)
                
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        
        for rank_pattern in ranks_pattern:
            for i, rank_rhythm in enumerate(ranks_rhythm):
                ranks = [12, rank_rhythm, rank_pattern]
                persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision)
                q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "chromas")[1][2]
                autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)

                segments = as_seg.dynamic_convolution_computation(autosimilarity, mix = 1, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
                segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                
                tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                zero_five_rank.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])

                tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
                prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
                three_rank.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])

        zero_five.append(zero_five_rank)
        three.append(three_rank)
    
    zero_five_array = np.array(zero_five)
    three_array = np.array(three)

    scores_oracle_NTD_zero_five = []
    scores_oracle_NTD_three = []
    
    for song in range(len(zero_five)):
        all_f_zero_five = zero_five_array[song,:,5]
        this_song_best_couple_idx_zero_five = np.argmax(all_f_zero_five)
        scores_oracle_NTD_zero_five.append(zero_five_array[song, this_song_best_couple_idx_zero_five,:])
        
        all_f_three = three_array[song,:,5]
        this_song_best_couple_idx_three = np.argmax(all_f_three)
        scores_oracle_NTD_three.append(three_array[song,this_song_best_couple_idx_three,:])
        
    final_scores_zero_five = [np.mean(np.array(scores_oracle_NTD_zero_five)[:,i]) for i in range(6)]
    best_dataframe_zero_five = pd.DataFrame(np.array(final_scores_zero_five), index = [np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'])], columns = ["Oracles ranks, at 0.5 seconds"])
    display(best_dataframe_zero_five.T)
    
    final_scores_three = [np.mean(np.array(scores_oracle_NTD_three)[:,i]) for i in range(6)]

    best_dataframe_three = pd.DataFrame(np.array(final_scores_three), index = [np.array(['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'])], columns = ["Oracles ranks, at 3 seconds"])
    display(best_dataframe_three.T)
    
    return zero_five_array

def cross_validation_on_signal(learning_dataset, testing_dataset, penalty_range, annotations_type = "MIREX10", subdivision = 96, penalty_func = "modulo8", convolution_type = "eight_bands"):
    """
    Segmentation results when segmenting directly the signal, where the penalty_weight is fitted by cross-validation.
    """
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    hop_length = 32
    hop_length_seconds = 32/44100
    
    learning_dataset_paths = scr.load_RWC_dataset(learning_dataset, annotations_type)

    zero_five = []
    three = []
    
    for song_and_annotations in learning_dataset_paths:
        #printmd('**Chanson courante: {}**'.format(song_and_annotations[0]))
        this_song = []
        this_song_three = []
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(learning_dataset, song_number), "pcp", hop_length)
                
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        
        unfolded = tl.unfold(tensor_spectrogram, 2)
        
        autosimilarity = as_seg.get_autosimilarity(unfolded, transpose = True, normalize = True)

        for param in penalty_range:
            segments = as_seg.dynamic_convolution_computation(autosimilarity, mix = 1, penalty_weight = param, penalty_func = penalty_func, convolution_type = convolution_type)[0] #, fixed_ponderation = fixed_ponderation
            segments_in_time = dm.segments_from_bar_to_time(segments, bars)
            
            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
            this_song.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
            
            tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
            prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
            this_song_three.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
            
        zero_five.append(this_song)
        three.append(this_song_three)
    
    fmes_param_means = []
    for idx_param in range(len(penalty_range)):
        fmes_param_means.append(np.mean(np.array(zero_five)[:,idx_param,5]))
    idx_current_best_param = np.argmax(fmes_param_means)
    best_param = penalty_range[idx_current_best_param]
    
    display(pd.DataFrame(np.array([best_param]), index = ['Best lambda: ponderation parameter.'], columns = ["Learned parameter."]).T)
      
    final_res_sig_zero_five, final_res_sig_three = fixed_conditions_signal(testing_dataset, best_param, annotations_type = annotations_type, subdivision = subdivision, penalty_func = penalty_func, legend = ", on test dataset.", convolution_type = convolution_type)
    
    return final_res_sig_zero_five, final_res_sig_three

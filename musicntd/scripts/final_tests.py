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

import musicntd.autosimilarity_segmentation as as_seg
import musicntd.data_manipulation as dm
import musicntd.tensor_factory as tf
import musicntd.scripts.overall_scripts as scr
from musicntd.model.current_plot import *

annotations_folder_path = "C:\\Users\\amarmore\\Desktop\\Audio samples\\RWC Pop\\annotations"
persisted_path = "C:\\Users\\amarmore\\Desktop\\data_persisted"

def final_results_fixed_conditions(dataset, ranks, penalty_weight, annotations_type = "MIREX10", subdivision = 96, penalty_func = "modulo8", legend = " en conditions non précisées.", convolution_type = "eight_bands"):
    """
    Segmentation results when ranks and penalty_weight are fixed before computation.
    """
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    paths = scr.load_RWC_dataset(dataset, annotations_type)
    hop_length = 32
    hop_length_seconds = 32/44100
    zero_five_results = []
    three_results = []
    
    for song_and_annotations in paths:
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(dataset, song_number), "pcp", hop_length)
                
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision)
        q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "chromas")[1][2]
        autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)
                    
        segments = as_seg.dynamic_convolution_computation(autosimilarity, mix = 1, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
        segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                    
        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
        zero_five_results.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
        three_results.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
    
    results_at_zero_five = np.array([np.mean(np.array(zero_five_results)[:,i]) for i in range(6)])
    dataframe_zero_five = pd.DataFrame(results_at_zero_five, index = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], columns = ["Results with 0.5 seconds tolerance window{}".format(legend)])
    display(dataframe_zero_five.T)
    
    results_at_three = np.array([np.mean(np.array(three_results)[:,i]) for i in range(6)])
    dataframe_three = pd.DataFrame(results_at_three, index = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], columns = ["Results with 3 seconds tolerance window{}".format(legend)])
    display(dataframe_three.T)
    
    return results_at_zero_five, results_at_three

def several_ranks_with_cross_validation_of_param_RWC(learning_dataset, testing_dataset, ranks_rhythm, ranks_pattern, penalty_range, annotations_type = "MIREX10", subdivision = 96, penalty_func = "modulo8", convolution_type = "eight_bands"):
    """
    Segmentation results when ranks and penalty parameter are fitted by cross validation.
    Results are shown for the test dataset.
    """
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    hop_length = 32
    hop_length_seconds = 32/44100
    
    learning_dataset_paths = scr.load_RWC_dataset(learning_dataset, annotations_type)

    zero_five = []
    three = []
    
    for song_and_annotations in learning_dataset_paths:
        #printmd('**Chanson courante: {}**'.format(song_and_annotations[0]))
        zero_five_rank = []
        three_rank = []
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(learning_dataset, song_number), "pcp", hop_length)
                
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        
        for rank_pattern in ranks_pattern:
            for i, rank_rhythm in enumerate(ranks_rhythm):
                ranks = [12, rank_rhythm, rank_pattern]
                persisted_arguments = "_{}_{}_{}_{}".format(song_number, "pcp", "chromas", subdivision)
                q_factor = scr.NTD_decomp_as_script(persisted_path, persisted_arguments, tensor_spectrogram, ranks, init = "chromas")[1][2]
                autosimilarity = as_seg.get_autosimilarity(q_factor, transpose = True, normalize = True)
                penalty_score_zero_five = []
                penalty_score_three = []
                for penalty in penalty_range:
                    segments = as_seg.dynamic_convolution_computation(autosimilarity, mix = 1, penalty_weight = penalty, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
                    segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                    
                    tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                    prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                    penalty_score_zero_five.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
                    tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
                    prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
                    penalty_score_three.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
                    
                zero_five_rank.append(penalty_score_zero_five)
                three_rank.append(penalty_score_three)

        zero_five.append(zero_five_rank)
        three.append(three_rank)
    
    learned_params = []
    all_means_ranks = []
    for idx_rank_couple in range(len(zero_five[0])):
        fmes_param_means = []
        for idx_param in range(len(penalty_range)):
            fmes_param_means.append(np.mean(np.array(zero_five)[:,idx_rank_couple,idx_param,5]))
        idx_current_best_param = np.argmax(fmes_param_means)
        learned_params.append(idx_current_best_param)
        all_means_ranks.append(np.mean(np.array(zero_five)[:,idx_rank_couple,idx_current_best_param,5]))
    idx_best_couple = np.argmax(all_means_ranks)
    best_param = penalty_range[learned_params[idx_best_couple]]
    best_couple = (ranks_rhythm[idx_best_couple%len(ranks_rhythm)], ranks_pattern[idx_best_couple//len(ranks_rhythm)])
    
    display(pd.DataFrame(np.array([best_couple[0], best_couple[1], best_param]), index = ['Best rank for $H$','Best rank for $Q$','Best lambda: ponderation parameter.'], columns = ["Learned parameters"]).T)
    
    learned_ranks = [12, best_couple[0], best_couple[1]]
    results_at_zero_five, results_at_three = final_results_fixed_conditions(testing_dataset, learned_ranks, best_param, annotations_type = annotations_type, subdivision = subdivision, penalty_func = penalty_func, legend = ", on test dataset.", convolution_type = convolution_type)
    
    return best_param, results_at_zero_five, results_at_three


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
    
      
def fixed_conditions_signal(dataset, penalty_param, annotations_type = "MIREX10", subdivision = 96, penalty_func = "modulo8", legend = ", on test dataset.", convolution_type = "eight_bands"):
    """
    Segmentation results when segmenting directly the signal, where the penalty_weight is fixed before computation.
    """
    paths = scr.load_RWC_dataset(dataset, annotations_type)
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    hop_length = 32
    hop_length_seconds = 32/44100
    zero_five = []
    three = []
    
    for song_and_annotations in paths:
        song_name = song_and_annotations[0].replace(".wav","")
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(dataset, song_name), "pcp", hop_length)
        
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        
        unfolded = tl.unfold(tensor_spectrogram, 2)

        autosimilarity = as_seg.get_autosimilarity(unfolded, transpose = True, normalize = True)
        
        segments = as_seg.dynamic_convolution_computation(autosimilarity, mix = 1, penalty_weight = penalty_param, penalty_func = penalty_func, convolution_type = convolution_type)[0] #, fixed_ponderation = fixed_ponderation
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


def results_on_signal_without_lambda(dataset, annotations_type = "MIREX10", subdivision = 96, legend = " en conditions non précisées.", convolution_type = "eight_bands"):
    """
    Segmentation results on the autosimilarity of the signal without penalty function.
    """
    annotations_folder = "{}\\{}".format(annotations_folder_path, annotations_type)
    paths = scr.load_RWC_dataset(dataset, annotations_type)
    hop_length = 32
    hop_length_seconds = 32/44100
    zero_five_results = []
    three_results = []
    
    for song_and_annotations in paths:
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}\\{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        bars, spectrogram = scr.load_or_save_spectrogram_and_bars(persisted_path, "{}\\{}".format(dataset, song_number), "pcp", hop_length)
                
        tensor_spectrogram = tf.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        
        unfolded = tl.unfold(tensor_spectrogram, 2)
        
        autosimilarity = as_seg.get_autosimilarity(unfolded, transpose = True, normalize = True)
        
        segments = as_seg.dynamic_convolution_computation(autosimilarity, mix = 1, penalty_weight = 0)[0]                
        segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                    
        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
        zero_five_results.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
        tp,fp,fn = dm.compute_rates_of_segmentation(references_segments, segments_in_time, window_length = 3)
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
        three_results.append([tp,fp,fn,round(prec,4),round(rap,4),round(f_mes,4)])
    
    dataframe_zero_five = pd.DataFrame(np.array([np.mean(np.array(zero_five_results)[:,i]) for i in range(6)]), index = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], columns = ["0.5 secondes results {}".format(legend)])
    display(dataframe_zero_five.T)
    
    dataframe_three = pd.DataFrame(np.array([np.mean(np.array(three_results)[:,i]) for i in range(6)]), index = ['True Positives','False Positives','False Negatives','Precision', 'Recall', 'F measure'], columns = ["3 secondes results {}".format(legend)])
    display(dataframe_three.T)

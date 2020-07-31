# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:29:17 2019

@author: amarmore
"""

# Defining current plotting functions.

import context

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_me_this_spectrogram(spec, title = "Spectrogram", x_axis = "x_axis", y_axis = "y_axis", invert_y_axis = True, cmap = cm.Greys, figsize = None):
    """
    Plots a spectrogram in a colormesh.
    """
    if figsize != None:
        plt.figure(figsize=figsize)
    elif spec.shape[0] == spec.shape[1]:
        plt.figure(figsize=(7,7))
    padded_spec = pad_factor(spec)
    plt.pcolormesh(np.arange(padded_spec.shape[1]), np.arange(padded_spec.shape[0]), padded_spec, cmap=cmap)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if invert_y_axis:
        plt.gca().invert_yaxis()
    plt.show()
    
def pad_factor(factor):
    """
    Pads the factor with zeroes on both dimension.
    This is made because colormesh plots values as intervals (post and intervals problem),
    and so discards the last value.
    """
    padded = np.zeros((factor.shape[0] + 1, factor.shape[1] + 1))
    for i in range(factor.shape[0]):
        for j in range(factor.shape[1]):
            padded[i,j] = factor[i,j]
    return np.array(padded)

def plot_me_this_tucker(factors, core, cmap = cm.Greys):
    """
    Plot all factors, and each slice of the core (as musical pattern) from the NTD.
    """
    plot_me_this_spectrogram(factors[0], title = "Midi factor", x_axis = "Atoms", y_axis = "Midi value", invert_y_axis = False, cmap = cmap)
    plot_me_this_spectrogram(factors[1].T, title = "Rythmic patterns factor", x_axis = "Position in bar", y_axis = "Atoms", cmap = cmap)
    plot_me_this_spectrogram(factors[2].T, title = "Structural patterns factor", x_axis = "Bar index", y_axis = "Atoms", cmap = cmap)
    print("Core:")
    for i in range(len(core[0,0,:])):
        plot_me_this_spectrogram(core[:,:,i], title = "Core, slice " + str(i), x_axis = "Time atoms", y_axis = "Freq Atoms", cmap = cmap)

def permutate_factor(factor):
    """
    Computes the permutation of columns of the factors for them to be visually more comprehensible.
    """
    permutations = []
    for i in factor:
        idx_max = np.argmax(i)
        if idx_max not in permutations:
            permutations.append(idx_max)
    for i in range(factor.shape[1]):
        if i not in permutations:
            permutations.append(i)
    return permutations

def plot_permuted_factor(factor, title = None,x_axis = None, y_axis = None, cmap = cm.Greys):
    """
    Plots this factor, but permuted to be easier to understand visually.
    """
    permut = permutate_factor(factor)
    plot_me_this_spectrogram(factor.T[permut], title = title,x_axis = x_axis, y_axis = y_axis,
                             figsize=(factor.shape[0]/10,factor.shape[1]/10), cmap = cmap)

def plot_permuted_tucker(factors, core, cmap = cm.Greys, plot_core = True):
    """
    Plots every factor and slice of the core from the NTD, but permuted to be easier to understand visually.
    """
    plot_me_this_spectrogram(factors[0], title = "W matrix (muscial content)",
                         x_axis = "Atoms", y_axis = "Pitch-class Index", cmap = cmap)
    h_permut = permutate_factor(factors[1])
    plot_me_this_spectrogram(factors[1].T[h_permut], title = "H matrix: time at barscale (rythmic content)",
                             x_axis = "Position in the bar (in frame indexes)", y_axis = "Atoms\n(permuted for visualization purpose)", 
                             figsize=(factors[1].shape[0]/4,factors[1].shape[1]/4), cmap = cmap)
    q_permut = permutate_factor(factors[2])
    plot_me_this_spectrogram(factors[2].T[q_permut], title = "Q matrix: Bar content feature",
                             x_axis = "Index of the bar", y_axis = "Musical pattern index\n(permuted for visualization purpose)", 
                             figsize=(factors[2].shape[0]/4,factors[2].shape[1]/4), cmap = cmap)
    if plot_core:
        for i, idx in enumerate(q_permut):
            plot_me_this_spectrogram(core[:,h_permut,idx], title = "Core, slice {} (slice {} in original decomposition order)".format(i, idx), x_axis = "Time atoms", y_axis = "Freq Atoms", cmap = cm.Greys)

def compare_both_c(spec_1, spec_2, title_1 = "First", title_2 = "Second"):
    """
    Compare two Q matrices.
    Given two different Q, it will plot Q^T and Q.Q^T (autosimilairty of Q^T).
    """
    fig, axs = plt.subplots(2, 2, figsize=(15,15))
    axs[0, 0].pcolormesh(np.arange(spec_1.shape[0]), np.arange(spec_1.shape[1]), spec_1.T, cmap=cm.gray)
    axs[0, 0].set_title(title_1 +  ' spectrogram')
    axs[0, 1].pcolormesh(np.arange(spec_2.shape[0]), np.arange(spec_2.shape[1]), spec_2.T, cmap=cm.gray)
    axs[0, 1].set_title(title_2 + ' spectrogram')
    axs[1, 0].pcolormesh(np.arange(spec_1.shape[0]), np.arange(spec_1.shape[0]), spec_1@spec_1.T, cmap=cm.gray)
    axs[1, 0].set_title('Autosimilarity of ' + title_1)
    axs[1, 1].pcolormesh(np.arange(spec_2.shape[0]), np.arange(spec_2.shape[0]), spec_2@spec_2.T, cmap=cm.gray)
    axs[1, 1].set_title('Autosimilarity of ' + title_2)
    axs[0, 0].invert_yaxis()
    axs[1, 0].invert_yaxis()
    axs[1, 1].invert_yaxis()
    axs[0, 1].invert_yaxis()
    plt.show()
    
def compare_both_annotated(spec_1, spec_2, ends_1, ends_2, annotations, title_1 = "First", title_2 = "Second"):
    """
    Compare two Q matrices, with the annotation of the segmentation.
    Given two different Q, it will plot Q^T and Q.Q^T (autosimilairty of Q^T).
    """
    fig, axs = plt.subplots(1, 2, figsize=(20,10))
    padded_spec_1 = pad_factor(spec_1)
    axs[0].pcolormesh(np.arange(padded_spec_1.shape[0]), np.arange(padded_spec_1.shape[0]), padded_spec_1)
    axs[0].set_title('Autosimilarity of ' + title_1)
    padded_spec_2 = pad_factor(spec_2)
    axs[1].pcolormesh(np.arange(padded_spec_2.shape[0]), np.arange(padded_spec_2.shape[0]), padded_spec_2)
    axs[1].set_title('Autosimilarity of ' + title_2)
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()    
    for x in annotations:
        axs[0].plot([x,x], [0,len(spec_1) - 1], '-', linewidth=1, color = "red")
        axs[1].plot([x,x], [0,len(spec_2) - 1], '-', linewidth=1, color = "red")
    for x in ends_1:
        if x in annotations:
            axs[0].plot([x,x], [0,len(spec_1)], '-', linewidth=1, color = "#8080FF")#"#17becf")
        else:
            axs[0].plot([x,x], [0,len(spec_1)], '-', linewidth=1, color = "orange")
    for x in ends_2:
        if x in annotations:
            axs[1].plot([x,x], [0,len(spec_2)], '-', linewidth=1, color = "#8080FF")#"#17becf")
        else:
            axs[1].plot([x,x], [0,len(spec_2)], '-', linewidth=1, color = "orange")
    plt.show()
    
def plot_measure_with_annotations(measure, annotations, color = "red"):
    """
    Plots the measure (typically novelty) with the segmentation annotation.
    """
    plt.plot(np.arange(len(measure)),measure, color = "black")
    for x in annotations:
        plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = color)
    plt.show()
    
def plot_measure_with_annotations_and_prediction(measure, annotations, frontiers_predictions, title = "Title"):
    """
    Plots the measure (typically novelty) with the segmentation annotation and the estimated segmentation.
    """
    plt.title(title)
    plt.plot(np.arange(len(measure)),measure, color = "black")
    ax1 = plt.axes()
    ax1.axes.get_yaxis().set_visible(False)
    for x in annotations:
        plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "red")
    for x in frontiers_predictions:
        if x in annotations:
            plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "#8080FF")#"#17becf")
        else:
            plt.plot([x, x], [0,np.amax(measure)], '-', linewidth=1, color = "orange")
    plt.show()

def plot_spec_with_annotations(factor, annotations, color = "yellow", title = None):
    """
    Plots a spectrogram with the segmentation annotation.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(9,9))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cm.Greys)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = color)
    plt.show()
    
def plot_spec_with_annotations_abs_ord(factor, annotations, color = "green", title = None, cmap = cm.gray):
    """
    Plots a spectrogram with the segmentation annotation in both x and y axes.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cmap)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = color)
        plt.plot([0,len(factor)], [x,x], '-', linewidth=1, color = color)
    plt.show()

def plot_spec_with_annotations_and_prediction(factor, annotations, predicted_ends, title = "Title"):
    """
    Plots a spectrogram with the segmentation annotation and the estimated segmentation.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    else:
        plt.figure(figsize=(factor.shape[1]/4,factor.shape[0]/4))

    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cm.Greys)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "#8080FF")
    for x in predicted_ends:
        if x in annotations:
            plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "green")#"#17becf")
        else:
            plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "orange")
    plt.show()
    
def plot_segments_with_annotations(seg, annot):
    """
    Plots the estimated labelling of segments next to with the frontiers in the annotation.
    """
    for x in seg:
        plt.plot([x[0], x[1]], [x[2]/10,x[2]/10], '-', linewidth=1, color = "black")
    for x in annot:
        plt.plot([x[1], x[1]], [0,np.amax(np.array(seg)[:,2])/10], '-', linewidth=1, color = "red")
    plt.show()

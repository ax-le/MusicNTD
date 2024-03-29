# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:01:42 2020

@author: amarmore
"""

# Everything related to the segmentation of the autosimilarity.

import numpy as np
import math
from scipy.sparse import diags
import musicntd.model.errors as err
import warnings

def get_autosimilarity(an_array, transpose = False, normalize = False):
    """
    Encapsulates the autosimilarity generation of a matrix.

    Parameters
    ----------
    an_array : numpy array
        The array/matrix seen as array which autosimilarity os to compute.
    transpose : boolean, optional
        Whether the array has to be transpose for computing the autosimilarity.
        The default is False.
    normalize : boolean, optional
        Whether to normalize the autosimilarity.
        The default is False.

    Returns
    -------
    numpy array
        The autosimilarity of this array.

    """
    if type(an_array) is list:
        this_array = np.array(an_array)
    else:
        this_array = an_array
    if transpose:
        this_array = this_array.T
    if normalize:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in true_divide") # Avoiding to show the warning, as it's handled, not te confuse the user.
            this_array = np.array([list(i/np.linalg.norm(i)) for i in this_array.T]).T
            this_array = np.where(np.isnan(this_array), 1e-10, this_array) # Replace null lines, avoiding best-path retrieval to fail
    return this_array.T@this_array

def compute_all_kernels(max_size, convolution_type = "full"):
    """
    Precomputes all kernels of size 0 ([0]) to max_size, and feed them to the Dynamic Progamming algorithm.

    Parameters
    ----------
    max_size : integer
        The maximal size (included) for kernels.
    convolution_type: string
        The type of convolution. (to explicit)
        Possibilities are :
            - "full" : squared matrix entirely composed of one, except on the diagonal where it's zero.
            The associated convolution cost for a segment (b_1, b_2) will be
            .. math::
                c_{b_1,b_2} = \\frac{1}{b_2 - b_1 + 1}\\sum_{i,j = 0, i \\ne j}^{n - 1}  a_{i + b_1, j + b_1}
            - "eight_bands" : squared matrix where the only nonzero values are ones on the 
            8 subdiagonals surrounding the main diagonal.
            The associated convolution cost for a segment (b_1, b_2) will be
            .. math::
                c_{b_1,b_2} = \\frac{1}{b_2 - b_1 + 1}\\sum_{i,j = 0, 1 \\leq |i - j| \\leq 4}^{n - 1}  a_{i + b_1, j + b_1}
            - "mixed" : sum of both previous kernels, i.e. values are zero on the diagonal,
            2 on the 8 subdiagonals surrounding the main diagonal, and 1 elsewhere.
            The associated convolution cost for a segment (b_1, b_2) will be
            .. math::
                c_{b_1,b_2} = \\frac{1}{b_2 - b_1 + 1}(2*\\sum_{i,j = 0, 1 \\leq |i - j| \\leq 4}^{n - 1}  a_{i + b_1, j + b_1} \\ + \sum_{i,j = 0, |i - j| > 4}^{n - 1}  a_{i + b_1, j + b_1})
        
    Returns
    -------
    kernels : array of arrays (which are kernels)
        All the kernels, of size 0 ([0]) to max_size.

    """
    kernels = [[0]]
    for p in range(1,max_size + 1):
        if p < 4:
            kern = np.ones((p,p)) - np.identity(p)
        else:
            if convolution_type == "full":
                # Full kernel (except for the diagonal)
                kern = np.ones((p,p)) - np.identity(p)
            elif convolution_type == "eight_bands":
                # Diagonal where only the eight subdiagonals surrounding the main diagonal is one
                k = np.array([np.ones(p-4),np.ones(p-3),np.ones(p-2),np.ones(p-1),np.zeros(p),np.ones(p-1),np.ones(p-2),np.ones(p-3),np.ones(p-4)])
                offset = [-4,-3,-2,-1,0,1,2,3,4]
                kern = diags(k,offset).toarray()
            elif convolution_type == "mixed":
                # Sum of both previous kernels
                k = np.array([np.ones(p-4),np.ones(p-3),np.ones(p-2),np.ones(p-1),np.zeros(p),np.ones(p-1),np.ones(p-2),np.ones(p-3),np.ones(p-4)])
                offset = [-4,-3,-2,-1,0,1,2,3,4]
                kern = np.ones((p,p)) - np.identity(p) + diags(k,offset).toarray()
            else:
                raise err.InvalidArgumentValueException("Convolution type not understood.")
        kernels.append(kern)
    return kernels

def convolutionnal_cost(cropped_autosimilarity, kernels):
    """
    The convolution measure on this part of the autosimilarity matrix.

    Parameters
    ----------
    cropped_autosimilarity : list of list of floats or numpy array (matrix representation)
        The part of the autosimilarity which convolution measure is to compute.
    kernels : list of arrays
        Acceptable kernels.

    Returns
    -------
    float
        The convolution measure.

    """
    p = len(cropped_autosimilarity)
    kern = kernels[p]
    #return np.mean(np.multiply(kern,cropped_autosimilarity))
    return np.sum(np.multiply(kern,cropped_autosimilarity)) / p**2

def convolution_entire_matrix_computation(autosimilarity_array, kernels, kernel_size = 8):
    """
    Computes the convolution measure on the entire autosimilarity matrix, with a defined and fixed kernel size.

    Parameters
    ----------
    autosimilarity_array : list of list of floats or numpy array (matrix representation)
        The autosimilarity matrix.
    kernels : list of arrays
        All acceptable kernels.
    kernel_size : integer
        The size of the kernel for this measure.

    Returns
    -------
    cost : list of float
        List of convolution measures, at each bar of the autosimilarity.

    """
    cost = np.zeros(len(autosimilarity_array))
    for i in range(kernel_size, len(autosimilarity_array)):
        cost[i] = convolutionnal_cost(autosimilarity_array[i - kernel_size:i,i - kernel_size:i], kernels)
    return cost

def dynamic_convolution_computation(autosimilarity, mix = 1, min_size = 1, max_size = 36, novelty_kernel_size = 16, penalty_weight = 1, penalty_func = "modulo4", convolution_type = "eight_bands"):
    """
    Dynamic programming algorithm, computing a maximization of a cost, sum of segments' costs on the autosimilarity.
    This cost is a combination of
     - the convolutionnal cost on the segment, with a dynamic size, 
     - a penalty cost, function of the size of the segment, to enforce specific sizes (with prior knowledge),
     - the novelty cost applied on the end of the segment, with a fixed kernel size.
        EDIT: Not supported anymore (left for potential comparison tests).
     
    The penalty cost is computed in the function "penalty_cost_from_arg()".
    See this function for further details.
     
    This trade-off is handled by the <mix> parameter, with:
        cost = mix * convolutionnal cost + (1 - mix) * novelty cost
    EDIT: this behavior is not supported anymore, but could be in the future.
     
    It returns the optimal segmentation according to this cost.

    Parameters
    ----------
    autosimilarity : list of list of float (list of columns)
        The autosimilarity to segment.
    mix : float \in (0,1], optional
        The trade-off parameter between convolutionnal cost and novelty cost.
        It shouldn't be set to zero as it correspond to the basic novelty cost.
        The default is 0.5.
    min_size : integer, optional
        The minimal length of segments.
        The default is 1.
    max_size : integer, optional
        The maximal length of segments.
        The default is 36.
    novelty_kernel_size : integer, optional
        The size of the novelty_kernel.
        The default is 12.
        EDIT: Not supported anymore (left for potential comparison tests)
    penalty_weight : float, optional
        The ponderation parameter for the penalty function
    penalty_func : string
        The type of penalty function to use.
        See "penalty_cost_from_arg()" for further details.
    convolution_type : string
        The type of convolution we want to use in this computation.
        See "compute_all_kernels()" for a detailed list of possibilities.

    Raises
    ------
    InvalidArgumentValueException and ToDebugException
        Errors.

    Returns
    -------
    list of tuples
        The segments, as a list of tuples (start, end).
    integer
        Global cost (the minimal among all).

    """
    if novelty_kernel_size % 2 == 1:
        raise err.InvalidArgumentValueException("The novelty kernel should be even.") from None
    if mix < 0 or mix > 1:
        raise err.InvalidArgumentValueException("Mix is a weight, between 0 and 1, to mitigate between convolutionnal and novelty cost.") from None
    if mix == 0:
        raise err.InvalidArgumentValueException("As novelty cost use a fixed kernel, a 0 cost, neutralizing the convolutionnal cost, shouldn't be used.") from None

    costs = [-math.inf for i in range(len(autosimilarity))]
    segments_best_starts = [None for i in range(len(autosimilarity))]
    segments_best_starts[0] = 0
    costs[0] = 0
    kernels = compute_all_kernels(max_size, convolution_type = convolution_type)
    #novelty = novelty_computation(autosimilarity, novelty_kernel_size)
    conv_eight = convolution_entire_matrix_computation(autosimilarity, kernels)
    
    for current_idx in range(1, len(autosimilarity)): # Parse all indexes of the autosimilarity
        for possible_start_idx in possible_segment_start(current_idx, min_size = min_size, max_size = max_size):
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index.")
                
            # Convolutionnal cost between the possible start of the segment and the current index (entire segment)
            conv_cost = convolutionnal_cost(autosimilarity[possible_start_idx:current_idx,possible_start_idx:current_idx], kernels)
            
            # Novelty cost, computed with a fixed kernel (doesn't make sense otherwise), on the end of the segment
            #nov_cost = novelty[current_idx]
            
            segment_length = current_idx - possible_start_idx
            penalty_cost = penalty_cost_from_arg(penalty_func, segment_length)            
            
            # Formula with the mix argument
            #this_segment_cost = (mix * conv_cost + (1 - mix) * nov_cost) * segment_length - penalty_cost * penalty_weight * np.max(conv_eight)
            # Clean formula, to avoid errors.
            this_segment_cost = conv_cost * segment_length - penalty_cost * penalty_weight * np.max(conv_eight)
            # Note: the length of the segment does not appear in conv_eight (not a problem in itself as size is contant, but generally not specified in formulas).

            # Avoiding errors, as segment_cost are initially set to -inf.
            if possible_start_idx == 0:
                if this_segment_cost > costs[current_idx]:
                    costs[current_idx] = this_segment_cost
                    segments_best_starts[current_idx] = 0
            else:
                if costs[possible_start_idx] + this_segment_cost > costs[current_idx]:
                    costs[current_idx] = costs[possible_start_idx] + this_segment_cost
                    segments_best_starts[current_idx] = possible_start_idx

    segments = [(segments_best_starts[len(autosimilarity) - 1], len(autosimilarity) - 1)]
    precedent_frontier = segments_best_starts[len(autosimilarity) - 1] # Because a segment's start is the previous one's end.
    while precedent_frontier > 0:
        segments.append((segments_best_starts[precedent_frontier], precedent_frontier))
        precedent_frontier = segments_best_starts[precedent_frontier]
        if precedent_frontier == None:
            raise err.ToDebugException("Well... Viterbi took an impossible path, so it failed. Understand why.") from None
    return segments[::-1], costs[-1]

def penalty_cost_from_arg(penalty_func, segment_length):
    """
    Returns a penalty cost, function of the size of the segment.
    The penalty function has to be specified, and is bound to evolve in the near future,
    so this docstring won't explain it.
    Instead, you'll have to read the code, sorry! It is pretty straightforward though.

    Parameters
    ----------
    penalty_func : string
        Identifier of the penalty function.
    segment_length : integer
        Size of the segment.

    Returns
    -------
    float
        The penalty cost.

    """
    if penalty_func == "modulo4":        
        # if segment_length % 4 != 0:
        #     return 1/(min(segment_length % 4, -segment_length % 4))
        # else:
        #     return 0
        if segment_length %4 == 0:
            return 0
        elif segment_length %2 == 0:
            return 1/2
        else:
            return 1
    if penalty_func == "modulo8":        
        if segment_length == 8:
            return 0
        elif segment_length %4 == 0:
            return 1/4
        elif segment_length %2 == 0:
            return 1/2
        else:
            return 1
    if penalty_func == "moduloSmall8and4": 
        if segment_length > 12:
            return 100
        elif segment_length == 8:
            return 0
        elif segment_length == 4:
            return 1/4
        elif segment_length %2 == 0:
            return 1/2
        else:
            return 1
    if penalty_func == "sargentdemi": 
         return abs(segment_length - 8) ** (1/2)
    if penalty_func == "sargentun": 
         return abs(segment_length - 8)
    if penalty_func == "sargentdeux": 
         return abs(segment_length - 8) ** 2
    else:
        raise err.InvalidArgumentValueException("Penalty function not understood.")

def possible_segment_start(idx, min_size = 1, max_size = None):
    """
    Generates the list of all possible starts of segments given the index of its end.
    
    Parameters
    ----------
    idx: integer
        The end of a segment.
    min_size: integer
        Minimal length of a segment.
    max_size: integer
        Maximal length of a segment.
        
    Returns
    -------
    list of integers
        All potentials starts of structural segments.
    """
    if min_size < 1: # No segment should be allowed to be 0 size
        min_size = 1
    if max_size == None:
        return range(0, idx - min_size + 1)
    else:
        if idx >= max_size:
            return range(idx - max_size, idx - min_size + 1)
        elif idx >= min_size:
            return range(0, idx - min_size + 1)
        else:
            return []
        
# %% Testing inline convolution, but is in fact very long and not that better.
def compute_full_kernels(max_size, convolution_type = "full"):
    """
    Kernel with diagonals equal to 1. Shouldn't be used.
    """
    kernels = [[0]]
    for p in range(1,max_size + 1):
        if p < 4:
            kern = np.ones((p,p))
        else:
            if convolution_type == "full":
                # Full kernel
                kern = np.ones((p,p))
            elif convolution_type == "eight_bands":
                # Diagonal where only the eight subdiagonals surrounding the main diagonal is one
                k = np.array([np.ones(p-4),np.ones(p-3),np.ones(p-2),np.ones(p-1),np.ones(p),np.ones(p-1),np.ones(p-2),np.ones(p-3),np.ones(p-4)])
                offset = [-4,-3,-2,-1,0,1,2,3,4]
                kern = diags(k,offset).toarray()
            elif convolution_type == "mixed":
                # Sum of both previous kernels
                k = np.array([np.ones(p-4),np.ones(p-3),np.ones(p-2),np.ones(p-1),np.ones(p),np.ones(p-1),np.ones(p-2),np.ones(p-3),np.ones(p-4)])
                offset = [-4,-3,-2,-1,0,1,2,3,4]
                kern = np.ones((p,p)) + np.identity(p) + diags(k,offset).toarray()
            else:
                raise err.InvalidArgumentValueException("Convolution type not understood.")
        kernels.append(kern)
    return kernels

def dynamic_convolution_computation_test_line(autosimilarity, line_conv_weight = 1, min_size = 2, max_size = 36, novelty_kernel_size = 16, penalty_weight = 1, penalty_func = "modulo8", convolution_type = "eight_bands"):
    """
    Segmentation algo with inline convolution test, doesn't work that much in practice.
    """
    costs = [-math.inf for i in range(len(autosimilarity))]
    segments_best_starts = [None for i in range(len(autosimilarity))]
    segments_best_starts[0] = 0
    costs[0] = 0

    kernels = compute_all_kernels(max_size, convolution_type = convolution_type)
    full_kernels = compute_full_kernels(max_size, convolution_type = convolution_type)
    #novelty = novelty_computation(autosimilarity, novelty_kernel_size)
    conv_eight = convolution_entire_matrix_computation(autosimilarity, kernels)
    
    for current_idx in range(1, len(autosimilarity)): # Parse all indexes of the autosimilarity
        for possible_start_idx in possible_segment_start(current_idx, min_size = min_size, max_size = max_size):
            if possible_start_idx < 0:
                raise err.ToDebugException("Invalid value of start index.")
                
            # Convolutionnal cost between the possible start of the segment and the current index (entire segment)
            conv_cost = convolutionnal_cost(autosimilarity[possible_start_idx:current_idx,possible_start_idx:current_idx], kernels)

            # Novelty cost, computed with a fixed kernel (doesn't make sense otherwise), on the end of the segment
            #nov_cost = novelty[current_idx]
            
            segment_length = current_idx - possible_start_idx
            penalty_cost = penalty_cost_from_arg(penalty_func, segment_length)            
            
            current_line_conv_max = 0
            # if possible_start_idx >= segment_length:
            #     for before_start in range(0, possible_start_idx - segment_length + 1):
            #         line_conv_cost = convolutionnal_cost(autosimilarity[possible_start_idx:current_idx,before_start:before_start + segment_length], full_kernels)
            #         if line_conv_cost > current_line_conv_max:
            #             current_line_conv_max = line_conv_cost
            
            # if current_idx + segment_length < len(autosimilarity):
            #     for after_start in range(current_idx, len(autosimilarity) - segment_length):
            #         line_conv_cost = convolutionnal_cost(autosimilarity[possible_start_idx:current_idx,after_start:after_start + segment_length], full_kernels)
            #         if line_conv_cost > current_line_conv_max:
            #             current_line_conv_max = line_conv_cost
            
            mat_vec = []
            if possible_start_idx >= segment_length:
                for before_start in range(0, possible_start_idx - segment_length + 1):
                    mat_vec.append(autosimilarity[possible_start_idx:current_idx,before_start:before_start + segment_length].flatten())

            if current_idx + segment_length < len(autosimilarity):
                for after_start in range(current_idx, len(autosimilarity) - segment_length):
                    mat_vec.append(autosimilarity[possible_start_idx:current_idx,after_start:after_start + segment_length].flatten())

            if mat_vec == []:
                current_line_conv_max = 0
            else:
                kern = full_kernels[segment_length]
                convs_on_line = np.matmul(kern.reshape(1,segment_length**2), np.array(mat_vec).T)                
                current_line_conv_max = np.amax(convs_on_line) / segment_length**2
            
            this_segment_cost = (conv_cost + line_conv_weight * current_line_conv_max) * segment_length - penalty_cost * penalty_weight * np.max(conv_eight)
            # Note: the length of the segment does not appear in conv_eight (not a problem in itself as size is contant, but generally not specified in formulas).

            # Avoiding errors, as segment_cost are initially set to -inf.
            if possible_start_idx == 0:
                if this_segment_cost > costs[current_idx]:
                    costs[current_idx] = this_segment_cost
                    segments_best_starts[current_idx] = 0
            else:
                if costs[possible_start_idx] + this_segment_cost > costs[current_idx]:
                    costs[current_idx] = costs[possible_start_idx] + this_segment_cost
                    segments_best_starts[current_idx] = possible_start_idx

    segments = [(segments_best_starts[len(autosimilarity) - 1], len(autosimilarity) - 1)]
    precedent_frontier = segments_best_starts[len(autosimilarity) - 1] # Because a segment's start is the previous one's end.
    while precedent_frontier > 0:
        segments.append((segments_best_starts[precedent_frontier], precedent_frontier))
        precedent_frontier = segments_best_starts[precedent_frontier]
        if precedent_frontier == None:
            raise err.ToDebugException("Well... Viterbi took an impossible path, so it failed. Understand why.") from None
    return segments[::-1], costs[-1]

# %% Novelty cost, deprecated, but could be used in comparison tests.
def novelty_cost(cropped_autosimilarity):
    """
    Novelty measure on this part of the autosimilarity matrix.
    The size of the kernel will be the size of the parameter matrix.

    Parameters
    ----------
    cropped_autosimilarity : list of list of floats or numpy array (matrix representation)
        The part of the autosimilarity which novelty measure is to compute.

    Raises
    ------
    NotImplementedError
        If the size of the autosimilarity is odd (novlety kernel can't fit this matrix).

    Returns
    -------
    float
        The novelty measure.

    """
    # Kernel is of the size of cropped_autosimilarity
    if len(cropped_autosimilarity) == 0:
        return 0
    
    if len(cropped_autosimilarity) % 2 == 1:
        raise NotImplementedError("Error")
        #return (novelty_cost(cropped_autosimilarity[:-1, :-1]) + novelty_cost(cropped_autosimilarity[1:, 1:])) / 2
    
    kernel_size = int(len(cropped_autosimilarity) / 2)
    kernel = np.kron(np.array([[1,-1], [-1, 1]]), np.ones((kernel_size, kernel_size)))
    return np.mean(kernel*cropped_autosimilarity)
            
    
def novelty_computation(autosimilarity_array, kernel_size):
    """
    Computes the novelty measure of all of the autosimilarity matrix, with a defined and fixed kernel size.

    Parameters
    ----------
    autosimilarity_array : list of list of floats or numpy array (matrix representation)
        The autosimilarity matrix.

    kernel_size : integer
        The size of the kernel.

    Raises
    ------
    NotImplementedError
        If the kernel size is odd, can't compute the novelty measure.

    Returns
    -------
    cost : list of float
        List of novelty measures, at each bar of the autosimilarity.

    """
    if kernel_size % 2 == 1:
        raise NotImplementedError("The kernel should be even.") from None
    cost = np.zeros(len(autosimilarity_array))
    half_kernel = int(kernel_size / 2)
    for i in range(half_kernel, len(autosimilarity_array) - half_kernel):
        cost[i] = novelty_cost(autosimilarity_array[i - half_kernel:i + half_kernel,i - half_kernel:i + half_kernel])
    return cost

# %% Related to the novelty computation, so deprecated.
def peak_picking(tab, window_size = 1):
    """
    Returns the indexes of peaks of values in the given list of values.
    A value is considered "peak" if it's a local maximum,
    and if all values in the window (defined by 'window_size) before and after 
    are strictly monotonous.    
    Used for peak picking in the novelty measure.

    Parameters
    ----------
    tab : list of float
        The list of values to study.
    window_size : boolean, optional
        Size of the window around a possible peak to be considered "peak",
        ie number of consecutive values where the values should increase (before) and (decrease) after.
        The default is 1.

    Returns
    -------
    to_return : list of integers
        The indexes where values are peaking.

    """
    to_return = []
    for current_idx in range(window_size, len(tab) - window_size):
        if is_increasing(tab[current_idx - window_size:current_idx + 1]) and is_increasing(tab[current_idx:current_idx + window_size + 1][::-1]):
            to_return.append(current_idx)
    return to_return

def valley_picking(tab, window_size = 1):
    """
    Returns the indexes of valleys of values in the desired list of values.
    A value is considered "valley" if it's a local minimum,
    and if all values in the window (defined by 'window_size) before and after 
    are strictly monotonous.
    Used for peak picking in the novelty measure.

    Parameters
    ----------
    tab : list of float
        The list of values to study.
    window_size : boolean, optional
        Size of the window around a possible valley to be considered "valley",
        ie number of consecutive values where the values should decrease (before) and increase (after).
        The default is 1.

    Returns
    -------
    to_return : list of integers
        The indexes where values are valleys.

    """
    to_return = []
    for current_idx in range(window_size, len(tab) - window_size):
        if is_increasing(tab[current_idx - window_size:current_idx + 1][::-1]) and is_increasing(tab[current_idx:current_idx + window_size + 1]):
            to_return.append(current_idx)
    return to_return

def is_increasing(tab):
    """
    Tests if the tab values are increasing.
    Used for peak picking in the novelty measure.

    Parameters
    ----------
    tab : list of float
        The values.

    Returns
    -------
    boolean
        Whether the values are increasing or not.

    """
    if len(tab) <= 1 or len(np.unique(tab)) == 1:
        return False
    for idx in range(len(tab) - 1):
        if tab[idx] > tab[idx+1]:
            return False
    return True

def decreasing_peaks(data):
    """
    Returns the peaks indexes of a list of values in their decreasing order of values.
    Used for peak picking in the novelty measure.

    Parameters
    ----------
    data : list of float
        The values.

    Returns
    -------
    list of integers
        The indexes of the peaks, sorted in their decreasing order of values.

    """
    peaks_and_value = []
    for idx in peak_picking(data, window_size = 1):
        peaks_and_value.append((idx, data[idx]))
    return sorted(peaks_and_value, key=lambda x:x[1], reverse = True)

def select_highest_peaks_thresholded_indexes(data, percentage = 0.33):
    """
    Returns the peaks higher than a percentage of the maximal peak from a list of values.
    Used for peak picking in the novelty measure.
    
    Parameters
    ----------
    data : list of floats
        The values.
    percentage : float, optional
        The percentage of the maximal value for a peak to be valid.
        The default is 0.33.

    Returns
    -------
    list of integers
        Indexes of the valid peaks.

    """
    peaks = np.array(decreasing_peaks(data))
    max_peak = peaks[0,1]
    for idx, peak in enumerate(peaks):
        if peak[1] < percentage * max_peak:
            return [int(i) for i in sorted(peaks[:idx, 0])]
    return [int(i) for i in sorted(peaks[:,0])]

def mean(val_a, val_b):
    """
    A function returning the mean of both values.
    This function is redeveloped so as to be called as choice_func in the function "values_as_slop()" (see below) in external projects.

    Parameters
    ----------
    val_a : float
        First value.
    val_b : float
        Second value.

    Returns
    -------
    float: mean of both values.

    """
    return (val_a + val_b) / 2

def values_as_slop(value, choice_func = max):
    """
    Compute peaks of a value (typically novelty measure)
    as the difference between absolute peaks and absolute valleys.
    Function choice_func determines the way of computing this gap.
    
    Typically, max will compute peaks as the maximum gap between a peaks and its two closest valleys,
    whereas min will select the minimal gap.
    
    This returns an array containing zeroes where there is no peak in absoluite value,
    and this new value as a gap computation where there was peaks before.

    Parameters
    ----------
    value : array of float
        The absolute value of the measure.
    choice_func : function name, optional
        Type of the function selecting the difference between peaks and valleys.
        Classical values are "max" for selecting the maximum gap between the peak and both its closest valleys,
        "min" for the minimum of both gaps, and "mean" (called autosimilarity_segmentation.mean) for the mean of both gaps.
        The default is max.

    Returns
    -------
    peak_valley_slop : array of floats
        The new values of peaks as gaps, and 0 everywhere else.

    """
    peaks = peak_picking(value, window_size = 1)
    valleys = valley_picking(value, window_size = 1)
    peak_valley_slop = np.zeros(len(value))
    for peak in peaks:
        i = 0
        while i < len(valleys) and valleys[i] < peak:
            i+=1
        if i == 0:
            left_valley = 0
            right_valley = valleys[i]
        elif i == len(valleys):
            left_valley = valleys[i - 1]
            right_valley = 0
        else:
            left_valley = valleys[i - 1]
            right_valley = valleys[i]
        chosen_valley_value = choice_func(value[left_valley], value[right_valley])
        peak_valley_slop[peak] = value[peak] - chosen_valley_value
    return peak_valley_slop
    
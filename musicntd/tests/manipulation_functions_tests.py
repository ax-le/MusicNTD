# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:47:27 2020

@author: amarmore
"""
# Tests on development of the NTD algorithm.

import unittest
import random
import numpy as np
import musicntd.data_manipulation as dm
import musicntd.autosimilarity_segmentation as auto_seg
import musicntd.tensor_factory as tf
import musicntd.model.errors as err

class ManipTests(unittest.TestCase):
    
    def test_ends_to_seg(self):
        """
        Little test for the "frontiers_to_segments()" function from auto_seg.
        """
        self.assertEqual(dm.frontiers_to_segments([3,7,19,24]), [(0,3),(3,7),(7,19),(19,24)])
        
    def test_seg_to_ends(self):
        """
        Little test for the "segments_to_frontiers()" function from auto_seg.
        """
        self.assertEqual(dm.segments_to_frontiers([(0,3),(3,7),(7,19),(19,24)]), [3,7,19,24])
        
    def test_get_annotation_name_from_song(self):
        """Testing the function formatting the name of the annotation starting from its name."""
        self.assertEqual(dm.get_annotation_name_from_song(1, "MIREX10"), "RM-P001.BLOCKS.lab")
        self.assertEqual(dm.get_annotation_name_from_song("1", "MIREX10"), "RM-P001.BLOCKS.lab")        
        self.assertEqual(dm.get_annotation_name_from_song(1, "AIST"), "RM-P001.CHORUS.TXT")
        self.assertEqual(dm.get_annotation_name_from_song("1", "AIST"), "RM-P001.CHORUS.TXT") 
        with self.assertRaises(err.InvalidArgumentValueException):
            dm.get_annotation_name_from_song("1", "Another set of annot")
            
    def test_frontiers_from_time_to_frame_idx(self):
        """Testing the function converting frontiers from time to frame indexes."""
        hop_length_seconds = 1024/44100 # Pretty standard value
        indexes = [0,1,8,10,15]
        seconds = []
        for k in indexes:
            seconds.append(k*hop_length_seconds)
        self.assertEqual(dm.frontiers_from_time_to_frame_idx(seconds, hop_length_seconds), indexes)   
        
    def test_segments_from_time_to_frame_idx(self):
        """Testing the function converting segments from time to frame indexes."""
        hop_length_seconds = 1024/44100 # Pretty standard value
        indexes = [[0,1],[1,8],[8,10],[10,15]]
        seconds = []
        for k in indexes:
            seconds.append((k[0]*hop_length_seconds,k[1]*hop_length_seconds))
        self.assertEqual(dm.segments_from_time_to_frame_idx(seconds, hop_length_seconds), indexes)  

    def test_frontiers_from_time_to_bar(self):
        """Testing the function converting frontiers from time to bar indexes."""
        hop_length_seconds = 1024/44100 # Pretty standard value
        indexes = [[0,1],[1,8],[8,10],[10,15]]
        bars = []
        for k in indexes:
            bars.append((k[0]*hop_length_seconds,k[1]*hop_length_seconds))
        time_frames = []
        for k in range(0, 20, 2):
            time_frames.append(k*hop_length_seconds)
        self.assertEqual(dm.frontiers_from_time_to_bar(time_frames, bars), [0, 0, 0, 1, 1, 2, 2, 3])
        
    def test_frontiers_from_bar_to_time(self):
        """Testing the function converting frontiers from bar indexes to time."""
        hop_length_seconds = 1024/44100 # Pretty standard value
        indexes = [[0,1],[1,8],[8,10],[10,15],[15,20]]
        bars = []
        for k in indexes:
            bars.append((k[0]*hop_length_seconds,k[1]*hop_length_seconds))
        self.assertEqual(dm.frontiers_from_bar_to_time([0,2,4], bars), [bars[0][1], bars[2][1], bars[4][1]])
        
    def test_compute_score_of_segmentation(self):
        """Testing the function computing segmentation scores from segments."""
        estimated = np.array(dm.frontiers_to_segments([1,2,5,7,9]))
        reference = np.array(dm.frontiers_to_segments([0,2,4,6,8,10]))
        self.assertEqual(dm.compute_score_of_segmentation(reference, estimated, window_length = 0.5), (3/7,4/8,(2 * 3/7 * 4/8)/(3/7 + 4/8)))
        self.assertEqual(dm.compute_score_of_segmentation(reference, estimated, window_length = 1.5), (6/7,1,(2 * 6/7)/(6/7 + 1)))

    def test_compute_rates_of_segmentation(self):
        """Testing the function computing segmentation rates (TP, FP, FN) from segments."""
        estimated = np.array(dm.frontiers_to_segments([1,2,5,7,9]))
        reference = np.array(dm.frontiers_to_segments([0,2,4,6,8,10]))
        self.assertEqual(dm.compute_rates_of_segmentation(reference, estimated, window_length = 0.5), (3,4,3))
        self.assertEqual(dm.compute_rates_of_segmentation(reference, estimated, window_length = 1.5), (6,1,0))
        
    def test_is_increasing(self):
        """
        Little test for the "is_increasing()" function from auto_seg.
        """
        self.assertTrue(auto_seg.is_increasing([3,7,19,24]))
        self.assertTrue(auto_seg.is_increasing([1,1,1,2]))
        self.assertFalse(auto_seg.is_increasing([1,1,1,1]))
        self.assertFalse(auto_seg.is_increasing([1,1,3,2]))
        self.assertFalse(auto_seg.is_increasing([1]))
        self.assertFalse(auto_seg.is_increasing([]))
        
    def test_peak_picking(self):
        """
        Little test for the "peak_picking()" function from auto_seg.
        """
        self.assertEqual(auto_seg.peak_picking([1,2,1,2,3,4,5,4,3,2], window_size = 1),
                         [1, 6])
        self.assertEqual(auto_seg.peak_picking([1,2,1,2,3,4,5,4,3,2], window_size = 2),
                         [6])
        self.assertEqual(auto_seg.peak_picking([1,2,1,2,3,4,5,4,5,3,2], window_size = 2),
                         [])
    
    def test_longest_bar_len(self):
        self.assertEqual(tf.longest_bar_len(dm.frontiers_to_segments([0,5,15,19,20])), 10)

if __name__ == '__main__':
    unittest.main()
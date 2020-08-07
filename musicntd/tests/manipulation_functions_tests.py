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
        self.assertEqual(tf.longest_bar_len([0,5,15,19,20]), 10)

if __name__ == '__main__':
    unittest.main()
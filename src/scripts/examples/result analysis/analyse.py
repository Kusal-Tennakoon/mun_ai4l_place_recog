#!/usr/bin/env python3

import rospy
import os
import sys

from mun_ai4l_place_recog.mun_ai4l_prnlc import Analyser

if __name__ == '__main__':
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Current directory
    src_dir = os.path.abspath(os.path.join(current_dir, '../../..')) # File path of the src directory
    
    analyser = Analyser() # Analyser Object
    
    ## Parameters of the test results (Alter to suit your reference data)
    # -------------------------------------------------------------------
    
    analyser.dataset_name = "Example_dataset" # Name for the test dataset
    analyser.data_dir = os.path.join(current_dir,"data/") # File path to results data
    analyser.res_dir = current_dir # File path to save the analysis results
    
    # ------------------------------------------------------------------------
    
    analyser.initialize() # Initializing and setup the data analyser
    analyser.analyse_lc_time() # Analyse time data
    analyser.save_lc_results() # Save loop closure detection results
    analyser.save_pr_results() # Save place recognition results
    


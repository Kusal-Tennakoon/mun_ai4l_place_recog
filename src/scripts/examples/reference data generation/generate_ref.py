#!/usr/bin/env python3

import rospy
import os
import sys

from mun_ai4l_place_recog.mun_ai4l_prnlc import DataGenerator

if __name__ == '__main__':
    
    generator = DataGenerator() # Data generator Object
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Current directory
    src_dir = os.path.abspath(os.path.join(current_dir, '../../..')) # File path of the src directory
    

    ## Parameters of the reference data (Alter to suit your reference data)
    # ----------------------------------------------------------------------
    
    generator.dataset_name = "Example_dataset" # Name for the reference dataset
    generator.no_of_images = 6 # Number of reference images
    generator.image_size = 320 # Size of the reference images (eg. 320 px x 320 px)
    generator.image_type = "png" # File type of reference images
    generator.image_dir = os.path.join(current_dir,"images/") # File path of the directory containing the reference images
    generator.gps_file = os.path.join(current_dir,"images/gps_coords.txt") # File path of the text file with the reference GPS coordinates
    generator.data_dir = os.path.join(current_dir,generator.dataset_name + "_ref_data/") # File path of the directory to store generated reference data
    generator.device = '/CPU:0' # Device used '/CPU:0' or '/GPU:0'
    generator.weights = os.path.join(src_dir,"data/netvlad_weights.h5")
    
    # ------------------------------------------------------------------------
    
    # Creating a directory to store reference data
    if not os.path.exists(generator.data_dir):
        try:
            os.mkdir(generator.data_dir)
            print("[Data Generator]: " + generator.dataset_name + "_ref_data folder created !")
        except OSError as error:
            print(error)
    
    generator.initialize() # Initializing and setup the data generator
    generator.generate_data() # Generate reference data files
    


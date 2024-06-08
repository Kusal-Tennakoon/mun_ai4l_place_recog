#!/usr/bin/env python3

import rospy
import os
import sys

from mun_ai4l_place_recog.mun_ai4l_prnlc import Nodes   # If (i). input image msg type => Image. (ii). Save results for analysis => False


if __name__ == "__main__":

    # config_file = "/home/lupus/catkin_ws/src/mun_ai4l_place_recog/config/config_bell.yaml" # Bell 412 dataset
    config_file = "/home/lupus/catkin_ws/src/mun_ai4l_place_recog/config/config_lighthouse.yaml" # Lighthouse dataset
    # config_file = "/home/lupus/catkin_ws/src/mun_ai4l_place_recog/config/config_mars.yaml" # MarsLab dataset
    # config_file = "/home/lupus/catkin_ws/src/mun_ai4l_place_recog/config/config_vicar.yaml" # vicar dataset

    Nodes.ParameterServer(config_file) # Parameter server

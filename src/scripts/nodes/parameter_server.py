#!/usr/bin/env python3

import rospy
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')), 'modules/'))
from mun_ai4l_prnlc import Nodes

if __name__ == "__main__":

    # config_file = "/home/lupus/catkin_ws/src/mun_ai4l_place_recog/config/config_bell.yaml" # Bell 412 dataset
    config_file = "/home/lupus/catkin_ws/src/mun_ai4l_place_recog/config/config_lighthouse.yaml" # Lighthouse dataset
    # config_file = "/home/lupus/catkin_ws/src/mun_ai4l_place_recog/config/config_mars.yaml" # MarsLab dataset
    # config_file = "/home/lupus/catkin_ws/src/mun_ai4l_place_recog/config/config_vicar.yaml" # vicar dataset

    Nodes.ParameterServer(config_file) # Parameter server

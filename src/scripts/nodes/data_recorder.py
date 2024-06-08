#!/usr/bin/env python3

import rospy
import os
import sys

from mun_ai4l_place_recog.mun_ai4l_prnlc import Nodes

if __name__ == '__main__':

    Nodes.Data_recorder() # Start data recorder and visualizer

#!/usr/bin/env python3

import rospy
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')), 'modules/'))
from mun_ai4l_prnlc_eval import Nodes

if __name__ == '__main__':

    Nodes.Data_recorder() # Start data recorder and visualizer

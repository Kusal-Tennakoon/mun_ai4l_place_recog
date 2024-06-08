#!/usr/bin/env python3

import rospy
import os
import sys

# from mun_ai4l_place_recog.mun_ai4l_prnlc_comp_eval import Nodes # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => True
# from mun_ai4l_place_recog.mun_ai4l_prnlc_eval import Nodes  # If (i). input image msg type => Image. (ii). Save results for analysis => True
# from mun_ai4l_place_recog.mun_ai4l_prnlc_comp import Nodes  # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => False
from mun_ai4l_place_recog.mun_ai4l_prnlc import Nodes   # If (i). input image msg type => Image. (ii). Save results for analysis => False

if __name__ == '__main__':

    Nodes.PR() # Start place recognition and loop closure node

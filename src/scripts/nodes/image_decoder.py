#!/usr/bin/env python3

import rospy
import warnings
import os
import sys
import cv2 # OpenCV
import faiss # Faiss
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

bridge = CvBridge()

def convert(msg):

    # Convert CompressedImage msg to cv2 image
    q_img = cv2.imdecode(np.fromstring(msg.data, np.uint8), cv2.IMREAD_COLOR) # CompressedImage msg -> cv2 image
    img_msg = bridge.cv2_to_imgmsg(q_img,"bgr8") # cv2 image -> Image msg
    pub.publish(img_msg)


if __name__ == '__main__':

    while not rospy.is_shutdown():
        rospy.init_node('image_Decoder', anonymous=False) # Initializing the node
        rospy.loginfo("Image decoder activated!\n")
        pub = rospy.Publisher("/camera/image_mono/",Image,queue_size=100)
        subs = rospy.Subscriber("/left_camera/image/compressed/",CompressedImage,convert,queue_size=100)

        rospy.spin()

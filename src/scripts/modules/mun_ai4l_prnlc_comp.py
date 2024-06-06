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
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation #scipy
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import regularizers, initializers, layers
from tensorflow.keras.layers import Conv2D,Dense, Dropout, Conv2D, Activation, Input, concatenate, Lambda, ZeroPadding2D, MaxPooling2D, Layer, Flatten
import tensorflow.keras.backend as Kback
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.backend import l2_normalize, expand_dims, variable, constant
import json
import yaml
from time import time, gmtime, strftime
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from mun_ai4l_place_recog.msg import PRmatch, LCmatch, PRmetrics, LCmetrics

# sys.path.append(os.path.abspath(__file__))

################################################################################
# ----------------------------- NetVLAD class ----------------------------------
################################################################################

class NetVLAD(layers.Layer):

    def __init__(self, num_clusters, assign_weight_initializer=None,cluster_initializer=None, skip_postnorm=False, **kwargs):
        self.K = num_clusters
        self.assign_weight_initializer = assign_weight_initializer
        self.skip_postnorm = skip_postnorm
        self.outdim = 32768
        super(NetVLAD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.D = input_shape[-1]
        self.C = self.add_weight(name='cluster_centers',
                                shape=(1,1,1,self.D,self.K),
                                initializer='zeros',
                                dtype='float32',
                                trainable=True)

        self.conv = Conv2D(filters = self.K,kernel_size=1,strides = (1,1),use_bias=False, padding = 'valid',kernel_initializer='zeros')
        self.conv.build(input_shape)
        #self._trainable_weights.append(self.conv.trainable_weights[0])
        super(NetVLAD, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        s = self.conv(inputs)
        a = tf.nn.softmax(s)
        a = tf.expand_dims(a,-2)
        v = tf.expand_dims(inputs,-1)+self.C
        v = a*v
        v = tf.reduce_sum(v,axis=[1,2])
        v = tf.transpose(v,perm=[0,2,1])

        if not self.skip_postnorm:
            v = self.matconvnetNormalize(v, 1e-12)
            v = tf.transpose(v, perm=[0, 2, 1])
            v = self.matconvnetNormalize(tf.compat.v1.layers.flatten(v), 1e-12)

        return v

    def matconvnetNormalize(self,inputs, epsilon):
        return inputs / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=-1, keepdims=True)+ epsilon)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.outdim])

################################################################################
# ------------------------------ NetVLAD ---------------------------------------
################################################################################

def NetVLADModel(outputsize = 4096,input_shape=(None,None,3)):

    model = Sequential()

    model.add(SubstractAverage(input_shape = input_shape))  #0
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))    #1
    model.add(Conv2D(64, (3, 3), padding="same"))   #2
    model.add(MaxPooling2D(strides=(2,2)))  #3
    model.add(Activation('relu'))   #4

    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))   #5
    model.add(Conv2D(128, (3, 3), padding="same"))  #6
    model.add(MaxPooling2D(strides=(2,2)))  #7
    model.add(Activation('relu'))   #8

    model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))   #9
    model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))   #10
    model.add(Conv2D(256, (3, 3), padding="same"))     #11
    model.add(MaxPooling2D(strides=(2,2)))  #12
    model.add(Activation('relu'))   #13

    model.add(Conv2D(512, (3, 3), activation='relu', padding="same"))   #14
    model.add(Conv2D(512, (3, 3), activation='relu', padding="same"))   #15
    model.add(Conv2D(512, (3, 3), padding="same"))  #16
    model.add(MaxPooling2D(strides=(2,2)))  #17
    model.add(Activation('relu'))   #18

    model.add(Conv2D(512, (3, 3), activation='relu', padding="same"))   #19
    model.add(Conv2D(512, (3, 3), activation='relu', padding="same"))   #20
    model.add(Conv2D(512, (3, 3), padding="same"))  #21

    model.add(Lambda(lambda a: l2_normalize(a,axis=-1)))    #22
    model.add(NetVLAD(num_clusters=64)) #23

    #PCA
    model.add(Lambda(lambda a: expand_dims(a,axis=1)))  #24
    model.add(Lambda(lambda a: expand_dims(a,axis=1)))  #25
    model.add(Conv2D(4096,(1,1)))   #26
    model.add(Flatten())    #27
    model.add(Lambda(lambda a: l2_normalize(a,axis=-1)))    #28

    sgd = SGD(learning_rate=0.001, decay=0.001, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

#Initializes pre-trained average RGB
def average_rgb_init(shape, dtype=None):
    return np.array([123.68, 116.779, 103.939])

# ------------- Custom layer for subtracting a tensor from another -------------
class SubstractAverage(Layer):

    def __init__(self, **kwargs):
        super(SubstractAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.average_rgb = self.add_weight(name='average_rgb',
                                            initializer=average_rgb_init,
                                            shape=(3,),
                                            dtype='float32',
                                            trainable=False)

        super(SubstractAverage, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        #subtract
        v = inputs - self.average_rgb
        return v

################################################################################
# -------------------------- Loop closure class --------------------------------
################################################################################

class LoopClosure:

    def __init__(self):

        # Loop clousre params
        self.time_last = rospy.Time(0)
        self.queue_size = 1
        self.device = '/' + rospy.get_param('pr_device') + ':0' # '/CPU:0' or '/GPU:0'
        self.data_dir = rospy.get_param('data_dir_path')
        self.image_topic = rospy.get_param('lc_cam_img_topic')
        self.rate = rospy.get_param('loop_close_rate')
        self.delay = rospy.Duration(rospy.get_param('lc_delay')) # Minimum delay between two consecutive frames added to the index
        self.n_skip = rospy.get_param('lc_n_skip') # No of frames to before current frame that needs to be skipped when searching for matches
        self.dist_thresh = rospy.get_param('lc_dist_thresh')
        self.dist_ratio = rospy.get_param('lc_dist_ratio')
        self.p_thresh = rospy.get_param("lc_p_thresh")
        self.ransac_thresh = rospy.get_param("lc_ransac_thresh")
        self.min_match_count = rospy.get_param("lc_min_match_count")
        self.n_features = rospy.get_param("lc_n_features") # Minimum no. of sift features
        self.n_neighbours = rospy.get_param("lc_n_neighbours") # No of nearest neighbours
        self.image_scale = rospy.get_param("lc_image_scale")
        self.input_img_dim = rospy.get_param("lc_input_img_dim")
        self.index_params = dict(algorithm = 0, trees = 5) # algorithm = FLANN_INDEX_KDTREE = 0
        self.search_params = dict(checks = 100)
        self.flann = cv2.FlannBasedMatcher(self.index_params,self.search_params) # Flann based matcher
        self.brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        self.sift = cv2.xfeatures2d.SIFT_create(self.n_features) # Sift feature extractor
        self.orb = cv2.ORB_create()
        self.db = faiss.IndexFlatL2(4096) # Faiss index
        self.bridge = CvBridge() # CV bridge
        self.frame_index = 0 # Index of the current frame
        self.frames = {} # Dictionary to store frames
        self.candidates = [] # List to store candidate matches
        self.can_R = [] # List to store candidate rotation matrices
        self.can_t = [] # List to store candidate upto scale translations
        self.upper_index = self.n_skip + self.n_neighbours
        self.lower_index = self.n_skip - 1
        # Camera params
        self.fov_x = rospy.get_param("cam_fov_x")
        self.fov_y = rospy.get_param("cam_fov_y")
        self.fx = rospy.get_param("ref_cam_fx")
        self.fy = rospy.get_param("ref_cam_fy")
        self.cx = rospy.get_param("ref_cam_cx")
        self.cy = rospy.get_param("ref_cam_cy")
        self.distCoeff = np.array(rospy.get_param("ref_cam_dist_coeff"))
        self.K = np.matrix([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]]) # Camera matrix object for the camera

        # Publishers
        self.pub_lc_match = rospy.Publisher('/loop_closure/match',LCmatch,queue_size=self.queue_size) # Best match for LC

        # ROS messages
        self.rel_pose = PoseStamped() # Variable to store relative pose
        self.lc_match = LCmatch() # Variable to store loop closure match

        # CNN initialization
        self.model = NetVLADModel() # Create netVLAD CNN
        self.model.build(input_shape=(None,self.input_img_dim,self.input_img_dim,3))
        self.model.load_weights(self.data_dir + rospy.get_param('model_weights'))

        rospy.loginfo("[LC]: Initialization successful!") ## DEBUGGING ##

    def detectLC(self,msg):

        with tf.device(self.device):

            if ((rospy.Time.now() - self.time_last) > self.delay):

                # Convert CompressedImage msg to cv2 image
                q_img = cv2.imdecode(np.fromstring(msg.data, np.uint8), cv2.IMREAD_COLOR) # CompressedImage msg -> cv2 image
                q_img = cv2.resize(q_img,(0,0),fx = self.image_scale,fy = self.image_scale,interpolation = cv2.INTER_AREA)

                msg = self.bridge.cv2_to_imgmsg(q_img,"bgr8") # cv2 image -> Image msg

                # Extracting SIFT descriptors
                q_img_gray = cv2.cvtColor(q_img,cv2.COLOR_BGR2GRAY)
                # q_img_gray = cv2.resize(q_img_gray,(0,0),fx = self.image_scale,fy = self.image_scale,interpolation = cv2.INTER_AREA)
                kp,des = self.sift.detectAndCompute(q_img_gray,None)
                # kp,des = self.orb.detectAndCompute(q_img_gray,None)

                # Converting image to vlad vector
                q_img_small = cv2.resize(q_img,(self.input_img_dim,self.input_img_dim))
                q_img_small = cv2.cvtColor(q_img_small,cv2.COLOR_BGR2RGB)
                q_img_vlad = q_img_small.astype(np.float32)
                q_img_vlad = np.expand_dims(q_img_vlad,axis=0)
                q_vlad = self.model.predict(q_img_vlad,use_multiprocessing=True)

                # img_msg = self.bridge.cv2_to_imgmsg(cv2.resize(q_img,(80,80)),encoding="bgr8")  ## DEBUGGING ##

                # Adding the frame into a dictionary
                self.frames[self.frame_index] = {'index' : self.frame_index,
                                                  'timestamp' : msg.header.stamp,
                                                  'image' : msg,
                                                  'keypoints' : kp,
                                                  'descriptors' : des,
                                                  'vlad' : q_vlad
                                                  }

                if (self.frame_index > self.upper_index):

                    # Adding vlad vector to the faiss index
                    self.db.add(self.frames[self.frame_index-self.n_skip]['vlad'])

                    # Searching putative matches for the query vlad vector
                    sq_distances, matches = self.db.search(q_vlad,self.n_neighbours)

                    # rospy.loginfo(f"[LC]: {self.frame_index} -> {sq_distances} -> {matches}")  ## DEBUGGING ##

                    # Second stage verification

                    if (sq_distances[0][0] <=  self.dist_thresh): # Is the min distance less than the threshold(1.0) ?

                        # Sift keypoints and descriptors of the current image
                        kp_q, des_q = self.frames[self.frame_index]['keypoints'], self.frames[self.frame_index]['descriptors']

                        for dist_rat,match_index in zip(sq_distances[0]/sq_distances[0][0],matches[0]):
                            # Distance thresholding
                            if dist_rat <=  self.dist_ratio:

                                # Sift keypoints and descriptors of the putative match
                                kp_m, des_m = self.frames[match_index]['keypoints'], self.frames[match_index]['descriptors']

                                # Homography check

                                # for SIFT
                                des_matches = self.flann.knnMatch(des_q,des_m,k=2)
                                good_matches = [first for first,second in des_matches if first.distance < 0.7*second.distance]

                                # for ORB
                                # des_matches = self.brute_force.match(des_q,des_m)
                                # good_matches = sorted(des_matches, key = lambda x:x.distance)

                                # rospy.loginfo(f"[LC]: good matches -> {len(good_matches)}") ## DEBUGGING ##

                                if len(good_matches) > self.min_match_count: # MIN_MATCH_COUNT = 10 (Set a value higher than 10)
                                    query_pts = np.float32([kp_q[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                                    match_pts = np.float32([kp_m[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

                                    query_norm = cv2.undistortPoints(query_pts,self.K,self.distCoeff)
                                    match_norm = cv2.undistortPoints(match_pts,self.K,self.distCoeff)

                                    F,mask = cv2.findFundamentalMat(query_norm,match_norm, cv2.RANSAC,self.ransac_thresh,0.99)
                                    p_match = 2 * np.sum(mask)/(len(kp_q)+len(kp_m)) # no of inliers = np.sum(mask)
                                    # rospy.loginfo(f"[LC]: Match frame = {self.frames[match_index]['index']} --> P match = {p_match}") ## DEBUGGING ##
                                    # rospy.loginfo(f"[LC]: F = {F}") ## DEBUGGING ##

                                    if (F is not None) and (p_match >= self.p_thresh):
                                        E = np.matmul(np.matmul(self.K.transpose(),F),self.K)
                                        # src_correct, dst_correct = cv2.correctMatches(F, src_norm[:,0,:], dst_norm[:,0,:])
                                        pts, R, t, mask = cv2.recoverPose(E, query_norm, match_norm)

                                        self.candidates.append(match_index)
                                        self.can_R.append(R)
                                        self.can_t.append(t)

                        # Finding the prediction with the minimum index
                        if np.any(self.candidates): # Check whether candidates is empty
                            arg_min = np.argmin(self.candidates)
                            min_index = self.candidates[arg_min] # same as np.min(self.candidates)
                            min_t = self.can_t[arg_min]
                            min_th = Rotation.from_matrix(self.can_R[arg_min]).as_quat()

                            # rospy.loginfo(f"[LC] : Loop detected ! \n --> current frame = {self.frame_index} \n --> match frame = {min_index} \n") ## DEBUGGING ##
                            rospy.loginfo(f"[LC] : Loop detected ! {self.frame_index} --> {min_index} \n") ## DEBUGGING ##

                            self.lc_match.query_time = self.frames[self.frame_index]['timestamp']
                            self.lc_match.query_image = self.frames[self.frame_index]['image']
                            self.lc_match.match_time = self.frames[min_index]['timestamp']
                            self.lc_match.match_image = self.frames[min_index]['image']
                            self.lc_match.rel_pose.position.x = min_t[0]
                            self.lc_match.rel_pose.position.y = min_t[1]
                            self.lc_match.rel_pose.position.z = min_t[2]
                            self.lc_match.rel_pose.orientation.x = min_th[0]
                            self.lc_match.rel_pose.orientation.y = min_th[1]
                            self.lc_match.rel_pose.orientation.z = min_th[2]
                            self.lc_match.rel_pose.orientation.w = min_th[3]

                            self.pub_lc_match.publish(self.lc_match)

                            # self.candidates.clear()
                            # self.can_R.clear()
                            # self.can_t.clear()
                            # self.can_p.clear()

                elif (self.frame_index > self.lower_index):
                    self.db.add(self.frames[self.frame_index - self.n_skip]['vlad'])

                self.frame_index +=1
                self.candidates.clear()
                self.can_R.clear()
                self.can_t.clear()
                self.can_p.clear()
                # rospy.loginfo(f"[LC]: Frame added... --> index = {self.frame_index}")  ## DEBUGGING ##
                self.time_last = rospy.Time.now()

################################################################################
# ------------------------ Place recognition class -----------------------------
################################################################################

class PlaceRecognition:

    def __init__(self):

        # Place recog params
        self.time_last = rospy.Time(0)
        self.queue_size = 1
        self.device = '/' + rospy.get_param('pr_device') + ':0' # '/CPU:0' or '/GPU:0'
        self.data_dir = rospy.get_param('data_dir_path')
        self.image_topic = rospy.get_param('pr_cam_img_topic')
        self.rate = rospy.get_param('place_recog_rate')
        self.delay = rospy.Duration(rospy.get_param('pr_delay')) # Delay between two consecutive checks
        self.dist_thresh = rospy.get_param('pr_dist_thresh')
        self.dist_ratio = rospy.get_param('pr_dist_ratio')
        self.n_neighbours = rospy.get_param("pr_n_neighbours") # No of nearest neighbours
        self.image_scale = rospy.get_param("pr_image_scale")
        self.input_img_dim = rospy.get_param("pr_input_img_dim")
        self.db = faiss.IndexFlatL2(4096) # Faiss index
        self.locations = {}
        self.bridge = CvBridge() # CV bridge
        self.gps_coord = NavSatFix() # Variable to store GPS coordinate

        # ROS messages
        self.pr_match = PRmatch() # Variable to store place recog match

        # Publishers
        self.pub_pr_match = rospy.Publisher('/place_recog/match',PRmatch,queue_size=self.queue_size) # Best match for PR

        # CNN initialization
        self.model = NetVLADModel() # Create netVLAD CNN
        self.model.build(input_shape=(None,self.input_img_dim,self.input_img_dim,3))
        self.model.load_weights(self.data_dir + rospy.get_param('model_weights'))

        rospy.loginfo("[PR]: Initialization successful!") ## DEBUGGING ##

    def load_data(self):
        # Loading reference data
        ref_imgs = np.load(self.data_dir + rospy.get_param('ref_imgs'),allow_pickle=True)
        ref_gps = np.load(self.data_dir + rospy.get_param('ref_gps'),allow_pickle=True)
        ref_vlad = np.load(self.data_dir + rospy.get_param('ref_vlad'),allow_pickle=True)

        self.locations = {i : {'image' : img,
                                 'lat' : lat,
                                 'lon' : lon,
                                 'alt' : alt
                                 } for i,(img,(lat,lon,alt)) in enumerate(zip(ref_imgs,ref_gps))} # Reference data

        rospy.loginfo("[PR] : Database loading successful!")  ## DEBUGGING ##

        # Adding reference vlad vectors to the faiss index
        self.db.add(ref_vlad)

        del ref_imgs,ref_gps, ref_vlad # Delete temp var

    def detectPR(self,msg):

        with tf.device(self.device):

            now = rospy.Time.now()

            if ((now - self.time_last) > self.delay):

                # Convert CompressedImage msg to cv2 image
                q_img = cv2.imdecode(np.fromstring(msg.data, np.uint8), cv2.IMREAD_COLOR) # CompressedImage msg -> cv2 image
                q_img = cv2.resize(q_img,(0,0),fx = self.image_scale,fy = self.image_scale,interpolation = cv2.INTER_AREA)
                # msg = self.bridge.cv2_to_imgmsg(q_img,"bgr8") # cv2 image -> Image msg


                # Converting image to vlad vector
                q_img = cv2.resize(q_img,(self.input_img_dim,self.input_img_dim))
                q_img = cv2.cvtColor(q_img,cv2.COLOR_BGR2RGB)
                q_img = q_img.astype(np.float32)
                q_img = np.expand_dims(q_img,axis=0)
                q_vlad = self.model.predict(q_img,use_multiprocessing=True)

                # Searching putative matches for the query vlad vector
                sq_distances, matches = self.db.search(q_vlad,self.n_neighbours)

                # Second stage verification
                if (sq_distances[0][0] <=  self.dist_thresh): # Is the min distance less than the threshold(1.0) ?
                    match_index = matches[0][0]

                    rospy.loginfo(f"[PR] : Match detected {self.frame_index} --> {match_index}\n") # \n --> sq_dist = {sq_distances[0][0]}\n")  ## DEBUGGING ##

                    img_msg = self.locations[match_index]['image']
                    img_msg = cv2.cvtColor(img_msg.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    # img_msg = cv2.resize(img_msg,(160,160))
                    img_msg = self.bridge.cv2_to_imgmsg(img_msg,encoding="bgr8")

                    self.pr_match.query_time = msg.header.stamp
                    self.pr_match.match_node = match_index
                    self.pr_match.match_gps.latitude = self.locations[match_index]['lat']
                    self.pr_match.match_gps.longitude = self.locations[match_index]['lon']
                    self.pr_match.match_gps.altitude = self.locations[match_index]['alt']
                    self.pr_match.query_image = msg
                    self.pr_match.match_image = img_msg

                    self.pub_pr_match.publish(self.pr_match)

                self.frame_index +=1
                self.time_last = rospy.Time.now()


###############################################################################
# ---------------------------------- Visualizer -------------------------------
###############################################################################

class Visualizer:

    def __init__(self):

        # Publishers (PR)
        self.match_gps = rospy.Publisher('/visualizer/place_recog/gps_coord',NavSatFix,queue_size=100) # Best match gps coords
        self.query_img_pr = rospy.Publisher('/visualizer/place_recog/img_query',Image,queue_size=100) # Query image
        self.match_img_pr = rospy.Publisher('/visualizer/place_recog/img_match',Image,queue_size=100) # Matching image

        # Publishers (LC)
        self.match_pose_lc = rospy.Publisher('/visualizer/loop_closure/rel_pose',PoseStamped,queue_size=100) # Best match pose
        self.query_img_lc = rospy.Publisher('/visualizer/loop_closure/img_query',Image,queue_size=100) # Query image
        self.match_img_lc = rospy.Publisher('/visualizer/loop_closure/img_match',Image,queue_size=100) # Matching image

        # Ros msgs
        self.gps_coord = NavSatFix()
        self.rel_pose = PoseStamped()
        self.img_lc_q = Image() # LC query
        self.img_lc_m = Image() # LC match
        self.img_pr_q = Image() # PR query
        self.img_pr_m = Image() # PR match

    # Publish PR match results
    def visualizePR(self,msg):

        self.gps_coord.header.stamp = msg.query_time
        self.gps_coord.latitude =  msg.match_gps.latitude
        self.gps_coord.longitude =  msg.match_gps.longitude
        self.gps_coord.altitude =  msg.match_gps.altitude
        self.img_pr_q = msg.query_image
        self.img_pr_m = msg.match_image

        self.match_gps.publish(self.gps_coord)
        self.query_img_pr.publish(self.img_pr_q)
        self.match_img_pr.publish(self.img_pr_m)

    # Publish LC macth results
    def visualizeLC(self,msg):

        self.rel_pose.header.stamp = msg.query_time
        self.rel_pose.pose = msg.rel_pose
        self.img_lc_q = msg.query_image
        self.img_lc_m = msg.match_image

        self.match_pose_lc.publish(self.rel_pose)
        self.query_img_lc.publish(self.img_lc_q)
        self.match_img_lc.publish(self.img_lc_m)

##############################################################################
# -------------------------- Data recorder class -----------------------------
##############################################################################

class DataRecorder:

    def __init__(self):

        self.lc_frame_ids = []
        self.lc_db_len = []
        self.lc_db_size = []
        self.lc_images = []
        self.lc_candidate_dis = []
        self.lc_candidate_ids = []
        self.lc_sv_ids = []
        self.lc_best_id = []
        self.lc_best_pose = []
        self.lc_t_feat = []
        self.lc_t_vlad = []
        self.lc_t_fit = []
        self.lc_t_pred = []
        self.lc_t_sv = []

        self.pr_current_ids = []
        self.pr_candidate_ids = []
        self.pr_candidate_dis = []
        self.pr_best_ids = []
        self.pr_images = []

        self.bridge = CvBridge()

        self.res_dir = rospy.get_param("res_dir_path") + datetime.now().strftime("%Y-%b-%d-%I:%M%p-[%A]") + "/" #, datetime.now()) + "/" # Results directory

        # Create a results folder
        if not os.path.exists(self.res_dir):
            try:
                os.mkdir(self.res_dir)
                os.mkdir(self.res_dir + "PR/") # Create PR sub folders
                os.mkdir(self.res_dir + "LC/") # Create LC sub folders
                print("[Data Recorder] : Results folder created !")
            except OSError as error:
                print(error)

    # Collect LC metrics
    def recordLC(self,msg):

        self.lc_frame_ids.append(msg.current_id)
        self.lc_db_len.append(msg.db_len)
        self.lc_db_size.append(msg.db_size)
        self.lc_images.append(self.bridge.imgmsg_to_cv2(msg.image,"bgr8"))
        self.lc_candidate_ids.append(msg.candidate_ids)
        self.lc_candidate_dis.append(msg.candidate_dis)
        self.lc_sv_ids.append(msg.sv_ids)
        self.lc_best_id.append(msg.best_id)
        self.lc_t_feat.append((msg.t_feat).to_sec())
        self.lc_t_vlad.append((msg.t_vlad).to_sec())
        self.lc_t_fit.append((msg.t_fit).to_sec())
        self.lc_t_pred.append((msg.t_pred).to_sec())
        self.lc_t_sv.append((msg.t_sv).to_sec())
        self.lc_best_pose.append([msg.best_pose.position.x,
                                msg.best_pose.position.y,
                                msg.best_pose.position.z,
                                msg.best_pose.orientation.x,
                                msg.best_pose.orientation.y,
                                msg.best_pose.orientation.z,
                                msg.best_pose.orientation.w]
                                )

    # Collect PR metrics
    def recordPR(self,msg):

        self.pr_current_ids.append(msg.current_id)
        self.pr_candidate_ids.append(msg.candidate_ids)
        self.pr_candidate_dis.append(msg.candidate_dis)
        self.pr_best_ids.append(msg.best_id)
        self.pr_images.append(self.bridge.imgmsg_to_cv2(msg.image,"bgr8"))

    # Save metrics
    def save(self):

        np.save(self.res_dir + "LC/" + "lc_frame_ids.npy",np.asarray(self.lc_frame_ids))
        np.save(self.res_dir + "LC/" + "lc_db_len.npy",np.asarray(self.lc_db_len))
        np.save(self.res_dir + "LC/" + "lc_db_size.npy",np.asarray(self.lc_db_size))
        np.save(self.res_dir + "LC/" + "lc_images.npy",np.asarray(self.lc_images))
        np.save(self.res_dir + "LC/" + "lc_candidate_ids.npy",np.asarray(self.lc_candidate_ids))
        np.save(self.res_dir + "LC/" + "lc_candidate_dis.npy",np.asarray(self.lc_candidate_dis))
        np.save(self.res_dir + "LC/" + "lc_sv_ids.npy",np.asarray(self.lc_sv_ids))
        np.save(self.res_dir + "LC/" + "lc_best_id.npy",np.asarray(self.lc_best_id))
        np.save(self.res_dir + "LC/" + "lc_best_pose.npy",np.asarray(self.lc_best_pose))
        np.save(self.res_dir + "LC/" + "lc_t_feat.npy",np.asarray(self.lc_t_feat))
        np.save(self.res_dir + "LC/" + "lc_t_vlad.npy",np.asarray(self.lc_t_vlad))
        np.save(self.res_dir + "LC/" + "lc_t_fit.npy",np.asarray(self.lc_t_fit))
        np.save(self.res_dir + "LC/" + "lc_t_sv.npy",np.asarray(self.lc_t_sv))
        np.save(self.res_dir + "LC/" + "lc_t_pred.npy",np.asarray(self.lc_t_pred))

        np.save(self.res_dir + "PR/" + "pr_frame_ids.npy",np.asarray(self.pr_current_ids))
        np.save(self.res_dir + "PR/" + "pr_candidate_ids.npy",np.asarray(self.pr_candidate_ids))
        np.save(self.res_dir + "PR/" + "pr_candidate_dis.npy",np.asarray(self.pr_candidate_dis))
        np.save(self.res_dir + "PR/" + "pr_best_ids.npy",np.asarray(self.pr_best_ids))
        np.save(self.res_dir + "PR/" + "pr_images.npy",np.asarray(self.pr_images))

        rospy.loginfo("[Data Recorder] : Metrics saved successfully!")

################################################################################
# -------------------------------- ROS nodes -----------------------------------
################################################################################

class Nodes:
    # --------------------------- LC main node --------------------------
    def LC():

        suppressWarnings() # Suppress unwanted warnings
        rospy.init_node('loop_close', anonymous=False) # Initializing the node
        rospy.sleep(2.0) # Wait for the parameter server to load
        LC = LoopClosure() # Loop Closure object
        limit_memory() # Limit memory growth
        rospy.loginfo("Loop closure detector activated!\n")

        LC_subs = rospy.Subscriber(LC.image_topic,CompressedImage,LC.detectLC,queue_size=LC.queue_size) # Subscriber for loop closure (CompressedImage msg)
        rospy.spin()

    # --------------------------- PR and LC main node --------------------------
    def PR():

        suppressWarnings() # Suppress unwanted warnings
        rospy.init_node('place_recog', anonymous=False) # Initializing the node
        rospy.sleep(2.0) # Wait for the parameter server to load
        PR = PlaceRecognition() # Place recognition object
        PR.load_data() # Load the reference data
        limit_memory() # Limit memory growth
        rospy.loginfo("Place recognition activated!\n")

        PR_subs = rospy.Subscriber(PR.image_topic,CompressedImage,PR.detectPR,queue_size=PR.queue_size) # Subscriber for loop closure (CompressedImage msg)
        rospy.spin()

    # --------------------------- PR and LC main node --------------------------
    def PR_n_LC():

        suppressWarnings() # Suppress unwanted warnings
        rospy.init_node('place_recog_and_loop_close', anonymous=False) # Initializing the node
        rospy.sleep(2.0) # Wait for the parameter server to load
        LC = LoopClosure() # Loop Closure object
        PR = PlaceRecognition() # Place recognition object
        PR.load_data() # Load the reference data
        limit_memory() # Limit memory growth
        rospy.loginfo("Place recognition and Loop closure detector activated!\n")

        PR_subs = rospy.Subscriber(PR.image_topic,CompressedImage,PR.detectPR,queue_size=PR.queue_size) # Subscriber for place recognition (CompressedImage msg)
        LC_subs = rospy.Subscriber(LC.image_topic,CompressedImage,LC.detectLC,queue_size=LC.queue_size) # Subscriber for loop closure (CompressedImage msg)
        rospy.spin()

    # ---------------------------- Visualizer ----------------------------------
    def Visualizer():

        rospy.init_node('visualizer', anonymous=False) # Initializing the node
        rospy.sleep(2.0)  # Wait for the parameter server to load
        vis = Visualizer() # Debugger object
        rospy.loginfo("Visualizer activated!\n")

        while not rospy.is_shutdown():
            # Subscribers
            LC_res_subs = rospy.Subscriber('/loop_closure/match/',LCmatch,vis.visualizeLC,queue_size=100) # Subscriber for loop closure
            PR_res_subs = rospy.Subscriber('/place_recog/match/',PRmatch,vis.visualizePR,queue_size=100) # Subscriber for place recognition

            rospy.spin()

    # -------------------------- Data recorder --------------------------------
    def Data_recorder():

        rospy.init_node('data_recorder', anonymous=False) # Initializing the node
        rospy.sleep(2.0)  # Wait for the parameter server to load
        rec = DataRecorder() # Debugger object
        rospy.loginfo("Data recorder activated!\n")

        while not rospy.is_shutdown():
            LC_metric_subs = rospy.Subscriber('/loop_closure/metrics/',LCmetrics,rec.recordLC,queue_size=100) # Subscriber for loop closure metrics
            PR_metric_subs = rospy.Subscriber('/place_recog/metrics/',PRmetrics,rec.recordPR,queue_size=100) # Subscriber for place recognition metrics
            rospy.spin()

        rec.save() # Save results before shutting down

    # --------------------------- Parameter server -----------------------------
    def ParameterServer(config_file):

        yaml.warnings({'YAMLLoadWarning': False})
        rospy.init_node('Parameter server', anonymous=False)
        rate = rospy.Rate(100)
        rospy.loginfo("Parameter server activated!\n")
        stream = open(config_file,'r') # Reading parameter data from file
        params = yaml.load(stream) # Reconstructing the data as a dictionary

        while not rospy.is_shutdown():
            for key,value in params.items():
                rospy.set_param(key,value) # Publishing the parameter values

            rate.sleep()
            
######################################################################################
# ------------------------ Reference data generator class -----------------------------
#######################################################################################

class DataGenerator:

    def __init__(self):

        # data generator params
        
        self.data_dir = None
        self.weights = None
        self.image_dir = None
        self.gps_file = None
        self.dataset_name = None
        self.image_type = "png"
        self.no_of_images = 10
        self.image_size = 320
        self.device = '/CPU:0' # '/CPU:0' or '/GPU:0'
        self.model = None
        self.dataset_dir = None
        
    # Initialize data generator
    def initialize(self):

        # CNN initialization
        self.model = NetVLADModel() # Create netVLAD CNN
        self.model.build(input_shape=(None,self.image_size,self.image_size,3))
        self.model.load_weights(self.weights)

        print("[Data Generator]: Initialization successful!") ## DEBUGGING ##

    # Generate reference data
    def generate_data(self):
        
        db_imgs =[]
        DB = []
        GPS = np.loadtxt(self.gps_file, delimiter=',', usecols=(0, 2))

        # Image conversion
        print("[Data Generator]: Converting images ...\n")

        for i in range(self.no_of_images):
            img_train = self.image_dir + str(i+1).zfill(3) + "." + self.image_type

            db_img = cv2.imread(img_train)
            db_img = cv2.cvtColor(db_img,cv2.COLOR_BGR2RGB)
            # db = cv2.GaussianBlur(db_img,(5,5),0)
            db = cv2.resize(db_img,(self.image_size,self.image_size))
            db = db.astype(np.float32)

            DB.append(db)

            print("[Data Generator]: " + str(i+1).zfill(3) + "." + self.image_type + " converted !")
            
            db_img = cv2.resize(db_img,(640,640))
            db_imgs.append(db_img)

        print("\n [Data Generator]: Image conversion completed !")
        
        # VLAD vectorization
        with tf.device(self.device):

            print("[Data Generator]: Generating vlad vectors...\n")

            # Database images
            db_batch = np.asarray(DB)
            db_vlad = self.model.predict(db_batch,use_multiprocessing=True)
            
            print("\n [Data Generator]: Vlad vectorization completed !")
        
        # Saving reference data
        print("[Data Generator]: Saving " + self.dataset_name + "_imgs.npy...")
        np.save(self.data_dir + self.dataset_name + '_imgs.npy',db_imgs)
        
        print("[Data Generator]: Saving " + self.dataset_name + "_vlad.npy...")
        np.save(self.data_dir + self.dataset_name + '_vlad.npy',db_vlad)
        
        print("[Data Generator]: Saving " + self.dataset_name + "_gps.npy...")
        np.save(self.data_dir + self.dataset_name + '_gps.npy',GPS)
        
        print("\n [Data Generator]: Data generation completed !")
        
        
#######################################################################################
# --------------------------------- Result analyser -----------------------------------
#######################################################################################

class Analyser:

    def __init__(self):

        # analyser params
        self.dataset_name = None       
        self.data_dir =  None
        self.res_dir = None
        self.lc_res_dir = None
        self.pr_res_dir = None
        
    # Initialize analyser
    def initialize(self):
        
        self.lc_res_dir = os.path.join(self.res_dir,self.dataset_name + "_lc_results/")
        self.pr_res_dir = os.path.join(self.res_dir,self.dataset_name + "_pr_results/")  
        
        if not os.path.exists(self.lc_res_dir):
            try:
                os.mkdir(self.lc_res_dir)
                print("[Analyser]: " + self.dataset_name + "_lc_results folder created !")
            except OSError as error:
                print(error)


        if not os.path.exists(self.pr_res_dir):
            try:
                os.mkdir(self.pr_res_dir)
                print("[Analyser]: " + self.dataset_name + "_pr_results folder created !")
            except OSError as error:
                print(error)

        print("[Analyser]: Initialization successful!\n") ## DEBUGGING ##
        
    # Analyse loop closure detection time
    def analyse_lc_time(self):
        
        db_size = np.load(self.data_dir + "lc_db_size.npy",allow_pickle=True)
        best_ids = np.load(self.data_dir + "lc_best_id.npy",allow_pickle=True)
        time_feat = np.load(self.data_dir + "lc_t_feat.npy",allow_pickle=True)
        time_vlad = np.load(self.data_dir + "lc_t_vlad.npy",allow_pickle=True)
        time_fit = np.load(self.data_dir + "lc_t_fit.npy",allow_pickle=True)
        time_pred = np.load(self.data_dir + "lc_t_pred.npy",allow_pickle=True)
        time_sv = np.load(self.data_dir + "lc_t_sv.npy",allow_pickle=True)
        
        time_lc= np.average([[time1,time2,time3,time4,time5,time1+time2+time3+time4+time5] for (time1,time2,time3,time4,time5,id) in zip(time_feat,time_vlad,time_fit,time_pred,time_sv,best_ids) if id >= 0],axis = 0)

        time_feat_avg = time_lc[0]
        time_vlad_avg = time_lc[1]
        time_fit_avg = time_lc[2]
        time_pred_avg =time_lc[3]
        time_sv_avg = time_lc[4]

        time_tot_avg = time_lc[5]
        
        print("\n[Analyser]: Analysing loop closure detection time...\n") ## DEBUGGING ##

        print(f"[Analyser]: Average feature extraction time = {time_feat_avg* 1000} ms")
        print(f"[Analyser]: Average feature vlad vectorization time = {time_vlad_avg * 1000} ms")
        print(f"[Analyser]: Average fit time = {time_fit_avg * 1000} ms")
        print(f"[Analyser]: Average prediction time = {time_pred_avg * 1000} ms")
        print(f"[Analyser]: Average SV time = {time_sv_avg * 1000} ms")
        print(f"[Analyser]: Average total execution time = {time_tot_avg * 1000} ms")
        
        # Plot of no. of database vectors vs prediction time
        
        sizes = [size for i,size in enumerate(db_size)]
        len = [i for i,size in enumerate(db_size)]
        times = [time * 1000 for i,time in enumerate(time_pred)]
        fig = plt.figure(figsize=(10,5))
        plt.rcParams.update({'font.size': 18})
        sns.regplot(x=len[206::], y=times[206::],color = "r", marker = "x",scatter_kws = {"alpha": 1,"linewidth":1},line_kws={"linestyle":"-","color":"b","alpha":1,"lw":2})
        plt.xlabel("No. of reference VLAD vectors",fontsize = 18,labelpad = 12)
        plt.ylabel("Prediction time (ms)", fontsize = 18, labelpad = 12)
        plt.ylim(0,4)
        plt.grid(linestyle='--', linewidth=0.75)
        fig.tight_layout()
        plt.savefig(f"{self.lc_res_dir}lc{self.dataset_name}_prediction_time_vs_no_of_database_vectors.png")
        # plt.show()
       
        print("\n[Analyser]: " + self.dataset_name +  "_prediction_time_vs_no_of_database_vectors.png saved !") ## DEBUGGING ##
        
        print("\n[Analyser]: Loop closure detection time analysis completed !") ## DEBUGGING ##
       
    # Save loop closure detection results
    def save_lc_results(self):
        
        print("\n[Analyser]: Saving loop closure detection results...\n") ## DEBUGGING ##
        
        frame_ids = np.load(self.data_dir + "lc_frame_ids.npy")
        images = np.load(self.data_dir + "lc_images.npy")
        best_ids = np.load(self.data_dir + "lc_best_id.npy",allow_pickle=True)
        
        matches = [[frame_id,match_id] for i,(frame_id,match_id) in enumerate(zip(frame_ids,best_ids)) if match_id >= 0]

        for sample in range(np.shape(matches)[0]):

            fig = plt.figure(figsize=(10,10))
            plt.axis('off')

            # Visualize the result
            plt.subplot(1,2,1)
            # plt.imshow(q_imgs[j])
            plt.imshow(cv2.cvtColor(images[matches[sample][0]],cv2.COLOR_RGB2BGR))
            plt.title("Current image : " + str(matches[sample][0]))
            plt.axis('off')

            plt.subplot(1,2,2)
            # plt.imshow(q_imgs[j])
            plt.imshow(cv2.cvtColor(images[matches[sample][1]],cv2.COLOR_RGB2BGR))
            plt.title("Previous image : " + str(matches[sample][1]))
            plt.axis('off')

            fig.tight_layout()
            plt.savefig(f"{self.lc_res_dir}lc{self.dataset_name}{sample+1}.png")
            
            print("[Analyser]: lc" + self.dataset_name + str(sample) + ".png saved") ## DEBUGGING ##
            
        print("\n[Analyser]: Saving results completed !") ## DEBUGGING ##

    # Save place recognition results
    def save_pr_results(self):
        
        print("\n[Analyser]: Saving place recognition results...\n") ## DEBUGGING ##
        
        frame_ids = np.load(self.data_dir + "pr_frame_ids.npy")
        q_images = np.load(self.data_dir + "pr_images.npy")
        db_images = np.load(self.data_dir + "lighthouse_dataset_imgs.npy")
        best_ids = np.load(self.data_dir + "pr_best_ids.npy",allow_pickle=True)
        
        matches = [[frame_id,match_id] for i,(frame_id,match_id) in enumerate(zip(frame_ids,best_ids)) if match_id >= 0]

        for sample in range(np.shape(matches)[0]):

            fig = plt.figure(figsize=(10,10))
            plt.axis('off')

            # Visualize the result
            plt.subplot(1,2,1)
            # plt.imshow(q_imgs[j])
            plt.imshow(cv2.cvtColor(q_images[matches[sample][0]],cv2.COLOR_RGB2BGR))
            plt.title("Current image : " + str(matches[sample][0]))
            plt.axis('off')

            plt.subplot(1,2,2)
            # plt.imshow(q_imgs[j])
            plt.imshow(cv2.cvtColor(db_images[matches[sample][1]],cv2.COLOR_RGB2BGR))
            plt.title("Matching reference image : " + str(matches[sample][1]))
            plt.axis('off')

            fig.tight_layout()
            plt.savefig(f"{self.pr_res_dir}pr{self.dataset_name}{sample+1}.png")
            
            print("[Analyser]: pr" + self.dataset_name + str(sample) + ".png saved") ## DEBUGGING ##
            
        print("\n[Analyser]: Saving results completed !") ## DEBUGGING ##

################################################################################
# ----------------------- Other helper functions -------------------------------
################################################################################

# --------------------Function to limit memory growth---------------------------
def limit_memory():

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# --------------------Function to suppress warnings ---------------------------
def suppressWarnings():

    # Suppress unwanted warnings
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category = DeprecationWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # 0 -> all , 1 -> INFO , 2 -> INFO + WARNING , 3 -> INFO + WARNING + ERROR.

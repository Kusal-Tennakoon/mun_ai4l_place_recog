# MUN_AI4L_PLACE_RECOG

## Description

mun_ai4l_place_recog is a unified system for visual place recognition(VPR) and visual loop closure detection(LCD) for robust and reliable localization for vertical Take-off and landing (VTOL) vehicles in GPS-denied environments. It can be integrated into VINS, VLOAM, or VILOAM systems and used to produce accurate trajectories by integrating VPR and LCD matches into a pose graph and solving them. The system is highly customizable. The system can operate in real-time and effectively localize in indoor, outdoor, terrestrial, and aerial environments.

A video demonstration of this system in action can be found [here](https://youtu.be/NvmjyGC5hSs).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [How to use](#how-to-use)
- [Examples](#examples)
- [Acknowledgements](#acknowledgements)
- [Contacts](#contacts)

## Features

- Provides robust and reliable localization for VTOL vehicles. 
- Data from public platforms like Google Maps or any custom source can be used as reference data for VPR.
- Minimal disk space requirements (Approximately 16 kB per reference image). 
- Offers extensive customization. Parameters such as input image sources, image resolutions, reference data, thresholds, camera intrinsic, VPR rate and LCD rate can be easily changed through an external configuration.
- Can maintain multiple configuration files for different applications and easily switch between them. 
- Effective localization in indoor, outdoor, terrestrial, and aerial environments.
- Can handle multiple types of input images (perspective/fisheye, RGB/grayscale, high/low resolutions, high/low frame rate). 
- Real-time operation (535 ms/image on average to detect a match).

## Installation

Step-by-step instructions on how to install and set up the project.

### Prerequisites

The system has the following prerequisites.
  - python3
  - pip
  - scipy
  - scikit-learn
  - faiss
  - tensorflow
  - opencv-contrib
  - pypy
  - seaborn
  - opencv3_catkin
  - Foxglove studio

Let's install the prerequisites in four simple steps.

* #### Step 1: Installing python packages

  * Method 1: Open the terminal and run the following commands.
      ```bash 
      sudo apt-get update
      sudo apt-get install python3
      python3 --version
      sudo apt-get install python3-pip
      pip3 --version
      sudo apt-get install python3-scipy
      pip install -U scikit-learn
      pip install faiss-cpu
      pip install tensorflow
      pip install opencv-contrib-python
      pip install seaborn
      sudo apt install pypy
    ```
  * Method 2: Open the terminal and run the following commands
    
      ```bash
      # Navigate to the catkin workspace folder
      cd ~/catkin_ws/src/
      
      # Clone the repository
      git clone https://github.com/Kusal-Tennakoon/mun_ai4l_place_recog.git
      
      # Navigate to the project directory
      cd mun_ai4l_place_recog
      
      # Run the requirements.sh script 
      . requirements.sh
      ```

* #### Step 2: Installing catkin packages
  * Run the following commands to install [opencv3_catkin](https://github.com/ethz-asl/opencv3_catkin.git) package

      ```bash
      # Navigate to the catkin workspace folder
      cd ~/catkin_ws/src/
      
      # Clone the opencv3_catkin repository
      git clone https://github.com/ethz-asl/opencv3_catkin.git
      
      # Navigate to the catkin_ws folder
      cd ~/catkin_ws
      
      # Run catkin_make
      catkin_make
      ```
* #### Step 3: Downloading the netvlad_weights.h5 file to _mun_ai4l_place_recog/src/data_ folder.
  * The weights file can be downloaded from [here](https://github.com/crlz182/Netvlad-Keras).

* #### Step 4: Installing FoxGlove studio (For visualization)
  * Download and install [FoxGlove studio](https://foxglove.dev/download).

## How to use

  * There are three main steps in using the package for the first time.
      1. Generating reference data for VPR
         * For VPR, reference data must be preloaded before a mission.
         * The reference data consists of three numpy arrays (.npy) comprised of _reference images_ (images corresponding to the reference locations), _reference vlad vectors_ (vlad vectors of the reference images) and _reference GPS coordinates_ (GPS coordinates of the reference locations).
         * As an example for a dataset named "example_dataset", the reference data would look as follows,
           
           * example_dataset_ref_imgs.npy - _Reference images_
           * example_dataset_ref_vlad.npy - _Reference vlad vectors_
           * example_dataset_ref_gps.npy - _Reference GPS coordinates_

      2. Executing unified VPR and LCD.
         * The system can be executed in two ways.
           * Execute and collect evaluation data -> For tuning parameters
           * Execute __without__ collecting evaluation data -> For final implementation
          
         * The ros_msg type of the input image can be of two types.
           * sensor_msgs/Image
           * sensor_msgs/CompressedImage
    
         * Thus, there are four cases under which the system can be executed
           * Case 1 => input image type = _sensor_msgs/ImageCompressed_ __AND__ collect evaluation data = _Yes_
           * Case 2 => input image type = _sensor_msgs/Image_ __AND__ collect evaluation data = _Yes_
           * Case 3 => input image type = _sensor_msgs/ImageCompressed_ __AND__ collect evaluation data = _No_
           * Case 4 => input image type = _sensor_msgs/Image_ __AND__ collect evaluation data = _No_
          
         * The system comprises a dedicated module for each use case (available at _mun_ai4l_place_recog/src/mun_ai4l_place_recog/_), as shown in the following table.

            |          input msg type         |    Collect evaluation data     |  Not collect evaluation data |
            |---------------------------------|--------------------------------|------------------------------|
            | __sensor_msgs/Image__           | mun_ai4l_place_recog_eval      | mun_ai4l_place_recog         |
            | __sensor_msgs/CompressedImage__ | mun_ai4l_place_recog_comp_eval | mun_ai4l_place_recog_comp    |

         * The use case can be changed by simply changing the module used.
           * Example: Consider the script pr_and_lc.py (available at _mun_ai4l_place_recog/src/scripts/nodes/_)
             
             * The system is currently setup for Case 4. Thus, the preamble contains the following line.
               
               ```python
               
                # from mun_ai4l_place_recog.mun_ai4l_prnlc_comp_eval import Nodes # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => True
                # from mun_ai4l_place_recog.mun_ai4l_prnlc_eval import Nodes  # If (i). input image msg type => Image. (ii). Save results for analysis => True
                # from mun_ai4l_place_recog.mun_ai4l_prnlc_comp import Nodes  # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => False
                from mun_ai4l_place_recog.mun_ai4l_prnlc import Nodes   # If (i). input image msg type => Image. (ii). Save results for analysis => False

               ```
               
             * The system can be changed from Case 4 to Case 1 by changing the preamble as follows (and the system will take care of the rest).
               ```python
               
                from mun_ai4l_place_recog.mun_ai4l_prnlc_comp_eval import Nodes # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => True
                # from mun_ai4l_place_recog.mun_ai4l_prnlc_eval import Nodes  # If (i). input image msg type => Image. (ii). Save results for analysis => True
                # from mun_ai4l_place_recog.mun_ai4l_prnlc_comp import Nodes  # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => False
                # from mun_ai4l_place_recog.mun_ai4l_prnlc import Nodes   # If (i). input image msg type => Image. (ii). Save results for analysis => False

               ```
           
      5. Analysing the results.
         * This step only applies if evaluation data was collected in step ii.
         * The following is done in this step
           * Analysis of execution time in LCD.
           * Analysis of prediction time vs number of reference vlad vectors in LCD.
           * Save VPR image results (Detected VPR matches).
           * Save LCD image results (Detected LCD matches)
          
      Let's look at each step in more detail using examples.
    
## Examples  

  ### (1). Generating reference data 
  ---
  
  * An example for reference data generation can be found at _mun_ai4l_place_recog/src/scripts/examples/reference data generation_.
  * Reference data for any VPR application can be generated by following the steps below.
    
    * __Step 1:__ Goto your mission area on Google Maps.
      
    * __Step 2:__ Select a set of reference locations.
      * The coordinates of the reference locations used for this example are the following.
        
        * Location 1 - (47.80841516,-52.78704164,18.48237984)
        * Location 2 - (47.70925861,-52.73116551,113.86800987)
        * Location 3 - (47.73575447,-52.74370857,91.1257634)
        * Location 4 - (47.73677338,-52.74050179,100.18161336)
        * Location 5 - (47.73635369,-52.73203656,80.08923787)
        * Location 6 - (47.71690957,-52.72936669,100.14711888)
          
      * In the example this step has already been done. However, if you do it all by yourself, you can copy and paste the coordinates into Google Maps to visit the locations.
          
    * __Step 3:__ Capture screenshots corresponding to each reference location and save them to the _images_ folder.
      * In the example this step has already been done. However, if you are doing it yourself, make sure to do the following before you capture the screen shots.
        * Turn on satellite view,
        * Turn off labels and
        * Have the reference location at the centre of the screen,
      
    * __Step 4:__ Rename the downloaded images as 001, 002, 003, ...
      * In the example the images are named as 001.png, 002.png, 003.png, 004.png, 005.png and 006.png.
        
    * __Step 5:__ Copy the GPS coordinate (latitude, longitude, altitude) of each reference location to the _gps_coords.txt_ file in the _images_ folder.
      * In the example this step has already been done. However, if you are doing it yourself, make sure that the GPS coordinates are saved in the same order as the images as shown below.
        
        ``` txt
        47.80841516,-52.78704164,18.48237984  ---------> GPS coordinate of 001.png 
        47.70925861,-52.73116551,113.86800987 ---------> GPS coordinate of 002.png
        47.73575447,-52.74370857,91.1257634   ---------> GPS coordinate of 003.png
        47.73677338,-52.74050179,100.18161336 ---------> GPS coordinate of 004.png
        47.73635369,-52.73203656,80.08923787  ---------> GPS coordinate of 005.png
        47.71690957,-52.72936669,100.14711888 ---------> GPS coordinate of 006.png
        ```
        
    * __Step 6:__ Modify the _generate_ref.py_ to suit your reference data.
      * You only need to modify the following four lines if you saved your reference images to the _images_ folder and GPS coordinates to the _gps_coords.txt_ file (Otherwise modify the parameters accordingly).
        
        ``` python
        generator.dataset_name = "Example_dataset" # Name for the reference dataset
        generator.no_of_images = 6 # Number of reference images
        generator.image_type = "png" # File type of reference images
        ```
      * Save your changes.
        
    * __Step 7:__ Run the _generate_ref.py_ script.
  
      ```bash
    
      # Navigate to the reference data generation folder
      cd ~/catkin_ws/src/scripts/examples/reference data generation/
    
      # Grant execute permission
      chmod +x generate_ref.py
      
      # Navigate to the catkin_ws folder
      cd ~/catkin_ws/
      
      # Execute the script
      rosrun mun_ai4l_place_recog generate_ref.py
      
      ```
      * If you did everything correctly a the _reference data generation_ folder should now have a new folder named _Example_dataset_ref_data_ with the following files in it.
  
          * Example_dataset_ref_imgs.npy
          * Example_dataset_ref_vlad.npy
          * Example_dataset_ref_gps.npy
         
    * __Step 8:__ Transfer the reference data files to the _data_ folder.
  
      * Once your reference data is ready, copy and paste (Or cut and paste. Your wish !) to the _mun_ai4l_place_recog/src/data_ folder.

  ### (2). Running the system
  --- 

  #### __1. For a public dataset__
  ---

  * Let's run the system for the public dataset AMtown03.bag (Available at https://mars.hku.hk/dataset.html).

    * __Step 1:__ Create a folder to save the results.
  
      ``` bash
      mkdir ~/catkin_ws/src/results/
      ```
          
    * __Step 2:__ Select the configuration file.
      
      * Goto _mun_ai4l_place_recog/src/scripts/nodes/_ folder. 
      * Open the _parameter_server.py_ script.
      * Modify the following line of code.
        
        ```python
        config_file = "/home/lupus/catkin_ws/src/mun_ai4l_place_recog/config/config_mars.yaml"
        
        ```
      
    * __Step 3:__ Modify the _pr.py_ and _lc.py_ scripts.
         
      * Goto _mun_ai4l_place_recog/src/scripts/nodes/_ folder. 
      * Open the _pr.py_ script.
      * Modify the following line of code.
   
      ```python
      from mun_ai4l_place_recog.mun_ai4l_prnlc_comp_eval import Nodes # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => True
      # from mun_ai4l_place_recog.mun_ai4l_prnlc_eval import Nodes  # If (i). input image msg type => Image. (ii). Save results for analysis => True
      # from mun_ai4l_place_recog.mun_ai4l_prnlc_comp import Nodes  # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => False
      # from mun_ai4l_place_recog.mun_ai4l_prnlc import Nodes   # If (i). input image msg type => Image. (ii). Save results for analysis => False
      
      ```
  
      * Save changes.
      * Repeat the above step for _lc.py_ script as well.
        
    * __Step 4:__ Execute the launch file
      
        ```bash
  
        # Navigate to the catkin_ws folder
        cd ~/catkin_ws
        
        # Run catkin_make
        catkin_make
  
        # Run the launch file
        roslaunch mun_ai4l_place_recog pr_and_lc.launch 
        
        ```
        
    * __Step 5:__ Setup the visualizer
      
      * Open FoxGlove studio.
      * Click "Open connection" and create a new connection with your localhost.
      * Click on the "Layout" dropdown menu (located at the top right of the window) and select import file.
      * Select _/mun_ai4l_place_recog/src/foxglove/mun_ai4l_dashboard_new.json_.
      * Click on the icon on the top left and select the connection your opened.
  
    * __Step 6:__ Run the rosbag
  
      ```bash
    
      # Navigate to the folder containing the rosbag
      cd <file path to the rosbag>
    
      # Play the rosbag
      rosbag play AMtown03.bag
      
      ```
      * 
         
    * __Step 7:__ Terminate the code to save the evaluation data
  
      * Once the rosbag finishes playing press Ctrl + C to terminate the code.
      * You will see the following message, once the evaluation data get saved.
     
        ```bash
        [INFO] [1717812764.937443]: [Data Recorder] : Metrics saved successfully!
        ```
  
      * If everything went correctly the _mun_ai4l_place_recog/src/results_ folder should have a new folder of the format <Date><Time>[Day] with subfolders _PR_ and _LC_.


  #### __2. For a custom dataset__
  ---

  * Let's say the custom dataset is _example.bag_ and has the follwoing specifications,

    |  Parameter | Value |
    |------------|-------|
    | Camera image topic | camera/example_topic |
    | Image size (width x height) | w x h (pixels) |

    and the intrinsic parameters of the camera used are,

    |  Parameter | Value |
    |------------|-------|
    | FOV (Horizontal) | fovx |
    | FOV (Vertical)    | fovy    |
    | Focal length (x) | fx |
    | Focal length (y)    | fy    |
    | Image center (x) | cx |
    | Image center (y) | cy |
    | Distortion coefficients | [d1,d2,d3,d4] |

  * The system can be run for this dataset by following the steps below.

    * __Step 1:__ Create a folder to save the results.
  
      ``` bash
      mkdir ~/catkin_ws/src/results/
      ```
      
    * __Step 2:__ Create a configuration file.
      
      * Goto _mun_ai4l_place_recog/config/_ folder.
      * Create a copy of one the existing files (Ex. config_lighthouse.yaml).
      * Rename the duplicate file as _config_example.yaml_.
      * Open the created configuration file and modify the following parameters.
     
        ```yaml
  
          # File path to the data directory
          # -------------------------------
             data_dir_path : /home/<___YOUR USERNAME___>/catkin_ws/src/mun_ai4l_place_recog/src/data/
             res_dir_path : /home/<___YOUR USERNAME___>/catkin_ws/src/mun_ai4l_place_recog/src/results/
        
          # Place recognition parameters
          # ----------------------------
             pr_cam_img_topic : /camera/example_topic
        
          # Loop closure parameters
          # ----------------------------
             lc_cam_img_topic : /camera/example_topic
             lc_img_width : w
             lc_img_height : h
        
          # Data arrays
          # -----------
             ref_imgs : Example_dataset_imgs.npy # Reference images for place recognition
             ref_vlad : Example_dataset_vlad.npy # Reference VLAD vectors for place recognition
             ref_gps : Example_dataset_gps.npy # Reference GPS data for place recognition
        
          # Intrinsics of the camera by which images are captured
          # -----------------------------------------------------
             cam_fov_x : fovx
             cam_fov_y : fovy
             ref_cam_fx : fx
             ref_cam_fy : fy
             ref_cam_cx : cx
             ref_cam_cy : cy
             ref_cam_dist_coeff : [d1,d2,d3,d4]
  
        ```
      * Save the changes.
          
    * __Step 3:__ Select a configuration file.
      
      * Goto _mun_ai4l_place_recog/src/scripts/nodes/_ folder. 
      * Open the _parameter_server.py_ script.
      * Modify the following line of code.
        
        ```python
        config_file = "/home/lupus/catkin_ws/src/mun_ai4l_place_recog/config/config_example.yaml"
        
        ```
      
    * __Step 4:__ Modify the _pr.py_ and _lc.py_ scripts.
      
      1. If image msg type of _example.bag_ is _sensor_msgs/Image_
         
        * Goto _mun_ai4l_place_recog/src/scripts/nodes/_ folder. 
        * Open the _pr.py_ script.
        * Modify the following line of code.
     
        ```python
        # from mun_ai4l_place_recog.mun_ai4l_prnlc_comp_eval import Nodes # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => True
        from mun_ai4l_place_recog.mun_ai4l_prnlc_eval import Nodes  # If (i). input image msg type => Image. (ii). Save results for analysis => True
        # from mun_ai4l_place_recog.mun_ai4l_prnlc_comp import Nodes  # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => False
        # from mun_ai4l_place_recog.mun_ai4l_prnlc import Nodes   # If (i). input image msg type => Image. (ii). Save results for analysis => False
        
        ```
        
        * Repeat the above step for _lc.py_ script as well.
        * Save changes.
          
      2. If image msg type of _example.bag_ is _sensor_msgs/CompressedImage_
         
        * Goto _mun_ai4l_place_recog/src/scripts/nodes/_ folder. 
        * Open the _pr.py_ script.
        * Modify the following line of code.
     
        ```python
        from mun_ai4l_place_recog.mun_ai4l_prnlc_comp_eval import Nodes # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => True
        # from mun_ai4l_place_recog.mun_ai4l_prnlc_eval import Nodes  # If (i). input image msg type => Image. (ii). Save results for analysis => True
        # from mun_ai4l_place_recog.mun_ai4l_prnlc_comp import Nodes  # If (i). input image msg type => CompressedImage. (ii). Save results for analysis => False
        # from mun_ai4l_place_recog.mun_ai4l_prnlc import Nodes   # If (i). input image msg type => Image. (ii). Save results for analysis => False
        
        ```
  
        * Save changes.
        * Repeat the above step for _lc.py_ script as well.
        
    * __Step 5:__ Execute the launch file
      
        ```bash
  
        # Navigate to the catkin_ws folder
        cd ~/catkin_ws
        
        # Run catkin_make
        catkin_make
  
        # Run the launch file
        roslaunch mun_ai4l_place_recog pr_and_lc.launch 
        
        ```
        
    * __Step 6:__ Setup the visualizer
      
      * Open FoxGlove studio.
      * Click "Open connection" and create a new connection with your localhost.
      * Click on the "Layout" dropdown menu (located at the top right of the window) and select import file.
      * Select _/mun_ai4l_place_recog/src/foxglove/mun_ai4l_dashboard_new.json_.
      * Click on the icon on the top left and select the connection your opened.
  
    * __Step 7:__ Run the rosbag
  
      ```bash
    
      # Navigate to the folder containing the rosbag
      cd <file path to the rosbag>
    
      # Play the rosbag
      rosbag play example.bag
      
      ```
      * 
         
    * __Step 8:__ Terminate the code to save the evaluation data
  
      * Once the rosbag finishes playing press Ctrl + C to terminate the code.
      * You will see the following message, once the evaluation data get saved.
     
        ```bash
        [INFO] [1717812764.937443]: [Data Recorder] : Metrics saved successfully!
        ```
  
      * If everything went correctly the _mun_ai4l_place_recog/src/results_ folder should have a new folder of the format <Date><Time>[Day] with subfolders _PR_ and _LC_.
       
  
  ### (3). Analysing evaluation results
  ---

  * __Step 1:__ Copy the contents of the folders _PR_ and _LC_ to the folder _mun_ai4l_place_recog/src/scripts/examples/result analysis/data/_ folder.
    
  * __Step 2:__ Modify the _analyse.py_ script.
    * Go to the _mun_ai4l_place_recog/src/scripts/examples/result analysis/_ folder.
    * Open _analyse.py_ script.
    * Modify the following lines of code.
   
      ``` python
      analyser.dataset_name = "Example_dataset" # Name for the test dataset
      analyser.data_dir = os.path.join(current_dir,"data/") # File path to results data
      analyser.res_dir = current_dir # File path to save the analysis results
      
      ```
      
      * Save your changes.
        
    * __Step 3:__ Run the _analyse.py_ script.
  
      ```bash
    
      # Navigate to the result analysis folder
      cd ~/catkin_ws/src/scripts/examples/result analysis/
    
      # Grant execute permission
      chmod +x analyse.py
      
      # Navigate to the catkin_ws folder
      cd ~/catkin_ws/
      
      # Execute the script
      rosrun mun_ai4l_place_recog analyse.py
      
      ```
      * If you did everything correctly a the _result analysis_ folder should now have two new folders named _Example_dataset_lc_results_ and _Example_dataset_pr_results_ containing the analysis results. with the following files in it. The terminal should display an analysis of execution time.

## Acknowledgements

I would like to extend my sincere gratitude to @crlz182 (https://github.com/crlz182) for the Keras implementation of NetVLAD, which inspired this work.

## Contacts

For any questions, suggestions, or feedback, please contact me, [Kusal Tennakoon] (kbtennakoon@mun.ca):

- **Project Lead**: [Kusal Tennakoon] ([kbtennakoon@mun.ca](mailto:kbtennakoon@mun.ca))
- **GitHub**: [Kusal-Tennakoon](https://github.com/Kusal-Tennakoon)
- **LinkedIn**: [Kusal Tennakoon](linkedin.com/in/kusal-tennakoon)

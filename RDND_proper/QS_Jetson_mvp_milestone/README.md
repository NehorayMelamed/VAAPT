# QS_Jetson
## Introduction
QS_Jetson is intended to offer a good, tunable, and efficient version of the QuickShot algorithm, with the option of using it as a 24/7 monitoring tool  
The source code should not be changed for the most part. It provides access to all of VPI's parameters through the config file. Results are saved to QS_Jetson/data/results.  
The original frames, BGS frames, and RANSAC data is all saved for analytical purposes  
## Using QS Jetson
The best way to use QS Jetson is by modifying the config file (application/config.yml). You can set parameters for Debugging, Background Subtraction, RANSAC, Harris, etc.   
Simply change the values there as you find them while being certain to keep the appropiate data type (floats should have a decimal point even if whole numbers).  
You can also easily set the program to work on local files using the 'use_rt' option (set to false). Then go to the algorithm parameters for filesystem runs and change the path to the desired RAW video file  
Please note that it only supports raw video files for now; not just a collection of frames. If you need to merge many frames, I have a script to do this in another repository (KAYA_RAW_utils)  
To view results, use the ResultsAnalysis scripts.
### After setting config.yml (!!)
$ cd QS_Jetson/   
$ export PYTHONPATH=$(pwd) # sets pythonpath
### config.yml 
Sample config.yml file  
use_rt: false # whether to use frames from the filesystem or the camera. If using the camera, the program will use parameters under 'kaya'  
algorithm: # algorithm
  num_frames: 500  # number of frames to use per batch. Usually is the number of frames taken per second  
  find_harris: false  # whether to use Harris corners. For now, this is not supported, and must be set to false  
  save_all: false # whether to save all results. If set to false, only results that pass Maor's filtering algorithm will be saved  
  num_bytes: 1 # data type of frames. So for uint16, for instance, set to 2  
  use_custom_bgs: false # whether to use Maor's BGS algorithm. If set to false, will use VPI's instead    
  bgs_n_sigma: 0.01  # n sigma for BGS    
  bgs_learn_rate: 0.01 #  learn rate for BGS  
  bgs_threshold: 40.0  # threshold for BGS   
  bgs_shadow: 127  # basically useless parameter, but one VPI exposes    
  harris_sensitivity: 0.0625  # do not use; Harris is not implemented    
  harris_strength: 20.0 # do not use; Harris is not implemented  
  harris_min_nms_distance: 8.0 # do not use; Harris is not implemented  
  ransac_min_samples: 2   
  ransac_max_iterations: 1000  # number of iterations for RANSAC to use  
  #if you want to use default for ransac residual threshold set it to -1  
  ransac_residual_threshold: 4  
  pnts_in_trj_limit: 0.05 # limit of proportionate share of trj. points  
  max_trajectory_velocity: 0.95  # for Maor's filtering algorithm  
  max_t_sus_disappear: 200  # for Maor's filtering algorithm  

kaya:  # will only be used if use_rt is true  
  use_fast: true # whether to use fast mode or slow mode for frame capturing. For now, transitioning between modes does not work  
  slow_mode:  
    height: 5460    
    width: 8192  
    multi_roi_mode: 0  
    exposure_time: 2300.0  
    fps: 13.0  
    batches_per_exposure: 2  
    max_gray_level: 255 # for autoexposure algorithm  
    max_saturated_pixels: 10  # cutoff for exposure time calculation   
    saturation_range: 50  # all pixels this value and above will be considered saturated  
    max_exposure: 40000.0  # maximum value exposure time will be set to with autoexposure  
  fast_mode:  
    offset_x: 4096  
    offset_y: 2700  
    roi_width: 1024  
    roi_height: 64  
    batches_per_exposure: 10  
    exposure_time: 1500.0  
    max_exposure: 1800.0  # maximum value exposure time will be set to with autoexposure  
    fps: 500.0    
    max_gray_level: 255  # for autoexposure algorithm  
    max_saturated_pixels: 10  # cutoff for exposure time calculation  
    saturation_range: 50  # all pixels this value and above will be considered saturated  
filesystem: # will only be used if use_rt is false  
  path: "/media/efcom/Elements/AVRAHAM/slow_mode_frames/slow_video.raw" # video file to use for filesystem runs   
  height: 5460 # needed since video files are raw  
  width: 8192  
### Results analysis  
There are 3 scripts in ResultsAnalysis:  
show_point_cloud.py - Don't use this    
show_BGS_movie.py - can also show the original frames. Simply give it the frames prefix. For example, data/results/BGS/1_ , if the video name was 1.raw  
local_PC_show.py - shows the point clouds. Simply give it the path to the data directory (QS_Jetson/data/) and the video prefix (matrix_400_ if video name was matrix_400.raw). When you run it, it allows you to pick which point cloud you would like to see (for which batch).    
### Upcoming Changes    
Will add option for semi-rt runs  


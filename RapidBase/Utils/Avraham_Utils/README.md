# KAYA_RAW_utils
utilities for dealing with RAW video and KAYA FG
# Guide
KYFGLib: the KAYA python API  
convert_small_mat_to_text: converts a SMALL (small enough it can fit into main memory multiple times) .mat file to .txt file  
convert_single_raw_to_multiple_pngs.py: converts raw video file to multiple png files  
close_FG_post_run.py: sometimes the frame grabber maintains the connection even after the process using it has ended.  
This script will close it properly.  
frame_grabber.py: wrapper I wrote for the KAYA frame grabber library  
make_video.py: converts raw frames to AVI video
seperate_raw_video.py: shows raw frames numbered name0.raw, name1.raw, etc  
show_np_movie.py: shows .npy frames numbered name0.npy, name1.npy etc.
show_seperate_batched_png_movie.py: shows batches png frames as movie. For instance, name_0_0.png is the first frame,
and name_0_1.png is the second frame, and name_1_0.png is first frame of the second batch  
ShowVideo.py: will show RAW videos in a single file  
time_close_open.py: times how fast the frame grabber can stop and start  
view_single_raw_frame.py: shows single raw frame
show_massive_raw_video.py: shows massive raw video, may be slight frame delay
write_raw_merger.py: generates bash script to merge multiple raw files into one raw file. Does not run the script    

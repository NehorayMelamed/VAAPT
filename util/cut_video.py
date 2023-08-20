from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

file_path = "/home/nehoray/Downloads/car_removed_road_from_stabilize_drone.mp4"
start_time = 200
end_time   = 3


ffmpeg_extract_subclip(file_path, start_time, end_time, targetname="/home/nehoray/Downloads/car_removed_road_from_stabilize_drone_cut.mp4")


3
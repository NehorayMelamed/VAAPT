import runpy, sys
saved_argv = sys.argv
video_path = "/SHABACK_POC_NEW/data/videos/netivay_israel/more/video_from_security_cmera_dark_mode.mp4"
reid_weights = "osnet_x0_25_market1501.pt"
classes = " 1 2 3 4 6 7 8"
script_file_name_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/track.py"
input_params = f"--source {video_path} --show-vid --reid-weights {reid_weights} --classes {classes}"

# runpy.run_path(f'{script_file_name_path}', run_name="__main__")
# sys.argv = saved_argv

# file_exe = open(script_file_name_path).read()
# final_executable = f"{file_exe} {input_params}"

# exec(f"{script_file_name_path} {input_params}")


# os.system(f"python track.py --source {video_path} --show-vid --reid-weights {reid_weights} --classes {classes}")
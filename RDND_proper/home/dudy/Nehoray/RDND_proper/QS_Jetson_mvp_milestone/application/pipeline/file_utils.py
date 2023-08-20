import os
from typing import List


def extract_video_name(video_path: str):
    last_dir = video_path.rfind("/")
    found_dir = last_dir != -1
    last_dir = max(0, last_dir)
    last_extension = video_path.rfind(".")
    return video_path[last_dir+(1*(found_dir)):last_extension]


def if_not_exist_build(directory):
    # if directory does not exist, build it
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_dirs(needed_directories: List[str]): # build non existing directories
    for directory in needed_directories:
        if_not_exist_build(directory)


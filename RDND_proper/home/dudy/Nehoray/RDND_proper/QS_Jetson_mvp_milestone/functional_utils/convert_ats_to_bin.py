import os
import numpy as np

import fnv
import fnv.file
import fnv.reduce
import os

src_directory = r"E:\Quickshot\19_7_22_flir_exp_Bin_2" ##TODO: put in correct directory
dst_dir = r"E:\Quickshot\19_7_22_flir_exp_Bin_2\Converted" ##TODO: put in correct dest directory
all_filenames = [os.path.join(src_directory, filename) for filename in os.listdir(src_directory) if filename.upper().endswith('.ATS')]

os.chdir(dst_dir)
print(os.getcwd())


def format_frame_n(frame_num):
    str_fn = str(frame_num)
    return "0"*(6-len(str_fn)) + str_fn


for filename in all_filenames:
    current_dest_dir = filename[filename.rfind(os.path.sep) + 1:(filename.find(".ats"))] + "_converted"

    try:

        print(current_dest_dir)
        os.mkdir(current_dest_dir)
    except FileExistsError:
        print(f"{current_dest_dir} dir already exists. Skipping")
        continue
    vid = fnv.file.ImagerFile(filename)
    print(f"Opened file {filename}")
    # for frame_num in range(10):#(vid.last_frame_number(0)[0]):
    for frame_num in np.arange(25000):
        vid.get_frame(frame_num)
        current_frame = vid.original
        writ = np.array(current_frame, copy=False).reshape((vid.height, vid.width))
        formatted_frame_num = format_frame_n(frame_num)
        writ.tofile(os.path.join(current_dest_dir, f"{formatted_frame_num}.Bin"))
        if frame_num % 1000 == 0:
            print(f"Got frame {frame_num}")
    # vid.close()
    print(f"Finished video {filename}")

print("Finished Converting")


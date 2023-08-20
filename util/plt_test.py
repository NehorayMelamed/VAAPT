import matplotlib.pyplot as plt
from matplotlib import image as mpimg

from RapidBase.import_all import *



base_directory = "/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/ex/crops/car/1"

all_files_list = path_get_all_filenames_from_folder(base_directory, flag_recursive=False, flag_full_filename=True)
# list_os_ = read_images_from_folder(crops_path, max_number_of_images=100, flag_return_torch=True)

for file_name in all_files_list:
    print(file_name)
    if file_name.endswith("pt"):
        continue
    img = mpimg.imread(file_name)
    numpy_to_torch()
    current_frame_torch = torch.nn.Upsample(size=(3, 256, 256), mode='bilinear')(current_frame_torch)
    plt.imshow(img)
    plt.show()

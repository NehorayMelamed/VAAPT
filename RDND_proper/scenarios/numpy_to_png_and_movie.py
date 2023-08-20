from RapidBase.Utils.MISCELENEOUS import string_rjust

from RapidBase.Utils.IO.Path_and_Reading_utils import save_image_numpy
from RapidBase.import_all import *




### right justify filenames: ###
file_path = r'/home/mafat/DataSets/DRONE_EXPERIMENTS/20180128180048-5-3ms'
filenames_list = read_image_filenames_from_folder(file_path, allowed_extentions=['.npy'])
for filename in filenames_list:
    current_number = filename.split('.npy')[0].split('/')[-1]
    new_number_string = string_rjust(current_number, 3)
    new_filename = os.path.join(file_path, new_number_string + '.npy')
    os.rename(filename, new_filename)

### Read filenames: ###
file_path = r'/home/mafat/DataSets/DRONE_EXPERIMENTS/20180128180048-5-3ms'
new_folder = os.path.join(file_path, 'images')
path_make_path_if_none_exists(new_folder)
video_name = os.path.join(file_path, 'images','video_out.avi')
max_list = []
filenames_list = read_image_filenames_from_folder(file_path, allowed_extentions=['.npy'])
H,W = np.load(filenames_list[0]).shape
video_object = cv2.VideoWriter(video_name, 0, 3, (W, H))
for frame_index, filename in enumerate(filenames_list):
    current_image = np.load(filename)
    # print(str(current_image.min()) + ',  ' + str(current_image.max()))
    max_list.append(current_image.max())
    current_image = current_image * (256/4096)
    current_image = current_image.astype(np.uint8)
    # save_image_numpy(new_folder, string_rjust(frame_index,3)+'.png', current_image, flag_convert_bgr2rgb=False, flag_scale=False)
    current_image = np.atleast_3d(current_image)
    current_image = np.concatenate([current_image, current_image, current_image],2)
    video_object.write(current_image)
    print(frame_index)

video_object.release()



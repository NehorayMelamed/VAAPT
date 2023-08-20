
from RapidBase.import_all import *
folder_path = '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/Gimballess/gimbales gray level experiment'
filenames_list = get_filenames_from_folder(folder_path)

expriment_numbers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
for experiment_number in expriment_numbers:
    # experiment_number = 19
    current_filename = filenames_list[experiment_number]

    scene_infile = open(current_filename, 'rb')
    H,W = (320,2048)
    scene_image_array = np.fromfile(scene_infile, dtype=np.uint8, count=-1)
    T = int(len(scene_image_array)/H/W)

    image = np.reshape(scene_image_array, (T,H,W))
    image_tensor = torch.tensor(image).unsqueeze(1).type(torch.float)
    image_tensor_max = image_tensor.max()
    image_tensor = image_tensor/image_tensor_max*255
    # imshow_torch_video(image_tensor, FPS=10)

    folder_path_new = os.path.join(folder_path, str(experiment_number))
    path_create_path_if_none_exists(folder_path_new)
    # video_torch_array_to_images(BW2RGB(image_tensor), folder_path_new, flag_convert_bgr2rgb=True, flag_scale_by_255=False, flag_array_to_uint8=True)
    video_torch_array_to_video(BW2RGB(image_tensor), video_name=os.path.join(folder_path_new, 'Movie.avi'), FPS=25)



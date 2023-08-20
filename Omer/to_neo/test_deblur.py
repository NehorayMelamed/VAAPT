import os
import time
import cv2
import numpy as np
import skvideo
from skvideo import io
import torch
from RVRT_deblur_inference import Deblur
# from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import numpy_unsqueeze, BW2RGB, scale_array_to_range
# from RapidBase.import_all import *
from util.crop_image_using_mouse import ImageCropper
from util.save_video_as_mp4 import save_video_as_mp4
DEVICE = 0


###### Rapid base

def scale_array_to_range(input_tensor, min_max_values_to_scale_to=(0,1)):
    input_tensor_normalized = (input_tensor - input_tensor.min()) / (input_tensor.max()-input_tensor.min() + 1e-16) * (min_max_values_to_scale_to[1]-min_max_values_to_scale_to[0]) + min_max_values_to_scale_to[0]
    return input_tensor_normalized

def BW2RGB(input_image):
    ### For Both Torch Tensors and Numpy Arrays!: ###
    # Actually... we're not restricted to RGB....
    if len(input_image.shape) == 2:
        if type(input_image) == torch.Tensor:
            RGB_image = input_image.unsqueeze(0)
            RGB_image = torch.cat([RGB_image,RGB_image,RGB_image], 0)
        elif type(input_image) == np.ndarray:
            RGB_image = np.atleast_3d(input_image)
            RGB_image = np.concatenate([RGB_image, RGB_image, RGB_image], -1)
        return RGB_image

    if len(input_image.shape) == 3:
        if type(input_image) == torch.Tensor and input_image.shape[0] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 0)
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 1:
            RGB_image = np.concatenate([input_image, input_image, input_image], -1)
        else:
            RGB_image = input_image

    elif len(input_image.shape) == 4:
        if type(input_image) == torch.Tensor and input_image.shape[1] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 1)
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 1:
            RGB_image = np.concatenate([input_image, input_image, input_image], -1)
        else:
            RGB_image = input_image

    elif len(input_image.shape) == 5:
        if type(input_image) == torch.Tensor and input_image.shape[2] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 2)
        else:
            RGB_image = input_image

    return RGB_image

def numpy_unsqueeze(input_tensor, dim=-1):
    return np.expand_dims(input_tensor, dim)

def numpy_array_to_video_ready(input_tensor):
    if len(input_tensor.shape) == 2:
        input_tensor = numpy_unsqueeze(input_tensor, -1)
    elif len(input_tensor.shape) == 3 and (input_tensor.shape[0] == 1 or input_tensor.shape[0] == 3):
        input_tensor = input_tensor.transpose([1, 2, 0])  # [C,H,W]/[1,H,W] -> [H,W,C]
    input_tensor = BW2RGB(input_tensor)

    threshold_at_which_we_say_input_needs_to_be_normalized = 2
    if input_tensor.max() < threshold_at_which_we_say_input_needs_to_be_normalized:
        scale = 255
    else:
        scale = 1

    input_tensor = (input_tensor * scale).clip(0, 255).astype(np.uint8)
    return input_tensor

def numpy_to_torch(input_image, device='cpu', flag_unsqueeze=False):
    #Assumes "Natural" RGB form to there's no BGR->RGB conversion, can also accept BW image!:
    if input_image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        input_image = np.expand_dims(input_image, axis=2) #[H,W]->[H,W,1]
    if input_image.ndim == 3:
        input_image = np.transpose(input_image, (2, 0, 1))  # [H,W,C]->[C,H,W]
    elif input_image.ndim == 4:
        input_image = np.transpose(input_image, (0, 3, 1, 2)) #[T,H,W,C] -> [T,C,H,W]
    input_image = torch.from_numpy(input_image.astype(np.float)).float().to(device) # to float32

    if flag_unsqueeze:
        input_image = input_image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
    return input_image



def video_torch_array_to_video(input_tensor, video_name='my_movie.mp4', FPS=25.0, flag_stretch=False, output_shape=None):
    ### Initialize Writter: ###
    T, C, H, W = input_tensor.shape
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Be sure to use lower case
    # video_writer = cv2.VideoWriter(video_name, fourcc, FPS, (W, H))

    ### Resize If Wanted: ###
    if output_shape is not None:
        output_tensor = torch.nn.functional.interpolate(input_tensor, size=output_shape)
    else:
        output_tensor = input_tensor

    ### Use Scikit-Video: ###
    output_numpy_array = numpy_array_to_video_ready(output_tensor.permute([0,2,3,1]).clamp(0, 255).cpu().numpy())
    if flag_stretch:
        output_numpy_array = scale_array_to_range(output_numpy_array, (0,255))
    skvideo.io.vwrite(video_name,
                      output_numpy_array,
                      outputdict={'-vcodec': 'libx264', '-b': '300000000'}) #chose the bitrate to be very high so no loss of information

    # for frame_counter in np.arange(T):
    #     current_frame = input_tensor[frame_counter]
    #     current_frame = current_frame.permute([1,2,0]).cpu().numpy()
    #     current_frame = BW2RGB(current_frame)
    #     current_frame = (current_frame * 255).clip(0,255).astype(np.uint8)
    #     video_writer.write(current_frame)
    # video_writer.release()


########

def video_read_video_to_numpy_tensor(input_video_path: str, frame_index_to_start, frame_index_to_end):
    if os.path.isfile(input_video_path) is False:
        raise FileNotFoundError("Failed to convert video to numpy array")
    print("Downloading target sub video to torch... it may take a few moments")

    # return skvideo.io.vread(input_video_path)
    ### Get video stream object: ###
    video_stream = cv2.VideoCapture(input_video_path)
    # video_stream.open()
    all_frames = []
    frame_index = 0
    while video_stream.isOpened():
        flag_frame_available, current_frame = video_stream.read()
        if frame_index < frame_index_to_start:
            frame_index += 1
            continue
        elif frame_index == frame_index_to_end:
            break

        if flag_frame_available:
            all_frames.append(current_frame)
            frame_index += 1
        else:
            break
    video_stream.release()
    # print("\n\n\n\npre stack")
    full_arr = np.stack(all_frames)
    # print("post stack")
    return full_arr


def main_deblur(video_path, use_roi=False, start_read_frame=0, end_read_frame=50, save_videos=True, blur_video_mp4="blur_video.mp4",
                deblur_video_mp4="deblur_video.mp4"):

    video_path = video_path

    ### if need to Read few frames

    numpy_video = video_read_video_to_numpy_tensor(video_path, start_read_frame, end_read_frame)

    ### If should use the crop ROI
    if use_roi is True:
        ### if need to read as a ROI crop
        ###  ROI crop image | Get relevant for ceop video ###
        first_frame = numpy_video[0]
        image_cropper = ImageCropper(first_frame)

        ### Get from user crop image
        first_image_cropped_image = image_cropper.get_tensor_crop()

        ### Set into members
        x1, y1, x2, y2 = image_cropper.get_coordinates_by_ROI_area()

        ### the new video crop acording to the ROI
        cropped_video = numpy_video[:, y1:y2, x1:x2, :]

        numpy_video = cropped_video
    ### Move numpy to torch
    torch_video = numpy_to_torch(numpy_video)

    my_deblur_obj = Deblur(input_torch_video=torch_video)
    input_torch_vid, output_torch_vid = my_deblur_obj.get_video_torch_deblur_result()


    ### Saving videos
    if save_videos is True:
        print("Save video to ", blur_video_mp4,deblur_video_mp4 )
        input_torch_vid = input_torch_vid[0].cpu()
        video_torch_array_to_video(input_torch_vid, video_name=blur_video_mp4)
        video_torch_array_to_video(output_torch_vid, video_name=deblur_video_mp4)

    #ToDo When we return it ?
    # return input_torch_vid, output_torch_vid
    return True

# dir_of_images_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/util/output_directory/scene_18_o_full_size_images"
# blur_shaback_video = "/home/dudy/Nehoray/SHABACK_POC_NEW/util/output_directory/scene_18.mp4"
# blur_shaback_video = "/home/dudy/Nehoray/SHABACK_POC_NEW/util/Downtest_ch0001_00000000323000000/scene_27.mp4"
# video_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/util/output_directory/scene_14.mp4"
# video_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/util/output_directory/scene_18.mp4"
# video_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/util/olqsutput_directory/scene_4.mp4"

if __name__ == '__main__':
    video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/video_example_from_security_camera.mp4"
    main_deblur(video_path=video_path,use_roi=True, save_videos=True)





# ### Reading and convert video ###
# video_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/TSOMET__HERTSOG_ZALMAN_SHNEOR/ch01_00000000212000000.mp4"
# numpy_video = video_read_video_to_numpy_tensor(video_path, 2, 10)
# torch_video = numpy_to_torch(numpy_video)
#
#
# #### Deblur the video ###
# my_deblur_obj = Deblur(input_torch_video=torch_video)
# input_torch_vid, output_torch_vid = my_deblur_obj.get_video_torch_deblur_result()
#
#
# ### Display the video ###
# # plt.imshow(input_torch_vid[0][0].permute(1, 2, 0).cpu())
# # plt.imshow(output_torch_vid[0].permute(1, 2, 0).cpu())
#
# imshow_torch_video(input_torch_vid, video_title="input")
# time.sleep(3)
# imshow_torch_video(output_torch_vid, video_title="output")
#
#
# ### Dudy
# # final_tensor = torch.cat([input_torch_vid.cpu().squeeze(0), output_torch_vid], -1)
# # imshow_torch_video(final_tensor, FPS=1)
# # bla = RGB2BW(output_torch_vid.cpu().squeeze(0)-output_torch_vid)
# # imshow_torch_video(bla, FPS=1)
#
# # plt.show()




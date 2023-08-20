import glob
import os
import sys

import numpy as np

sys.path.append('/home/yoav')
sys.path.append('/home/yoav/rdnd')
sys.path.append('/home/yoav/DVC/small_garden')
sys.path.append('/home/yoav/rdnd')
sys.path.append('/home/yoav/Anvil')
sys.path.append('/home/yoav/Anvil/_internal_utils')
sys.path.append('/home/yoav/Anvil/_alignments')
sys.path.append('/home/yoav/Anvil/_transforms')
sys.path.append('/home/yoav/Anvil/transforms')
sys.path.append('/home/yoav/Anvil/alignments')
import torch.nn.functional
import cv2
import time
# from object_detection_utils import *
from RapidBase.import_all import *


# todo: add seam carving functionality

# compression technique based on uniform coloring different classes
class SemanticSegmentationColorCompression(nn.Module):
    def __init__(self, model=None, num_classes=None, interesting_class_labels=[15], compute_metrics=False, device_id=0,
                 model_type='color'):
        super(SemanticSegmentationColorCompression, self).__init__()
        self.interesting_class_labels = interesting_class_labels
        self.num_classes = num_classes
        self.device = device_id
        self.comptue_metrics = compute_metrics

        self.original_frames = []
        self.transformed_frames = []
        self.video_dir = None

        self.model_type = model_type
        if self.model_type == 'color':
            self.transform_function = self._transform_frames_color
        elif self.model_type == 'blur':
            self.transform_function = self._transform_frames_blur

        if model:
            self.model = model
        else:
            self.model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)

        # override num_classes according to real model classes
        self.num_classes = 21

        self.model.eval().to(self.device)

    # todo : might want to set this classes such that they set attributes and not return them as function output and only the caller set attribute
    def _class_color_mapping(self, name='hsv'):
        """
        Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.
        """
        return plt.cm.get_cmap(name, self.num_classes)

    def _compute_metrics(self):
        return None

    def _load_video_frames(self):
        original_frames = []
        for frame in sorted(glob.glob(os.path.join(self.video_dir, '*'))):
            # grab all frames of the video
            f = torch.tensor(cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)).to(self.device).permute(2, 0, 1)
            scaled = torch.nn.functional.interpolate(f.unsqueeze(0), size=(1920 // 2, 1080 // 2))
            original_frames.append(scaled / 255.0)

        return original_frames

    def _transform_frames_blur(self, dilation_factor=16, ks=25, sigma=20.0):
        transformed_frames = []
        ratio_pixels_passed_avg = 0
        for frame in self.original_frames:
            # from RapidBase.import_all import *
            output = self.model(frame.unsqueeze(0))

            classes = output['out'][0].argmax(0)

            blur_object = torchvision.transforms.GaussianBlur(kernel_size=ks, sigma=sigma)
            # shape #interesting_classes x image size
            interesting_classes_masks = [(classes == interesting_class) for interesting_class in
                                         self.interesting_class_labels]
            # we can use sum/or since it can only be one class at a time
            any_interesting_class_mask = sum(interesting_classes_masks)

            # dilate the mask
            kernel = torch.ones((dilation_factor, dilation_factor)).to(any_interesting_class_mask.device)
            any_interesting_class_mask = any_interesting_class_mask.unsqueeze(0).unsqueeze(0)
            any_interesting_class_mask = dilation(any_interesting_class_mask, kernel=kernel)

            any_interesting_class_mask = any_interesting_class_mask.squeeze()
            num_pixles_passed = any_interesting_class_mask.sum()
            ratio_pixels_passed = num_pixles_passed / (
                    any_interesting_class_mask.shape[0] * any_interesting_class_mask.shape[1])
            ratio_pixels_passed_avg += ratio_pixels_passed

            blurred_frame = blur_object(frame)
            # take the interesting parts from the original frames
            blurred_frame[:, any_interesting_class_mask > 0] = frame[:, any_interesting_class_mask > 0]
            # append the resulting frame
            transformed_frames.append(blurred_frame)

        ratio_pixels_passed_avg /= len(transformed_frames)

        return transformed_frames, ratio_pixels_passed_avg

    def _transform_frames_color(self, class_color_mapping, dilation_factor=16):
        transformed_frames = []
        ratio_pixels_passed_avg = 0
        for frame in self.original_frames:
            # from RapidBase.import_all import *
            final_frame = torch.zeros_like(frame)  # init a mask per frame
            output = self.model(frame.unsqueeze(0))

            classes = output['out'][0].argmax(0)
            # todo: convert this and the function to torch
            color_map = class_color_mapping(classes.cpu().detach())[:, :, :3]  # extract rgb values
            # shape #interesting_classes x image size
            interesting_classes_masks = [(classes == interesting_class) for interesting_class in
                                         self.interesting_class_labels]
            # we can use sum/or since it can only be one class at a time
            any_interesting_class_mask = sum(interesting_classes_masks)
            final_frame = update_mask_color(final_frame, color_map, any_interesting_class_mask)

            # take the interesting parts from the original frames
            kernel = torch.ones((dilation_factor, dilation_factor)).to(final_frame.device)
            any_interesting_class_mask = any_interesting_class_mask.unsqueeze(0).unsqueeze(0)
            any_interesting_class_mask = dilation(any_interesting_class_mask, kernel=kernel)

            any_interesting_class_mask = any_interesting_class_mask.squeeze()
            num_pixles_passed = any_interesting_class_mask.sum()
            ratio_pixels_passed = num_pixles_passed / (
                    any_interesting_class_mask.shape[0] * any_interesting_class_mask.shape[1])
            ratio_pixels_passed_avg += ratio_pixels_passed
            final_frame[:, any_interesting_class_mask == 1] = frame[:, any_interesting_class_mask == 1]

            # append the resulting frame
            transformed_frames.append(final_frame)

        ratio_pixels_passed_avg /= len(transformed_frames)

        return transformed_frames, ratio_pixels_passed_avg

    def _delete_previous_video(self, video_dir):
        # delete images to save updated ones - I really don't care about this, might as well delete at the end as well,
        delete_current_images_command = f"rm -r {os.path.join(video_dir, '*')}"
        # execute delete command
        os.system(delete_current_images_command)

    def _save_new_video(self, output_dir):
        # save the transformed_frames into the output directory
        for id, frame in enumerate(self.transformed_frames):
            im = Image.fromarray(np.uint8(np.array((frame.permute(1, 2, 0) * 255).cpu().detach())))
            im.save(os.path.join(output_dir, f'{str(id).zfill(4)}.png'))

    def _save_concat_video(self, output_dir):
        for id, (frame_original, frame_segmented) in enumerate(zip(self.original_frames, self.transformed_frames)):
            im1 = Image.fromarray(np.uint8(np.array((frame_original.permute(1, 2, 0) * 255).cpu().detach())))
            im2 = Image.fromarray(np.uint8(np.array((frame_segmented.permute(1, 2, 0) * 255).cpu().detach())))

            im_final = Image.new('RGB', (im1.width + im2.width, im1.height))
            im_final.paste(im1, (0, 0))
            im_final.paste(im2, (im1.width, 0))
            im_final.save(os.path.join(output_dir, f'{str(id).zfill(4)}.png'))

    def _ffmpeg_concat_videos(self, video_path_1, video_path_2, output_path_dir):

        base_name = os.path.basename(video_path_1)[:2] + "concat.mp4"
        flags = "-y"  # overwrite
        concat_command = f'ffmpeg -i {video_path_1} -i {video_path_2} {flags} -filter_complex hstack=inputs=2 {os.path.join(output_path_dir, base_name)}'

        os.system(concat_command)

    def _save_videos_and_get_compression_data(self, video_dir, output_dir_frames, output_dir_video, crf_value):
        # save videos flags
        fps = 20
        flags = "-y"  # overwrite
        output_video_name = "original_video.mp4"
        output_blurred_video_name = "compressed_video.mp4"

        original_video_file_name = os.path.join(output_dir_video, str(crf_value) + '_' + output_video_name)
        transformed_video_file_name = os.path.join(output_dir_video, str(crf_value) + '_' + output_blurred_video_name)

        # save and compress original video and save to output_dir_video with name output_video_name
        command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx265 -x265-params "crf={}" {}'.format(
            fps, video_dir, flags, crf_value, original_video_file_name)
        os.system(command)

        # save and compress resulting video and save to output_dir_video with name output_blurred_video_name
        command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx265 -x265-params "crf={}" {}'.format(
            fps, output_dir_frames, flags, crf_value, transformed_video_file_name)
        os.system(command)

        self._ffmpeg_concat_videos(original_video_file_name, transformed_video_file_name, output_dir_video)

        # get file sizes of original vs adaptively blurred videos
        file_size_compressed = os.path.getsize(transformed_video_file_name)
        file_size_raw = os.path.getsize(original_video_file_name)

        # compression ratio(regardless of the h264 compression)
        compression_ratio = file_size_raw / file_size_compressed
        print(f'compressed file size(Bytes) : {file_size_compressed}')
        print(f'raw file size(Bytes) : {file_size_raw}')
        print(f'compression ratio : {compression_ratio}')

        return compression_ratio, file_size_compressed, file_size_raw

    def forward(self, video_dir="/raid/datasets/REDS/train_sharp/005",
                output_dir="/raid/yoav/DVC/frames_segmented_color_compressed",
                video_output_dir="/raid/yoav/DVC/videos_segmented_color_compressed", crf_value=30):
        if self.video_dir != video_dir or self.original_frames is None:
            print('\n\n\nreloading frames\n\n\n')
            self.video_dir = video_dir
            # only compute these in case we have new data, or if we haven't calculated it before
            self.original_frames = self._load_video_frames()
            class_color_mapping = self._class_color_mapping()
            # todo: change and use self.transform_function
            if self.model_type == 'color':
                self.transformed_frames, self.ratio_pixels_passed = self._transform_frames_color(class_color_mapping,
                                                                                                 dilation_factor=16)
            elif self.model_type == 'blur':
                self.transformed_frames, self.ratio_pixels_passed = self._transform_frames_blur(dilation_factor=16,
                                                                                                ks=25, sigma=20.0)

            self._delete_previous_video(output_dir)
            self._save_new_video(output_dir)

        if self.comptue_metrics:
            metrics = self.comptue_metrics()

        compression_ratio, file_size_compressed, file_size_raw = self._save_videos_and_get_compression_data(video_dir,
                                                                                                            output_dir,
                                                                                                            video_output_dir,
                                                                                                            crf_value)

        return round(compression_ratio, 2), \
               round(file_size_compressed, 0), \
               round(file_size_raw, 0), \
               torch.round(self.ratio_pixels_passed, decimals=2).cpu().detach()


# run_experiments()
def hakab():
    """
    perform analysis on compression technique based on uniform coloring different classes
    Returns:

    """
    device = 13
    crf = [0, 1, 2, 3, 5, 7, 10, 15, 20] + [i for i in range(30, 52)]  # 0 being lossless and 51 worst quality
    compression_ratios = []
    absolute_compression_ratios = []
    compressed_file_sizes = []
    raw_file_sizes = []
    pixels_passed = []

    ## saving locations
    model_type = 'color'
    object = SemanticSegmentationColorCompression(device_id=device, model_type=model_type)
    video_dir_output = "/raid/yoav/DVC/videos_segmented_color_compressed"
    frames_directory_output = "/raid/yoav/DVC/frames_segmented_color_compressed"
    video_dir_input = "/raid/datasets/REDS/train_sharp/005"
    csv_file_name = "/raid/yoav/DVC/crf_effect_results_color.csv"
    #
    # video_dir_output = "/raid/yoav/DVC/videos_segmented_blurred"
    # frames_directory_output = "/raid/yoav/DVC/frames_segmented_blurred"
    # video_dir_input = "/raid/datasets/REDS/train_sharp/005"
    # csv_file_name = "/raid/yoav/DVC/crf_effect_results_blurred.csv"

    for id, crf_value in enumerate(crf):
        print(f'finished {id / len(crf)}, working on crf : {crf_value}')
        compression_ratio, file_size_compressed, file_size_raw, ratio_pixels_passed = object(video_dir=video_dir_input,
                                                                                             output_dir=frames_directory_output,
                                                                                             video_output_dir=video_dir_output,
                                                                                             crf_value=crf_value)
        compression_ratios.append(compression_ratio)
        compressed_file_sizes.append(file_size_compressed)
        raw_file_sizes.append(file_size_raw)
        abs_compression_ratio = raw_file_sizes[0] / file_size_compressed  # relative to lossless compression
        absolute_compression_ratios.append(abs_compression_ratio)
        pixels_passed.append(ratio_pixels_passed)

    a = np.array(
        [crf, absolute_compression_ratios, compression_ratios, compressed_file_sizes, raw_file_sizes, pixels_passed])
    header = ",".join(['crf_value', 'absolute_compression_ratio', 'compression_ratio', 'segmented_compressed_size',
                       'original_compressed_size', 'avg_pixels_passed'])
    rows = ["{},{},{},{},{},{}".format(g, h, i, j, k, l) for g, h, i, j, k, l in a.transpose()]
    text = header + "\n" + "\n".join(rows)
    with open(csv_file_name, 'w') as f:
        f.write(text)
        f.close()


# def scale_video(video_path="/raid/yoav/DVC/masks.mp4"):
#     # todo: test if relevent
#     dir_name = os.path.dirname(video_path)
#     base_name = "masks_scaled.mp4"
#     full_name = os.path.join(dir_name, base_name)
#     scale = [1920, 1080]
#
#     scale_command = f"ffmpeg -y -i {video_path} -vf scale={scale[0]}:{scale[1]} {full_name}"
#     os.system(scale_command)


def convert_to_avi(video_path):
    """
    done for saliency encoding which only receives avi
    Args:
        video_path:

    Returns:

    """
    # command = f"ffmpeg -i {video_path} -y -c:a copy -c:v ayuv {video_path.split(os.path.extsep)[-2]}.avi"
    command = f"ffmpeg -i {video_path} -vcodec copy -acodec copy {video_path.split(os.path.extsep)[-2]}.avi"
    os.system(command)


def encode_h265_with_given_crf(images_dir, preset='medium', crf=23, file_save_dir="/raid/yoav/DVC/small_garden/videos",
                               string="", fps=100):
    flags = '-y -psnr'
    # file_save_path = os.path.join(images_dir, f"h265_{crf}.mp4")
    file_save_path = f"{file_save_dir}/h265_{crf}_{string}_{fps}.mp4"

    old_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx265 -x265-params "crf={} preset={}" {}'.format(
        fps, images_dir, flags, crf, preset, file_save_path)
    new_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v hevc -crf {} -preset {} {}'.format(
        fps, images_dir, flags, crf, preset, file_save_path)

    start = time.time()
    os.system(new_command)
    end = time.time()
    print(f"encoding time was : {end - start}")

    print(f"file size with crf={crf} is {os.path.getsize(file_save_path)}")
    return file_save_path


def encode_h264_with_given_crf(images_dir, preset='medium', crf=23, file_save_dir="/raid/yoav/DVC/small_garden/videos",
                               string="", fps=100, ext='.png'):
    flags = '-y'
    # file_save_path = os.path.join(images_dir, f"h265_{crf}.mp4")
    file_save_path = f"{file_save_dir}/h264_{crf}_{string}_{fps}.mp4"
    new_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*{}" {} -c:v h264 -crf {} -preset {} {}'.format(
        fps, images_dir, ext, flags, crf, preset, file_save_path)

    start = time.time()
    os.system(new_command)
    end = time.time()
    print(f"encoding time was : {end - start}")

    print(f"file size with crf={crf} is {os.path.getsize(file_save_path)}")
    return file_save_path


def encode_h265_with_given_qp(images_dir, preset='medium', qp=0, file_save_dir="/raid/yoav/DVC/small_garden/videos",
                              string="", fps=100):
    flags = '-y -psnr'
    # file_save_path = os.path.join(images_dir, f"h265_{crf}.mp4")
    file_save_path = f"{file_save_dir}/h265_{qp}_{string}_{fps}.mp4"

    old_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx265 -x265-params "qp={} preset={}" {}'.format(
        fps, images_dir, flags, qp, preset, file_save_path)
    new_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v hevc -qp {} -preset {} {}'.format(
        fps, images_dir, flags, qp, preset, file_save_path)

    start = time.time()
    os.system(new_command)
    end = time.time()
    print(f"encoding time was : {end - start}")

    print(f"file size with QP={qp} is {os.path.getsize(file_save_path)}")


def encode_h264_with_given_qp(images_dir, preset='medium', qp=0, file_save_dir="/raid/yoav/DVC/small_garden/videos",
                              string="", fps=100):
    flags = '-y'
    # file_save_path = os.path.join(images_dir, f"h265_{crf}.mp4")
    file_save_path = f"{file_save_dir}/h264_{qp}_{string}_{fps}.mp4"
    old_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx264 -x264-params "qp={} preset={}" {}'.format(
        fps, images_dir, flags, qp, preset, file_save_path)
    new_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v h264 -qp {} -preset {} {}'.format(
        fps, images_dir, flags, qp, preset, file_save_path)

    start = time.time()
    os.system(new_command)
    end = time.time()
    print(f"encoding time was : {end - start}")

    print(f"file size with QP={qp} is {os.path.getsize(file_save_path)}")
    return file_save_path


def encode_h264_skip_mode(images_dir, preset='medium', crf=0, file_save_dir="/raid/yoav/DVC/small_garden/videos",
                          string="", fps=100):
    # todo: figure out how to control skip mode in encoder
    flags = '-y'
    # file_save_path = os.path.join(images_dir, f"h265_{crf}.mp4")
    # static_thresh = 20000  # from 0 to max int
    for static_thresh in [0, 100, 5000, 25000]:
        file_save_path = f"{file_save_dir}/h264_{crf}_{string}_{fps}_{static_thresh}.mp4"
        skip_mode_encoding = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v h264 -crf {} -preset {} -static-thresh {} {}'.format(
            fps, images_dir, flags, crf, preset, static_thresh, file_save_path)

        start = time.time()
        os.system(skip_mode_encoding)
        end = time.time()
        print(f"encoding time was : {end - start}")

        print(f"file size with CRF={crf} is {os.path.getsize(file_save_path)}")
    return file_save_path


def create_static_video(image_path="/raid/yoav/DVC/small_garden/Drone_0_movie/0000.png",
                        frames_save_path="/raid/yoav/DVC/small_garden/temp_1", N=100):
    """
    encode static video of N frames(done to see how efficient the h264/5 are)
    Args:
        image_path: image
        frames_save_path: save N images to this folder
        N: frame count
    Returns:
    """
    image = torch.tensor(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    for i in range(N):
        print(i / N)
        save_image_torch(frames_save_path, f'static_image_{str(i).zfill(4)}.png', image)


def encode_static_video_and_static_video_with_moving_block():
    file = "/raid/datasets/SonyAlpha7_new/Augmented/C0025/0_1_0/00001.png"
    save_path_1 = "/raid/yoav/DVC/small_garden/temp_1" # save frames / intermideate res
    save_path_2 = "/raid/yoav/DVC/small_garden/temp_2" # save results
    create_static_video(image_path=file, frames_save_path=save_path_1, N=1380)
    image = torch.tensor(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)

    save_image_torch(save_path_2, "image.jpg", image)
    save_image_torch(save_path_2, "image.png", image)
    for crf in [25]:
        encode_h264_with_given_crf(save_path_1, crf=crf, file_save_dir=save_path_2, string="static")

    if False:
        add_block_to_constant_video(save_path_1)
        for crf in [25]:
            encode_h264_with_given_crf(save_path_1, crf=crf, file_save_dir=save_path_2, string="motion")


def add_block_to_constant_video(images_path='/raid/yoav/DVC/small_garden/temp_1'):
    """
    add some black square moving diagonally along the video
    Args:
        images_path:

    Returns:

    """
    location_0_x = 0
    location_0_y = 0
    size = [200, 200]
    for i, file in enumerate(sorted(glob.glob(os.path.join(images_path, '*.png')))):
        image = torch.tensor(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        image[:, location_0_x:location_0_x + size[0], location_0_y:location_0_y + size[1]] = 0
        save_image_torch(images_path, f'static_image_{str(i).zfill(4)}.png', image)
        location_0_x += 5
        location_0_y += 5


def extract_frames_from_mp4(video_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    command = f"ffmpeg -i  {video_path} -y {os.path.join(save_path, 'out_%05d.png')}"
    os.system(command)

# extract_frames_from_mp4("/raid/datasets/SOI_land/soi_land/soi_videos/360meter.mp4", "/raid/datasets/SOI_land/soi_land/soi_images/360meters")


def merge_frames_according_to_mask(frames_dir_1, frames_dir_2, mask_dir,
                                   save_path="/raid/yoav/DVC/small_garden/expNcomp/merge_videos_exp"):
    """
    merge 2 videos, where mask=1 take video 1 and when mask=0 take dir 2
    Args:
        frames_dir_1: video 1
        frames_dir_2: video 2
        mask_dir: masks

    Returns:
    """
    # output_frames = []
    frames_1 = sorted(glob(os.path.join(frames_dir_1, '*.png')))
    frames_2 = sorted(glob(os.path.join(frames_dir_2, '*.png')))
    masks = sorted(glob(os.path.join(mask_dir, '*.png')))
    for i, (frame1, frame2, mask) in enumerate(zip(frames_1, frames_2, masks)):
        frame1 = torch.tensor(cv2.cvtColor(cv2.imread(frame1), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        frame2 = torch.tensor(cv2.cvtColor(cv2.imread(frame2), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        mask = torch.tensor(cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)

        frame_merged = torch.zeros_like(frame1)
        frame_merged[mask == 0] = frame2[mask == 0]
        frame_merged[mask == 255] = frame1[mask == 255]
        # output_frames.append(frame_merged)
        save_image_torch(save_path, f'merged_frame_{str(i).zfill(4)}.png', frame_merged)


def predict_yolo_on_frame(frame, interesting_class_names=['person', 'car']):

    # frame = np.array(frame.detach().cpu())
    # handle case where image is not cv image
    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392
    weights_path = "/raid/yoav/DVC/yolov3.weights"
    class_file_path = "/raid/yoav/DVC/yolov3.txt"
    cfg_file = "/raid/yoav/DVC/yolov3.cfg"
    # read class names from text file
    classes = None
    with open(class_file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    interesting_class_labels = np.argwhere([name in interesting_class_names for name in classes])[:, 0]
    # generate different colors for different classes
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    # read pre-trained model and config file
    net = cv2.dnn.readNet(weights_path, cfg_file)
    # create input blob
    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)

    # function to get the output layer names
    # in the architecture
    def get_output_layers(net):
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    # for each detection from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # remove non interesting classes:
    interesting_class_labels = [0, 2]
    remove_list = []
    for i, label in enumerate(class_ids):
        if not (label in interesting_class_labels):
            remove_list.append(i)
    for i in sorted(remove_list, reverse=True): # otherwise pop will mess up the order
        class_ids.pop(i)
        confidences.pop(i)
        boxes.pop(i)

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # go through the detections remaining
    # after nms and draw bounding box

    mask = np.zeros_like(frame)
    mask = update_mask_given_bbox_coords_np_yolo(mask, [boxes[idx] for idx in indices])

    # for i in indices:
    #     try:
    #         box = boxes[i]
    #     except:
    #         i = i[0]
    #         box = boxes[i]
    #
    #     x = box[0]
    #     y = box[1]
    #     w = box[2]
    #     h = box[3]
    #     draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    #
    # # display output image
    # plt.imshow(frame)
    # plt.show()
    return mask


def double_compression_method(images_dir, preset='medium', temp_save_dir="/raid/yoav/DVC/small_garden/videos",
                              fps=25, encoder='h264', metric='crf', gammas=None,
                              final_save_dir="/raid/yoav/DVC/small_garden/videos", mask_dir=None, extra_str=""):
    """
    A method we will call double compression.
    We basically want a way to take a video and control the QP per macro block ourselves.
    This can be done the following way:
    * compress once in the highest quality possible - gamma 1 = 0
    * compress second time in the lowest quality that you are willing to tolerate(the unsegmented/not-important part) - gamma 2
    * merge both videos based on segmentation mask
    * compress result based on the lowest quality that you are willing to tolerate(the segmented/important part) - gamma 3

    Args:
        images_dir:
        preset:
        file_save_dir:
        fps:
        encoder:
        metric:
        gammas: [gamma1. gamma2. gamma3]

    Returns:

    """

    if str.upper(encoder) == 'H264':
        if str.lower(metric) == 'crf':
            encoding_function = encode_h264_with_given_crf
        elif str.lower(metric) == 'qp':
            encoding_function = encode_h264_with_given_qp
        else:
            raise RuntimeError('only valid metrics are qp and crf')
    elif str.upper(encoder) == 'H265':
        if str.lower(metric) == 'crf':
            encoding_function = encode_h265_with_given_crf
        elif str.lower(metric) == 'qp':
            encoding_function = encode_h265_with_given_qp
        else:
            raise RuntimeError('only valid metrics are qp and crf')
    else:
        raise RuntimeError('encoder not supported, H264 or H265 only')

    gamma_1 = gammas[0] if gammas else 0  # might be called gamma
    gamma_2 = gammas[1] if gammas else 51  # might be called beta
    gamma_3 = gammas[2] if gammas else 25  # might be called alpha
    # encode with different crf before merging
    file_path_high_quality = encoding_function(images_dir, preset, gamma_1, file_save_dir=temp_save_dir, string=metric, fps=fps)
    file_path_low_quality = encoding_function(images_dir, preset, gamma_2, file_save_dir=temp_save_dir, string=metric, fps=fps)
    # extract frames in order to merge
    temp_save_folder1 = "/raid/yoav/DVC/small_garden/temp"
    if os.path.exists(temp_save_folder1):
        shutil.rmtree(temp_save_folder1)
    os.makedirs(temp_save_folder1, exist_ok=True)
    temp_save_folder2 = "/raid/yoav/DVC/small_garden/temp_1"
    if os.path.exists(temp_save_folder2):
        shutil.rmtree(temp_save_folder2)
    os.makedirs(temp_save_folder2, exist_ok=True)
    extract_frames_from_mp4(file_path_high_quality, temp_save_folder1)
    extract_frames_from_mp4(file_path_low_quality, temp_save_folder2)

    # extract segmentation mask
    if mask_dir is None: # create a mask dir and save to it
        mask_dir = "/raid/yoav/DVC/small_garden/temp_masking"
        if os.path.exists(mask_dir):
            shutil.rmtree(mask_dir)
        os.makedirs(mask_dir, exist_ok=True)
        # save_semantic_segmentation_masks(images_dir, mask_dir)
        # save_yolo_semantic_segmentation_masks(images_dir, mask_dir, interesting_class_names=['person', 'car'])
        # save_outlier_masks(images_dir, mask_dir)

    save_path_merged = "/raid/yoav/DVC/small_garden/expNcomp/merge_videos_exp"
    if os.path.exists(save_path_merged):
        shutil.rmtree(save_path_merged)
    os.makedirs(save_path_merged, exist_ok=True)
    # merge frames
    merge_frames_according_to_mask(temp_save_folder1, temp_save_folder2, mask_dir=mask_dir, save_path=save_path_merged)
    # encode merged frames

    merged_video_path = encoding_function(save_path_merged, 'medium', gamma_3, file_save_dir=final_save_dir,
                                          string=f'{extra_str}_merged_video_{gamma_1}_{gamma_2}_{gamma_3}', fps=fps)

    _ = encoding_function(images_dir, 'medium', gamma_3, file_save_dir=final_save_dir,
                          string=f'{extra_str}_original_video_{gamma_3}', fps=fps)
    return merged_video_path


def double_compression_method_on_images(image_path="/raid/datasets/drones/T/all/0001/00000000.png",
                                        temp_save_dir="/raid/yoav/DVC/small_garden/temp",
                                        q_values=[0, 100, 100],
                                        mask=None,
                                        id=""):
    # load image and save different quality images
    im = Image.open(image_path)
    path_1 = os.path.join(temp_save_dir, f"{id}_low_quality_{q_values[0]}.jpg")
    path_2 = os.path.join(temp_save_dir, f"{id}_high_quality_{q_values[1]}.jpg")
    im.save(path_1, "JPEG", quality=q_values[0])
    im.save(path_2, "JPEG", quality=q_values[1])
    # read images back - todo: check if i can skip the saving part
    im_q1 = read_image_torch(path_1)
    im_q2 = read_image_torch(path_2)
    # initialize a top left square mask if none exists
    if mask is None:
        t, c, h, w = im_q1.shape
        mask = torch.ones_like(im_q1)
        mask[:, :, :h//2, :w//2] = 0
    # transform mask to bool
    mask = mask.to(torch.bool)
    # combine images based on mask, True values in mask correspond to high quality(im_q2) and False to low quality(im_q1
    final_image = im_q1.clone()
    final_image[mask] = im_q2[mask]
    path_final = os.path.join(temp_save_dir, f"{id}_final_{q_values[2]}.jpg")
    torchvision.transforms.ToPILImage()(final_image.squeeze()/255).save(path_final, "JPEG", quality=q_values[2])
    return final_image

# mask_count = 10
# # masks = torch.zeros((mask_count, 640, 512))
# for id in range(mask_count):
#     mask = torch.zeros((1, 3, 512, 640))
#     mask[:, :, :id*50, :id*50] = 1
#     final_image = double_compression_method_on_images(mask=mask, id=id)

def compute_metrics_between_image_folders(compressed_video, original_video):
    metrics = {}
    compressed_images = sorted(glob.glob(os.path.join(compressed_video, "*.png")))
    reference_images = sorted(glob.glob(os.path.join(original_video, "*.png")))
    psnr_total = 0
    fid_total = 0  # todo: implement FID score

    for id, (im_ref, im_comp) in enumerate(zip(compressed_images, reference_images)):
        ref_image = cv2.imread(im_ref)
        compressed_image = cv2.imread(im_comp)
        psnr_total += cv2.PSNR(ref_image, compressed_image)

    metrics['avg_psnr'] = psnr_total / (id + 1)
    metrics['FID'] = fid_total
    return metrics


def post_encoding_meta_data(original_video="/raid/yoav/DVC/small_garden/Drone_stable_moving_100fps",
                            videos_dir="/raid/yoav/DVC/small_garden/expNcomp",
                            log_file="/raid/yoav/DVC/small_garden/expNcomp/description.txt"):
    """
    for each video in the videos_dir, compute PSNR, FID, and other metrics with regards to original_video_frames
    Args:
        original_video: original video frames
        videos_dir: processed videos to compare
        log_file:

    Returns:

    """
    temp_save_frames_path = "/raid/yoav/DVC/small_garden/temp"  # save intermediate data
    for video_path in sorted(glob.glob(os.path.join(videos_dir, '*.mp4'))):
        # extract frames to temp folder
        extract_video_frames_command = f"ffmpeg -i {video_path} {os.path.join(temp_save_frames_path, 'out%03d.png')}"
        os.system(extract_video_frames_command)
        metrics = compute_metrics_between_image_folders(temp_save_frames_path, original_video)
        with open(log_file, 'a') as f:
            f.write(f"{os.path.basename(video_path)} \n")  # write video name
            for key, value in metrics.items():  # write metrics
                f.write(f"{key} : {value} \n")
                f.write('\n')
        f.close()

    # additional manual step to compare DVC decoded video as well
    metrics = compute_metrics_between_image_folders("/raid/yoav/DVC/frames_after_dvc_decoding/decoded_moving_drone",
                                                    original_video)
    with open(log_file, 'a') as f:
        f.write(f"DVC \n")
        for key, value in metrics.items():
            f.write(f"{key} : {value} \n")
            f.write('\n')
    f.close()


# def temp():
#     for crf in [0, 10, 20, 30, 40, 50]:
#         command_h265 = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx265 -x265-params "crf={}" {}'.format(
#             50, "/raid/yoav/DVC/small_garden/Drone_0_movie", "-y -preset fast", crf,
#             f"/raid/yoav/DVC/small_garden/videos/h265_{crf}.mkv")
#         os.system(command_h265)
#         command_h264 = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx264 -x264-params "crf={}" {}'.format(
#             50, "/raid/yoav/DVC/small_garden/Drone_0_movie", "-y -preset fast", crf,
#             f"/raid/yoav/DVC/small_garden/videos/h264_{crf}.mkv")
#         os.system(command_h264)

def encode_with_kvazaar(images_dir, crf=23, string=""):
    # could cinfigure ffmpeg with kvazaar and make it easier then installing independently
    # todo: configure with kvazaar, unknown encoder libkvazaar error
    # NOT WORKING
    fps = 50
    flags = '-y'
    # file_save_path = os.path.join(images_dir, f"h265_{crf}.mp4")
    file_save_path = f"/raid/yoav/DVC/small_garden/videos/h265_{crf}_{string}.mp4"
    command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libkvazaar -x264-params "crf={}" {}'.format(
        fps, images_dir, flags, crf, file_save_path)
    os.system(command)

    print(f"file size with crf={crf} is {os.path.getsize(file_save_path)}")


def run_exp_stab():
    """
    compare same params on 2 different movies
    Returns:
    """
    stab_movie = "/raid/yoav/DVC/small_garden/Drone_0_movie"
    save_path = "/raid/yoav/DVC/small_garden/expNcomp/lose_drone_exp"
    crfs = [i for i in range(0, 15)] + [i for i in range(40, 51)]
    for crf in crfs:
        encode_h265_with_given_crf(stab_movie, crf=crf, string="stab", file_save_dir=save_path)
        encode_h264_with_given_crf(stab_movie, crf=crf, string="stab", file_save_dir=save_path)
        print(f"Done with crf {crf}")


def run_exp_orig_stab():
    """
    compare same params on 2 different movies
    Returns:
    """
    orig_movie = "/raid/yoav/DVC/small_garden/Drone_0_movie_unstable"
    stab_movie = "/raid/yoav/DVC/small_garden/Drone_0_movie"
    save_path = "/raid/yoav/DVC/small_garden/expNcomp/stab_vs_orig_exp"
    crfs = [i*5 for i in range(0, 10)]
    for crf in crfs:
        # encode_h265_with_given_crf(orig_movie, crf, "orig")
        encode_h264_with_given_crf(orig_movie, crf=crf, string="orig", file_save_dir=save_path)
        encode_h264_with_given_crf(stab_movie, crf=crf, string="stab", file_save_dir=save_path)
        print(f"Done with crf {crf}")


def save_scaled_video(video_dir="/raid/datasets/REDS/train_sharp/005", output_dir="/raid/yoav/DVC/frames_scaled",
                      output_video_name="original_video.avi"):
    """
    scale a video to (1080, 1920): maybe we should do it generically but i only used this function to scale to
    saliency_coding which only accepts this scale apparently.
    Args:
        video_dir: images of video
        output_dir: directory in which to save output video(scale of 1080, 1920) and scaled frames

    """
    device = 13

    for id, frame in enumerate(sorted(glob.glob(os.path.join(video_dir, '*')))):
        # grab all frames of the video
        print(id)
        f = torch.tensor(cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1)
        scaled = torch.nn.functional.interpolate(f.unsqueeze(0), size=(1080, 1920))

        im = Image.fromarray(np.uint8(np.array((scaled.squeeze().permute(1, 2, 0)).cpu().detach())))
        im.save(os.path.join(output_dir, f'{str(id).zfill(4)}.png'))

    # save videos flags
    fps = 20
    flags = "-y"  # overwrite
    original_video_file_name = os.path.join(output_dir, output_video_name)
    # save and compress original video and save to output_dir_video with name output_video_name
    command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx265 -x265-params "crf={}" {}'.format(
        fps, output_dir, flags, 0, original_video_file_name)
    os.system(command)


def get_meta_data_on_segmentation_maps(masks_path):
    pass


def convert_segmentation_map_to_text_file(map, file_path_dir=None, file_name=None):
    """
    a function to take a map of QP values and convert them into text raster format to input kvazaar.
    the format will be :
    H W
    v_(1) ... v_(HxW)
    the map isn't necessarily the shape of the final video frame,
    the division is done proportionally in kvazaar.
    Args:
        map: delta QP values of shape H,W

    Returns: saves a .txt file at file_path_dir/file_name

    """
    # pre process map
    H, W = map.shape
    assert (file_path_dir and file_name), "should provide saving location parameters"
    path_to_file = os.path.join(file_path_dir, file_name)
    with open(path_to_file, 'w') as f:
        f.write(f"{H}, {W}\n")
        for value in map.flatten():
            f.write(f"{int(value)} ")


def kvazaar_encoding_h265(video_path):
    """
    kvazaar didnt work so well when tested...
    Args:
        video_path:

    Returns:

    """
    pass


def save_masked_video(masks_path="/raid/yoav/DVC/masks", dir="/raid/yoav/DVC/frames_scaled",
                      mask_video_name="masks.avi"):
    """
    compute segmentation masks per frame and save masked frames video into the masks path
    Args:
        masks_path: path to SAVE masks
        dir: path to SAVED frames
        mask_video_name: name for final video

    Returns:

    """
    device = 13
    # init segmentation model
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=True).to(device)
    human_class = 15
    interesting_class_labels = [human_class]
    mean_count_interesting_pixels, min_count_interesting_pixels, max_count_interesting_pixels = 0, 1, 0
    for id, frame in enumerate(sorted(glob.glob(os.path.join(dir, '*.png')))):
        # grab all frames of the video
        print(id)
        frame = torch.tensor(cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB), dtype=torch.float32).to(
            device).permute(2, 0, 1).to(device)
        with torch.no_grad():
            output = model(frame.unsqueeze(0))

        classes = output['out'][0].argmax(0)
        # todo: convert this and the function to torch
        # shape (#interesting_classes, image size)
        # boolean mask for interesting classes
        interesting_classes_masks = [(classes == interesting_class) for interesting_class in interesting_class_labels]
        # we can use sum/or since it can only be one class at a time
        any_interesting_class_mask = sum(interesting_classes_masks)
        current_count = any_interesting_class_mask.sum() / (
                    any_interesting_class_mask.shape[0] * any_interesting_class_mask.shape[1])
        mean_count_interesting_pixels += current_count
        max_count_interesting_pixels = max(max_count_interesting_pixels, current_count)
        min_count_interesting_pixels = min(min_count_interesting_pixels, current_count)

        # dilate the mask a bit to capture the surrounding and anything we have missed
        dilation_factor = 16
        kernel = torch.ones((dilation_factor, dilation_factor)).to(device)
        any_interesting_class_mask = any_interesting_class_mask.unsqueeze(0).unsqueeze(0)
        any_interesting_class_mask = dilation(any_interesting_class_mask, kernel=kernel)

        im = Image.fromarray(np.uint8(np.array((255 * any_interesting_class_mask.squeeze()).cpu().detach())))
        im.save(os.path.join(masks_path, f'{str(id).zfill(4)}.png'))

    # save videos flags
    fps = 20
    flags = "-y"  # overwrite
    original_video_file_name = os.path.join(masks_path, mask_video_name)
    # save and compress original video and save to output_dir_video with name output_video_name
    command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx265 -x265-params "crf={}" {}'.format(
        fps, masks_path, flags, 0, original_video_file_name)
    os.system(command)

    return min_count_interesting_pixels, max_count_interesting_pixels, mean_count_interesting_pixels / (id + 1)


def get_saliency_video_pipeline(video_dir="/raid/datasets/REDS/train_sharp/005",
                                scaled_frames_dir="/raid/yoav/DVC/frames_scaled", mask_path="/raid/yoav/DVC/masks",
                                result_save_path="/raid/yoav/DVC/saliency_video.mkv"):
    """

    Args:
        video_dir: video to process
        scaled_frames_dir: directory to store scaled frames(specific scale for saliency coding)
        mask_path: directory to store masks after segmentation
        result_save_path: save all the results

    Returns:

    """
    output_video_name = "original_video.mp4"
    mask_video_name = "masks.mp4"
    video_path = os.path.join(scaled_frames_dir, output_video_name)
    masks_video_path = os.path.join(mask_path, mask_video_name)
    # create the scaled and masked folders if they don't exist(Note that the video should match in that case)
    if not os.path.isfile(video_path) and not os.path.isfile(masks_video_path):
        save_scaled_video(video_dir=video_dir, output_dir=scaled_frames_dir, output_video_name=output_video_name)
        save_masked_video(masks_path=mask_path, dir=scaled_frames_dir, mask_video_name=mask_video_name)

        # scale_video(video_path)
        # scale_video(masks_video_path)

        convert_to_avi(video_path)
        convert_to_avi(masks_video_path)

    # encode the scaled video into defualt path with various crf's
    for crf in [36]:
        encode_h265_with_given_crf(scaled_frames_dir, crf)

    # perform the saliency coding - not on remote, perform locally
    # activate_saliency_coding(video_path,
    #                          masks_video_path,
    #                          result_save_path,
    #                          bitrate=1000, saliency_s0=60, saliency_bitrate=80)


def perform_comparison_on_parameters(video_dir="/raid/yoav/DVC/small_garden/Drone_stable_moving_100fps",
                           save_path="/raid/yoav/DVC/small_garden/expNcomp/lose_drone_exp", crfs=None, fpss=None):
    """
    function to perform a comparison(between parameters - single video) on the stabilized drone movie(by default, you can change it) with fps of 100:
    Compare H265, H264(maybe more like DVC). currently with no masking at all.
    todo: Save data regarding size, quality(PSNR), Throughput(MBPS), Latency(msec).
    Args:
        video_dir: folder of images of video
        save_path: dir to save to
    Returns:
    """
    crf_values = crfs if crfs else [23]
    fps_values = fpss if fpss else [100]
    for fps in fps_values:
        for crf in crf_values:
            encode_h265_with_given_crf(video_dir, preset="medium", crf=crf, file_save_dir=save_path, fps=fps)
            encode_h264_with_given_crf(video_dir, preset="medium", crf=crf, file_save_dir=save_path, fps=fps)

def psnr_between_videos(input_video, refrence_video):
    command = f"ffmpeg -i {input_video} -i {refrence_video} -filter_complex 'psnr' -f null /dev/null"
    os.system(command)


def encode_mp4_file_with_h264(video_path, crf, save_path):
    command = f"ffmpeg -i '{video_path}' -c:v h264 -crf {crf} {save_path}"
    os.system(command)


def effect_of_noise_on_compression(test_video_folder="/raid/yoav/DVC/small_garden/Drone_stable_moving_100fps"):
    pass


"""
Implementation of 3 Tests for the review:
for aerial videos: 
1) simply encode with different crfs and see when we lose the drone, set that crf + margin as target

for double_compression evaluation:
2) test that when gamma2=gamma3 we get roughly the same video(quality+compression) as simply encoding with gamma2

the general case:
3) when gamma1=0(the default case we would want, no reason to change that) compare the difference between compressing 
in this method with some gamma2 and gamma3 against single compression with gamma3.
this will test how the background compression by gamma2 effectively helped compress the data.

"""
def review_function_2(images_dir="/raid/datasets/REDS/train_sharp/005", save_path="/raid/yoav/DVC/small_garden/expNcomp/is_same_size_exp"):
    gammas = [10, 15, 20, 25, 30]
    for equal_gamma in gammas:
        double_compression_method(images_dir, encoder='H264', gammas=[0, equal_gamma, equal_gamma], final_save_dir=save_path)
        double_compression_method(images_dir, encoder='H265', gammas=[0, equal_gamma, equal_gamma], final_save_dir=save_path)


def review_function_3(images_dir="/raid/datasets/REDS/train_sharp/005", save_path="/raid/yoav/DVC/small_garden/expNcomp/extra_effect_exp"):
    gammas_set = [[0, 51, 25],
                  [0, 40, 25],
                  [0, 51, 30],
                  [0, 40, 30]]
    for gammas in gammas_set:
        double_compression_method(images_dir, encoder='H264', gammas=gammas, final_save_dir=save_path)
        double_compression_method(images_dir, encoder='H265', gammas=gammas, final_save_dir=save_path)


if __name__ == "__main__":
    # double_compression_method("/raid/datasets/REDS/train_sharp/005", fps=25, encoder='H264', metric='qp')
    # double_compression_method("/raid/datasets/REDS/train_sharp/005", fps=25, encoder='H265', metric='qp')

    # double_compression_method("/raid/datasets/REDS/train_sharp/005", fps=25, encoder='H264', metric='crf')
    # double_compression_method("/raid/datasets/REDS/train_sharp/005", fps=25, encoder='H265', metric='crf')
    # run_exp_orig_stab()
    # run_exp_stab()
    # review_function_2()
    # review_function_3()

    # crf = 21
    # encode_mp4_file_with_h264("/home/yoav/DVC/small_garden/speedsize/sent_data/speed_size_video.mp4", crf,
    #                           f"/home/yoav/DVC/small_garden/speedsize/sent_data/my_encoding_crf{crf}.mp4")
    # psnr_between_videos("/home/yoav/DVC/small_garden/speedsize/sent_data/speed_size_video.mp4", "/home/yoav/DVC/small_garden/speedsize/sent_data/original_video.MP4")
    # psnr_between_videos(f"/home/yoav/DVC/small_garden/speedsize/sent_data/my_encoding_crf{crf}.mp4", "/home/yoav/DVC/small_garden/speedsize/sent_data/original_video.MP4")
    # double_compression_method_on_images()
    # pass

    # image = cv2.imread("/raid/datasets/SonyAlpha7_new/Augmented/C0014/0_1_0/00001.png")
    # predict_yolo_on_frame(image)

    videos = ["/raid/datasets/SonyAlpha7_new/Augmented/C0025/0_1_0",
              "/raid/datasets/SonyAlpha7_new/Augmented/C0025/0_1_20",
              "/raid/datasets/SonyAlpha7_new/Augmented/C0025/0_1_40",
              "/raid/datasets/SonyAlpha7_new/Augmented/C0025/0_11_0",
              "/raid/datasets/SonyAlpha7_new/Augmented/C0025/0_11_20",
              "/raid/datasets/SonyAlpha7_new/Augmented/C0025/0_11_40"]
    # for video_path in videos:
    #     # video_path = "/raid/datasets/youtube_blur/Tilton/good_blur3_center"
    #     masks_path = "/raid/yoav/DVC/temp_masking"
    #     double_compression_method(video_path, fps=240, encoder='H264', metric='crf', mask_dir=masks_path, extra_str=os.path.basename(video_path))

    # for video_path in videos:
    #     encode_h264_with_given_crf(video_path, crf=30, ext=".png", fps=240, file_save_dir="/raid/yoav/DVC/small_garden/videos", string=os.path.basename(video_path))

    encode_static_video_and_static_video_with_moving_block()
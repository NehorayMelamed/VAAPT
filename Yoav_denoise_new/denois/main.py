import math
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skvideo
import torch
from skvideo import io

import torch.nn
import torchvision
from matplotlib import image as mpimg

from PARAMETER import path_restore_ckpt_denoise_flow_former, path_checkpoint_latest_things
from RDND_proper.imshow_torch_local import imshow_torch_video
from RDND_proper.models.FlowFormer.core.FlowFormer.LatentCostFormer.transformer import FlowFormer
from RDND_proper.models.FlowFormer.configs.sintel import get_cfg
from RDND_proper.models.irr.models.IRR_PWC import PWCNet
from easydict import EasyDict
from util.crop_image_using_mouse import ImageCropper

DEVICE = 0
# Get the current working directory
cwd = os.getcwd()
# Get the parent of the parent directory
BASE_NAME_RESTORE = os.path.dirname(os.path.dirname(cwd))
# Print the directory


###### Rapid base

def scale_array_to_range(input_tensor, min_max_values_to_scale_to=(0, 1)):
    input_tensor_normalized = (input_tensor - input_tensor.min()) / (
                input_tensor.max() - input_tensor.min() + 1e-16) * (
                                          min_max_values_to_scale_to[1] - min_max_values_to_scale_to[0]) + \
                              min_max_values_to_scale_to[0]
    return input_tensor_normalized


def BW2RGB(input_image):
    ### For Both Torch Tensors and Numpy Arrays!: ###
    # Actually... we're not restricted to RGB....
    if len(input_image.shape) == 2:
        if type(input_image) == torch.Tensor:
            RGB_image = input_image.unsqueeze(0)
            RGB_image = torch.cat([RGB_image, RGB_image, RGB_image], 0)
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


import fnmatch


def string_match_pattern(input_string, input_pattern):
    return fnmatch.fnmatch(input_string, input_pattern)


def path_get_files_from_folder(path, number_of_files=np.inf, flag_recursive=False, string_pattern_to_search='*',
                               flag_full_path=True):
    count = 0
    image_filenames_list = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        if count >= number_of_files and number_of_files != np.inf:
            break
        for fname in sorted(fnames):
            if count >= number_of_files and number_of_files != np.inf:
                break

            # if string_match_pattern(fname, string_pattern_to_search):
            if string_match_pattern(os.path.join(dirpath, fname), string_pattern_to_search):
                if flag_full_path:
                    img_path = os.path.join(dirpath, fname)
                else:
                    img_path = fname

                image_filenames_list.append(img_path)
                count += 1

        if flag_recursive == False:
            break
    return image_filenames_list


def video_torch_array_to_video(input_tensor, video_name='my_movie.mp4', FPS=25.0, flag_stretch=False,
                               output_shape=None):
    print("Sving video into ", video_name)
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
    output_numpy_array = numpy_array_to_video_ready(output_tensor.permute([0, 2, 3, 1]).clamp(0, 255).cpu().numpy())
    if flag_stretch:
        output_numpy_array = scale_array_to_range(output_numpy_array, (0, 255))
    skvideo.io.vwrite(video_name,
                      output_numpy_array,
                      outputdict={'-vcodec': 'libx264',
                                  '-b': '300000000'})  # chose the bitrate to be very high so no loss of information

    # for frame_counter in np.arange(T):
    #     current_frame = input_tensor[frame_counter]
    #     current_frame = current_frame.permute([1,2,0]).cpu().numpy()
    #     current_frame = BW2RGB(current_frame)
    #     current_frame = (current_frame * 255).clip(0,255).astype(np.uint8)
    #     video_writer.write(current_frame)
    # video_writer.release()


########

def get_full_shape_torch(input_tensor):
    if len(input_tensor.shape) == 1:
        W = input_tensor.shape
        H = 1
        C = 1
        T = 1
        B = 1
        shape_len = 1
        shape_vec = (W)
    elif len(input_tensor.shape) == 2:
        H, W = input_tensor.shape
        C = 1
        T = 1
        B = 1
        shape_len = 2
        shape_vec = (H, W)
    elif len(input_tensor.shape) == 3:
        C, H, W = input_tensor.shape
        T = 1
        B = 1
        shape_len = 3
        shape_vec = (C, H, W)
    elif len(input_tensor.shape) == 4:
        T, C, H, W = input_tensor.shape
        B = 1
        shape_len = 4
        shape_vec = (T, C, H, W)
    elif len(input_tensor.shape) == 5:
        B, T, C, H, W = input_tensor.shape
        shape_len = 5
        shape_vec = (B, T, C, H, W)
    shape_vec = np.array(shape_vec)
    return (B, T, C, H, W), shape_len, shape_vec


def torch_to_numpy(input_tensor):
    if type(input_tensor) == torch.Tensor:
        (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
        input_tensor = input_tensor.cpu().data.numpy()
        if shape_len == 2:
            # [H,W]
            return input_tensor
        elif shape_len == 3:
            # [C,H,W] -> [H,W,C]
            return np.transpose(input_tensor, [1, 2, 0])

        elif shape_len == 4:
            # [T,H,C,W] -> [T,H,W,C]
            if min(input_tensor.shape) == input_tensor.shape[len(input_tensor.shape) - 1]:
                # means the color is in the second index before the end i got it somthimes
                # print(np.transpose(input_tensor, [0, 1, 2, 3]))
                return np.transpose(input_tensor, [0, 1, 2, 3])
            # [T,C,H,W] -> [T,H,W,C]
            else:
                return np.transpose(input_tensor, [0, 2, 3, 1])
        elif shape_len == 5:
            # [B,T,C,H,W] -> [B,T,H,W,C]
            return np.transpose(input_tensor, [0, 1, 3, 4, 2])
    return input_tensor


def load_state_dict_into_module(model, state_dict, strict=True, loading_prefix=None):
    own_state = model.state_dict()
    count1 = 0
    count2 = 0
    count3 = 0

    for name, param in state_dict.items():
        if loading_prefix != None:
            name = name.partition(loading_prefix)[-1]

        if name in own_state:  # probably too specific to remove name prefix like this
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].resize_as_(param)
                own_state[name].copy_(param)
                count1 += 1
            except Exception:
                if strict:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
                else:
                    print(f'note : parameter {name} were not copied, you might have a wrong checkpoint')
                    count2 += 1
                    pass
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    print(
        f'out of {len(state_dict.items())} params: {count1} params were strictly copied / resized, {count2} params were not copied')

    return own_state


def numpy_to_torch(input_image, device='cpu', flag_unsqueeze=False):
    # Assumes "Natural" RGB form to there's no BGR->RGB conversion, can also accept BW image!:
    if input_image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        input_image = np.expand_dims(input_image, axis=2)  # [H,W]->[H,W,1]
    if input_image.ndim == 3:
        input_image = np.transpose(input_image, (2, 0, 1))  # [H,W,C]->[C,H,W]
    elif input_image.ndim == 4:
        input_image = np.transpose(input_image, (0, 3, 1, 2))  # [T,H,W,C] -> [T,C,H,W]
    input_image = torch.from_numpy(input_image.astype(np.float)).float().to(device)  # to float32

    if flag_unsqueeze:
        input_image = input_image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
    return input_image


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


def load_flow_former(checkpoint='RDND_proper/models/FlowFormer/check_points/sintel.pth', base_name_restore=BASE_NAME_RESTORE):
    """

    Args:
        checkpoint: model checkpoint
        train_devices: gpus
        base_name_restore: base name to make a valid path

    Returns:

    """
    args = EasyDict()
    args.restore_ckpt = os.path.join(base_name_restore, checkpoint)
    args.name = 'flowformer'
    args.stage = 'sintel'
    args.validation = 'true'
    args.mixed_precision = True
    cfg = get_cfg()
    cfg.update(vars(args))

    # model = torch.nn.DataParallel(FlowFormer(cfg.latentcostformer), device_ids=[DEVICE]).to(DEVICE)
    # model.load_state_dict(torch.load(args.restore_ckpt))
    # return model
    ##ToDo i just paste a const path, it should be like abouve
    model = torch.nn.DataParallel(FlowFormer(cfg.latentcostformer), device_ids=[DEVICE]).to(DEVICE)
    model.load_state_dict(torch.load(path_restore_ckpt_denoise_flow_former))
    return model

def bicubic_interpolate(input_image, X, Y):
    ### From [B,C,H,W] -> [B*C,H,W]
    x_shape = input_image.shape
    B, C, H, W = input_image.shape
    BXC = x_shape[0] * x_shape[1]
    input_image = input_image.contiguous().reshape(-1, int(x_shape[2]), int(x_shape[3]))  # [B,C,H,W]->[B*C,H,W]

    # height = new_grid.shape[1]
    # width = new_grid.shape[2]
    height = input_image.shape[1]
    width = input_image.shape[2]

    ### Reshape & Extract delta maps: ###
    # theta_flat = new_grid.contiguous().view(new_grid.shape[0], height * width, new_grid.shape[3])  #[B,H*W,2] - spatial flattening
    # delta_x_flat = theta_flat[:, :, 0:1]
    # delta_y_flat = theta_flat[:, :, 1:2]
    ###
    if X.shape[0] != input_image.shape[0]:  # input_image.shape=[B,C,H,W] but X.shape=[B,H,W,1]  --> X.shape=[BXC,H,W,1]
        X = X.repeat([C, 1, 1, 1])
        Y = Y.repeat([C, 1, 1, 1])
    delta_x_flat = X.contiguous().view(BXC, H * W, 1)
    delta_y_flat = Y.contiguous().view(BXC, H * W, 1)

    ### Flatten completely: [BXC,H*W,1] -> [B*H*W]: ###
    x_map = delta_x_flat.contiguous().view(-1)
    y_map = delta_y_flat.contiguous().view(-1)
    x_map = x_map.float()
    y_map = y_map.float()
    height_f = float(height)
    width_f = float(width)

    ### Take Care of needed symbolic variables: ###
    zero = 0
    max_y = int(height - 1)
    max_x = int(width - 1)
    ###
    x_map = (x_map + 1) * (
            width_f - 1) / 2.0  # Here i divide again by 2?!!?!....then why multiply by 2 in the first place?!
    y_map = (y_map + 1) * (height_f - 1) / 2.0
    ###
    x0 = x_map.floor().int()
    y0 = y_map.floor().int()
    ###
    xm1 = x0 - 1
    ym1 = y0 - 1
    ###
    x1 = x0 + 1
    y1 = y0 + 1
    ###
    x2 = x0 + 2
    y2 = y0 + 2
    ###
    tx = x_map - x0.float()
    ty = y_map - y0.float()

    ### the coefficients are for a=-1/2
    c_xm1 = ((-tx ** 3 + 2 * tx ** 2 - tx) / 2.0)
    c_x0 = ((3 * tx ** 3 - 5 * tx ** 2 + 2) / 2.0)
    c_x1 = ((-3 * tx ** 3 + 4 * tx ** 2 + tx) / 2.0)
    c_x2 = (1.0 - (c_xm1 + c_x0 + c_x1))

    c_ym1 = ((-ty ** 3 + 2 * ty ** 2 - ty) / 2.0)
    c_y0 = ((3 * ty ** 3 - 5 * ty ** 2 + 2) / 2.0)
    c_y1 = ((-3 * ty ** 3 + 4 * ty ** 2 + ty) / 2.0)
    c_y2 = (1.0 - (c_ym1 + c_y0 + c_y1))

    # TODO: pad image for bicubic interpolation and clamp differently if necessary
    xm1 = xm1.clamp(zero, max_x)
    x0 = x0.clamp(zero, max_x)
    x1 = x1.clamp(zero, max_x)
    x2 = x2.clamp(zero, max_x)

    ym1 = ym1.clamp(zero, max_y)
    y0 = y0.clamp(zero, max_y)
    y1 = y1.clamp(zero, max_y)
    y2 = y2.clamp(zero, max_y)

    dim2 = width
    dim1 = width * height

    ### Take care of indices base for flattened indices: ###
    # TODO: avoid using a for loop
    base = torch.zeros(dim1 * BXC).int().to(input_image.device)  # TODO: changed to dim1*B
    for i in np.arange(BXC):
        base[(i + 1) * H * W: (i + 2) * H * W] = torch.Tensor([(i + 1) * H * W]).to(input_image.device).int()

    base_ym1 = base + ym1 * dim2
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    base_y2 = base + y2 * dim2

    idx_ym1_xm1 = base_ym1 + xm1
    idx_ym1_x0 = base_ym1 + x0
    idx_ym1_x1 = base_ym1 + x1
    idx_ym1_x2 = base_ym1 + x2

    idx_y0_xm1 = base_y0 + xm1
    idx_y0_x0 = base_y0 + x0
    idx_y0_x1 = base_y0 + x1
    idx_y0_x2 = base_y0 + x2

    idx_y1_xm1 = base_y1 + xm1
    idx_y1_x0 = base_y1 + x0
    idx_y1_x1 = base_y1 + x1
    idx_y1_x2 = base_y1 + x2

    idx_y2_xm1 = base_y2 + xm1
    idx_y2_x0 = base_y2 + x0
    idx_y2_x1 = base_y2 + x1
    idx_y2_x2 = base_y2 + x2

    # (*). TODO: thought: this flattening coupled with torch.index_select assumes that number of elements in input_image_flat = number of elements in the indices (like idx_y2_xm1)...but input_imag_flat includes Channels!!!....
    input_image_flat = input_image.contiguous().view(-1).float()

    I_ym1_xm1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_ym1_xm1.long()))
    I_ym1_x0 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_ym1_x0.long()))
    I_ym1_x1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_ym1_x1.long()))
    I_ym1_x2 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_ym1_x2.long()))

    I_y0_xm1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y0_xm1.long()))
    I_y0_x0 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y0_x0.long()))
    I_y0_x1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y0_x1.long()))
    I_y0_x2 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y0_x2.long()))

    I_y1_xm1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y1_xm1.long()))
    I_y1_x0 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y1_x0.long()))
    I_y1_x1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y1_x1.long()))
    I_y1_x2 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y1_x2.long()))

    I_y2_xm1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y2_xm1.long()))
    I_y2_x0 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y2_x0.long()))
    I_y2_x1 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y2_x1.long()))
    I_y2_x2 = torch.index_select(input_image_flat, dim=0, index=torch.Tensor(idx_y2_x2.long()))

    output_ym1 = c_xm1 * I_ym1_xm1 + c_x0 * I_ym1_x0 + c_x1 * I_ym1_x1 + c_x2 * I_ym1_x2
    output_y0 = c_xm1 * I_y0_xm1 + c_x0 * I_y0_x0 + c_x1 * I_y0_x1 + c_x2 * I_y0_x2
    output_y1 = c_xm1 * I_y1_xm1 + c_x0 * I_y1_x0 + c_x1 * I_y1_x1 + c_x2 * I_y1_x2
    output_y2 = c_xm1 * I_y2_xm1 + c_x0 * I_y2_x0 + c_x1 * I_y2_x1 + c_x2 * I_y2_x2

    # TODO: changed from height,width to B,height,width
    output = c_ym1.view(BXC, height, width) * output_ym1.view(BXC, height, width) + \
             c_y0.view(BXC, height, width) * output_y0.view(BXC, height, width) + \
             c_y1.view(BXC, height, width) * output_y1.view(BXC, height, width) + \
             c_y2.view(BXC, height, width) * output_y2.view(BXC, height, width)

    # output = output.clamp(zero, 1.0)  #TODO: why clamp?!?!!?
    output = output.contiguous().reshape(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
    return output


def path_get_all_filenames_from_folder(path, flag_recursive=False, flag_full_filename=True):
    filenames_list = []
    # Look recursively at all current and children path's in given path and get all binary files
    for dirpath, _, fnames in sorted(os.walk(path)):  # os.walk!!!!
        for fname in sorted(fnames):
            if flag_full_filename:
                file_full_filename = os.path.join(dirpath, fname)
            else:
                file_full_filename = fname
            filenames_list.append(file_full_filename)
        if flag_recursive == False:
            break;
    return filenames_list;


def load_pwc(checkpoint='RDND_proper/models/irr/checkpoints/checkpoint_latest_things.ckpt', base_name_restore=BASE_NAME_RESTORE):
    """

    Args:
        checkpoint: model checkpoint
        train_devices: gpus
        base_name_restore: base name to make a valid path

    Returns:

    """
    args = EasyDict()
    args.restore_ckpt = os.path.join(base_name_restore, checkpoint)

    model = PWCNet(args).to(DEVICE)
    loading_prefix = '_model.'
    pretrained_dict = torch.load(path_checkpoint_latest_things, map_location=torch.device(DEVICE))['state_dict']
    state_dict = load_state_dict_into_module(model, pretrained_dict, strict=False, loading_prefix=loading_prefix)
    model.load_state_dict(state_dict)
    return model


class Warp_Object(torch.nn.Module):
    # Initialize this with a module
    def __init__(self):
        super(Warp_Object, self).__init__()
        self.X = None
        self.Y = None

    # Elementwise sum the output of a submodule to its input
    def forward(self, input_image, delta_x, delta_y, flag_bicubic_or_bilinear='bilinear'):
        # delta_x = map of x deltas from meshgrid, shape=[B,H,W] or [B,C,H,W].... same for delta_Y
        B, C, H, W = input_image.shape
        BXC = B * C

        ### ReOrder delta_x, delta_y: ###
        # TODO: this expects delta_x,delta_y to be image sized tensors. but sometimes i just wanna pass in a single number per image
        # (1). Dim=3 <-> [B,H,W], I Interpret As: Same Flow On All Channels:
        if len(delta_x.shape) == 3:  # [B,H,W] - > [B,H,W,1]
            delta_x = delta_x.unsqueeze(-1)
            delta_y = delta_x.unsqueeze(-1)
            flag_same_on_all_channels = True
        # (2). Dim=4 <-> [B,C,H,W], Different Flow For Each Channel:
        elif (len(delta_x.shape) == 4 and delta_x.shape[
            1] == C):  # [B,C,H,W] - > [BXC,H,W,1] (because pytorch's function only warps all channels of a tensor the same way so in order to warp each channel seperately we need to transfer channels to batch dim)
            delta_x = delta_x.view(B * C, H, W).unsqueeze(-1)
            delta_y = delta_y.view(B * C, H, W).unsqueeze(-1)
            flag_same_on_all_channels = False
        # (3). Dim=4 but C=1 <-> [B,1,H,W], Same Flow On All Channels:
        elif len(delta_x.shape) == 4 and delta_x.shape[1] == 1:
            delta_x = delta_x.permute([0, 2, 3, 1])  # [B,1,H,W] -> [B,H,W,1]
            delta_y = delta_y.permute([0, 2, 3, 1])
            flag_same_on_all_channels = True
        # (4). Dim=4 but C=1 <-> [B,H,W,1], Same Flow On All Channels:
        elif len(delta_x.shape) == 4 and delta_x.shape[3] == 1:
            flag_same_on_all_channels = True

        flag_same_on_all_channels = True

        ### Create "baseline" meshgrid (as the function ultimately accepts a full map of locations and not just delta's): ###
        # (*). ultimately X.shape=[BXC, H, W, 1]/[B,H,W,1]... so check if the input shape has changed and only then create a new meshgrid:
        flag_input_changed_from_last_time = (self.X is None) or (
                self.X.shape[0] != BXC and flag_same_on_all_channels == False) or (
                                                    self.X.shape[0] != B and flag_same_on_all_channels == True) or (
                                                    self.X.shape[1] != H) or (self.X.shape[2] != W)
        if flag_input_changed_from_last_time:
            print('new meshgrid')
            [X, Y] = np.meshgrid(np.arange(W), np.arange(H))  # X.shape=[H,W]
            if flag_same_on_all_channels:
                X = torch.Tensor(np.array([X] * B)).unsqueeze(-1)  # X.shape=[B,H,W,1]
                Y = torch.Tensor(np.array([Y] * B)).unsqueeze(-1)
            else:
                X = torch.Tensor(np.array([X] * BXC)).unsqueeze(-1)  # X.shape=[BXC,H,W,1]
                Y = torch.Tensor(np.array([Y] * BXC)).unsqueeze(-1)
            X = X.to(input_image.device)
            Y = Y.to(input_image.device)
            self.X = X
            self.Y = Y

        # [X, Y] = np.meshgrid(np.arange(W), np.arange(H))
        # X = torch.Tensor([X] * BXC).unsqueeze(-1)
        # Y = torch.Tensor([Y] * BXC).unsqueeze(-1)
        # X = X.to(input_image.device)
        # Y = Y.to(input_image.device)

        ### Add Difference (delta) Maps to Meshgrid: ###
        ### Previous Try: ###
        # X += delta_x
        # Y += delta_y
        # X = (X - W / 2) / (W / 2 - 1)
        # Y = (Y - H / 2) / (H / 2 - 1)
        # ### Previous Use: ###
        # new_X = ((self.X + delta_x) - W / 2) / (W / 2 - 1)
        # new_Y = ((self.Y + delta_y) - H / 2) / (H / 2 - 1)
        ### New Use: ###
        # new_X = 2 * ((self.X - delta_x)) / max(W-1,1) - 1
        # new_Y = 2 * ((self.Y - delta_y)) / max(H-1,1) - 1
        new_X = 2 * ((self.X + delta_x)) / max(W - 1, 1) - 1
        new_Y = 2 * ((self.Y + delta_y)) / max(H - 1, 1) - 1
        # ### No Internal Tensors: ###
        # new_X = 2 * ((X + delta_x)) / max(W - 1, 1) - 1
        # new_Y = 2 * ((Y + delta_y)) / max(H - 1, 1) - 1

        if flag_bicubic_or_bilinear == 'bicubic':
            # input_image.shape=[B,C,H,W] , new_X.shape=[B,H,W,1] OR new_X.shape=[BXC,H,W,1]
            warped_image = bicubic_interpolate(input_image, new_X, new_Y)
            return warped_image
        else:
            bilinear_grid = torch.cat([new_X, new_Y], dim=3)
            if flag_same_on_all_channels:
                # input_image.shape=[B,C,H,W] , bilinear_grid.shape=[B,H,W,2]
                input_image_to_bilinear = input_image
                warped_image = torch.nn.functional.grid_sample(input_image_to_bilinear, bilinear_grid)
                return warped_image
            else:
                # input_image.shape=[BXC,1,H,W] , bilinear_grid.shape=[BXC,H,W,2]
                input_image_to_bilinear = input_image.reshape(-1, int(H), int(W)).unsqueeze(1)  # [B,C,H,W]->[B*C,1,H,W]
                warped_image = torch.nn.functional.grid_sample(input_image_to_bilinear, bilinear_grid)
                warped_image = warped_image.view(B, C, H, W)
                return warped_image


def get_optical_flow_and_occlusion_on_video(input, flow_model=None, occ_model=None):
    if flow_model is None:
        flow_model = load_flow_former().eval()
    if occ_model is None:
        occ_model = load_pwc().eval()

    flow_model_output = []
    occlusion_model_output = []
    # self.outputs_dict.model_output = []
    num_frames = input.shape[1]
    with torch.no_grad():
        for i in range(num_frames - 1):
            ### Flowformer Output: ###
            flowformer_optical_flow, _, _, _ = flow_model(torch.cat([input[:, i:i + 1], input[:, i + 1:i + 2]], 1))
            flow_model_output.append(flowformer_optical_flow)

            ### Occlusion model output: ###
            _, occlusion_model_occlusion, _, _ = occ_model.forward(
                torch.cat([input[:, i:i + 1], input[:, i + 1:i + 2]], 1))
            occlusion_model_output.append(occlusion_model_occlusion)
            # self.outputs_dict.model_output.append(model_output[-1])

    ### Concat Optical Flow Outputs: ###
    torch.cuda.empty_cache()
    # flow_model_output = torch.cat([f[0].unsqueeze(0) for f in flow_model_output])  # take the flow(0) argument
    optical_flow = torch.cat([f.unsqueeze(0) for f in flow_model_output])
    torch.cuda.empty_cache()

    ### Concat Occlusion Outputs: ###
    torch.cuda.empty_cache()
    occlusion = torch.cat([f.unsqueeze(0) for f in occlusion_model_output])  # take the occlusion(1) argument
    torch.cuda.empty_cache()

    # returns [frame count, 1, 2, H, W], [frame count, 1, 1, H, W]
    optical_flow = torchvision.utils.flow_to_image(optical_flow.squeeze()).unsqueeze(1)
    return optical_flow, occlusion


def output_n_frames(flow_model, occ_model, input):
    flow_model_output = []
    occlusion_model_output = []
    # self.outputs_dict.model_output = []
    num_frames = input.shape[1]
    # TODO: figure out that we really compare the refrence to anything else
    # input = torch.from_numpy(input)
    ref_frame = input[:, ((num_frames - 1) // 2)].unsqueeze(0)
    with torch.no_grad():
        for i in range(input.shape[1]):
            # if i != ((Train_dict.frames - 1) // 2):
            # TODO: NOTE: it is important that the model calculates with ref as im1
            # print(i, ref_frame.shape)
            print(i)
            ### Flowformer Output: ###
            flowformer_optical_flow, bla1, bla2, bla3 = flow_model(torch.cat([ref_frame, input[:, i:i + 1]], 1))
            flow_model_output.append(flowformer_optical_flow)

            ### Occlusion model output: ###
            occlusion_model_optical_flow, occlusion_model_occlusion, bla1, bla2 = occ_model.forward(
                torch.cat([ref_frame, input[:, i:i + 1]], 1))
            occlusion_model_output.append(occlusion_model_occlusion)
            # self.outputs_dict.model_output.append(model_output[-1])

    ### Concat Optical Flow Outputs: ###
    torch.cuda.empty_cache()
    # flow_model_output = torch.cat([f[0].unsqueeze(0) for f in flow_model_output])  # take the flow(0) argument
    flow_model_output = torch.cat([f.unsqueeze(0) for f in flow_model_output])
    torch.cuda.empty_cache()

    ### Concat Occlusion Outputs: ###
    torch.cuda.empty_cache()
    occlusion_model_occlusion = torch.cat(
        [f.unsqueeze(0) for f in occlusion_model_occlusion])  # take the occlusion(1) argument
    torch.cuda.empty_cache()

    model_conf_xs = None
    model_conf_ys = None

    model_output = [flow_model_output, occlusion_model_occlusion, model_conf_xs, model_conf_ys]
    return model_output


def denoise_model_outputs(model_output, input_frames):
    warp_object = Warp_Object()
    flows, occlusion_maps, _, _ = model_output

    ### Get Original and Final Shape: ###
    N_Frames, B_flow_original, T_flow_original, H_flow_original, W_flow_original = flows.shape
    # valid_mask = valid_mask[0][0][0:1].repeat(c, 1, 1).unsqueeze(0)
    middle_frame_index = (flows.shape[0] - 1) // 2
    warp_list = []
    warp_list_unscaled = []

    for idx, flow in enumerate(flows):
        if idx != middle_frame_index:
            image_to_warp = input_frames[:, idx].float()  # upsampler(images_original[0, idx:idx + 1]) # up scale frames
            warped_frames_towrds_model = warp_object.forward(input_image=image_to_warp,
                                                             delta_x=flow[0:1, 0:1],
                                                             delta_y=flow[0:1, 1:2])
            warp_list.append(warped_frames_towrds_model)

    if occlusion_maps is not None:  # by default i get here [T, 1 H, W] or [T, 2, H, W]
        if occlusion_maps.shape[1] == 1:  # 1 dimensional confidence/occlusion estimate
            # exclude middle index
            occlusion_maps = torch.cat([occlusion_maps[:middle_frame_index], occlusion_maps[middle_frame_index + 1:]])

            Weights = 1 - occlusion_maps.squeeze(1)
            warped_tensor = torch.cat(warp_list, 0)
            final_weighted_estimate_nominator = (warped_tensor * Weights).sum(0)
            final_weighted_estimate_denominator = (Weights).sum(0) + 1e-3
            final_weighted_estimate = final_weighted_estimate_nominator / final_weighted_estimate_denominator
            final_unweighted_estimate = warped_tensor.mean(0)

            final_estimate = final_weighted_estimate.unsqueeze(0)


        elif occlusion_maps.shape[1] == 2:  # 2 dimensional confidence/occlusion estimate
            occlusion_maps = torch.cat([occlusion_maps[:middle_frame_index], occlusion_maps[middle_frame_index + 1:]])

            # todo: same code as callbacks, get a function for it
            # occlusion_maps /= occlusion_maps.sum(0)
            Weights = 1 - occlusion_maps.squeeze(1)
            warped_tensor = torch.cat(warp_list, 0)
            final_weighted_estimate_nominator = (warped_tensor * Weights).sum(0)
            final_weighted_estimate_denominator = (Weights).sum(0) + 1e-3
            final_weighted_estimate = final_weighted_estimate_nominator / final_weighted_estimate_denominator
            final_unweighted_estimate = warped_tensor.mean(0)

            final_estimate = final_weighted_estimate.unsqueeze(0)
    else:
        warped_tensor = torch.cat([warp_list[i].unsqueeze(0) for i in range(len(warp_list))], 0)
        final_estimate = warped_tensor.mean(0)

        warped_tensor_unscaled = torch.cat([warp_list[i].unsqueeze(0) for i in range(len(warp_list_unscaled))], 0)
        final_estimate_unscaled = warped_tensor_unscaled.mean(0)

    return final_estimate, warp_list


def denoise_main(input_frames, flow_model=None, occ_model=None):
    if flow_model is None:
        flow_model = load_flow_former().eval()
    if occ_model is None:
        occ_model = load_pwc().eval()

    num_frames = input_frames.shape[1]
    ref_frame = input_frames[:, ((num_frames - 1) // 2)].unsqueeze(0)

    ### Get Optical-Flow and Occlusions From each frame with reference to center frame: ###
    outputs = output_n_frames(flow_model, occ_model, input_frames)

    ### Use the optical-flow and occlusion outputs to "smart-average" out the images with regard to center frame: ###
    denoised_reference, warp_list = denoise_model_outputs(outputs, input_frames)

    return denoised_reference, warp_list


# from RapidBase.Utils.Classical_DSP.ECC_layer_points import *


def closest_power_of_two(num):
    if num <= 0:
        return 0
    power = int((math.log(num, 2)))
    return 2 ** power


##############################
##############################
#### Dudy ECC scope       ####
##############################
##############################
if __name__ == "__main__":
    pass
    # torch.cuda.empty_cache()

    ### Get images from .avi file: ####
    # video_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/HERTSEL_SHAZAR/ch01_00000000033000000.mp4"
    # input = video_read_video_to_numpy_tensor(video_path, 2, 5)
    # input_as_torch = torch.from_numpy(input)
    # input_as_torch = input_as_torch.to(DEVICE)

    # ### Get images from ID crops: ###
    # import cv2
    # crops_path = r'/home/dudy/Nehoray/SHABACK_POC_NEW/data/object_images_result/ID_5_Type_car'
    # crops_filenames_list = path_get_all_filenames_from_folder(crops_path, flag_recursive=False, flag_full_filename=True)
    # crops_tensor = read_images_from_folder(crops_path, max_number_of_images=100, flag_return_torch=True)
    # crops_list = []
    # for crop_filename in crops_filenames_list:
    #     current_frame = cv2.imread(crop_filename)
    #     current_frame_torch = numpy_to_torch(current_frame)
    #     current_frame_torch = torch.nn.Upsample(size=(3, 256, 256), mode='bilinear')(current_frame_torch)
    #     crops_list.append(current_frame_torch)
    # current_frame_torch = torch.cat(crops_list)

    ### Load image crops: ###
    # base_directory = "/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/ex/crops/car/1" #TODO: REPLACE PATH !!!!
    # segmentation_filenames_list = path_get_files_from_folder(base_directory, string_pattern_to_search='*.pt*')
    # crops_filenames_list = path_get_files_from_folder(base_directory, string_pattern_to_search='*.jpg*')  #TODO: change to .png!!#@%@
    #
    # # string_rjust(23,4)
    #
    # ### Make all crops and segmentation masks the same dimensions: ###
    # image_crops_list = []
    # segmentation_crops_list = []
    # final_crop_size = (256, 256)
    # for file_index in np.arange(10):
    #     print(file_index)
    #     ### Get current crop and segmentation filename: ###
    #     current_segmentation_filename = segmentation_filenames_list[file_index]
    #     current_crop_filename = crops_filenames_list[file_index]
    #
    #     ### Read crop and segmentation file: ###
    #     current_image_crop = mpimg.imread(current_crop_filename)
    #     current_segmentation_crop = torch.load(current_segmentation_filename)
    #
    #     ### Turn crop image to tensor: ###
    #     current_image_crop = numpy_to_torch(current_image_crop)
    #     current_image_crop = current_image_crop.unsqueeze(0)  # [C,H,W] -> [B,C,H,W]
    #
    #     ### Upsample All to same size: ###
    #     upsample_layer_bilinear = torch.nn.Upsample(size=final_crop_size, mode='bilinear')
    #     upsample_layer_nearest = torch.nn.Upsample(size=final_crop_size, mode='nearest')
    #     current_image_crop_upsampled = upsample_layer_bilinear(current_image_crop)
    #     current_segmentation_crop_upsampled = upsample_layer_nearest(current_segmentation_crop)
    #
    #     ### Append final outputs to list: ###
    #     image_crops_list.append(current_image_crop_upsampled)
    #     segmentation_crops_list.append(current_segmentation_crop_upsampled)
    #
    #     # ### Show images for debugging: ###
    #     # imshow_torch(current_image_crop/255)
    #     # imshow_torch(current_image_crop_upsampled/255)
    #     # # plt.imshow(current_image_crop.squeeze().premute(()))
    #     # plt.show()
    #
    # ### Concat all tensors in list to single tensor: ###
    # crops_tensor_final = torch.cat(image_crops_list)
    # segmentation_tensor_final = torch.cat(segmentation_crops_list)

    # ### Use ECC: ###
    # ### Stabilize Frames: ###
    # number_of_frames_per_batch = 1
    # input_tensor = crops_tensor_final
    # number_of_pixels_to_use = 30000
    # number_of_batches = input_tensor.shape[0] / number_of_frames_per_batch
    # reference_tensor = input_tensor[0:1]
    # precision = torch.float
    # input_tensor = RGB2BW(input_tensor)
    # reference_tensor = RGB2BW(reference_tensor)
    # ECC_layer_object = ECC_Layer_Torch_Points_Batch(input_tensor[0:1],
    #                                                 reference_tensor,
    #                                                 number_of_iterations_per_level=50,
    #                                                 number_of_levels=1,
    #                                                 transform_string='homography',
    #                                                 number_of_pixels_to_use=number_of_pixels_to_use,
    #                                                 delta_p_init=None,
    #                                                 precision=precision)
    # input_tensor = ECC_layer_object.forward_iterative(input_tensor[0:1].type(precision),
    #                                                   reference_tensor.type(precision),
    #                                                   max_shift_threshold=2e-3,
    #                                                   flag_print=False,
    #                                                   delta_p_init=None,
    #                                                   number_of_images_per_batch=number_of_frames_per_batch,
    #                                                   flag_calculate_gradient_in_advance=False,
    #                                                   segmentation_mask=segmentation_tensor_final[0:1])
    # del ECC_layer_object

    ### TODO: some logic which takes only the most relevant images (meaning the ones which are the largest): ###

    ### Take Segmentation map and input into "dudy denoise block" which take in images and segmentation maps and uses ECC to align them: ###

    # torch.cuda.empty_cache()
    # input_as_torch = input_as_torch.unsqueeze(0)  # [B, T, C, H, W] = [1, 5, 3, 128, 128] for example
    # input_as_torch = torch.permute(input_as_torch, (0,1,4,2,3))
    #
    # ### Crop Tensors Before Denoise (CUDA out of memory): ###
    # input_as_torch = input_as_torch[:,:,:,0:256,0:256]

    # with torch.no_grad():
    #     denoised_reference, warped_frames = denoise_main(input_as_torch)
    # print(denoised_reference.shape)
    # print(warped_frames[0].shape)


##############################
##############################
#### Dudy ECC scope | END ####
##############################
##############################


#### Ead and load video


### Downs-maple resize video
# input_as_torch = torch.nn.Upsample(size=[3, 256, 256])(input_as_torch)

#### Actialy perrform Optical Flow and occlusion
# OF, occ = get_optical_flow_and_occlusion_on_video(input_as_torch)


# imshow_torch_video(occ.permute(1, 0, 2, 3, 4), FPS=3)
# imshow_torch_video(OF.permute(1, 0, 2, 3, 4), FPS=3)


# divide input_as_torch to several windows with certein crop size
# works for stride 1


###### Performse denoise


#### Regular reading Reade video ####
# video_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/HERTSEL_SHAZAR/ch01_00000000033000000.mp4"
# input = video_read_video_to_numpy_tensor(video_path, 0, 15)
# input_as_torch = torch.from_numpy(input)
# input_as_torch = input_as_torch.to(DEVICE)
# input_as_torch = input_as_torch.unsqueeze(0).permute(0, 1, 4, 2, 3)  # [B, T, C, H, W] = [1, 5, 3, 128, 128] for example


def main_denoise(window_size_temporal, stride, video_path=None, base_directory_of_cropped_images=None,
                 final_crop_size=(256, 256),
                 loop_over="all_video", use_roi=False, frame_index_to_start=0, frame_index_to_end=10,
                 noise_save_video_file_name="noise_video.mp4",
                 denoise_save_video_file_name="denoise_video.mp4", show_video=True):


    if base_directory_of_cropped_images is not None:
        ### Reade video from crop images(Of object tracker for example)
        crops_filenames_list = path_get_files_from_folder(base_directory_of_cropped_images,
                                                          string_pattern_to_search='*.jpg*')

        if crops_filenames_list != []:

            ### Make all crops the same dimensions: ###
            image_crops_list = []
            for file_index in range(len(crops_filenames_list)):
                print(file_index)
                ### Get current cropfilename: ###
                current_crop_filename = crops_filenames_list[file_index]

                ### Read crop file: ###
                current_image_crop = mpimg.imread(current_crop_filename)

                ### Turn crop image to tensor: ###
                current_image_crop = numpy_to_torch(current_image_crop)
                current_image_crop = current_image_crop.unsqueeze(0)  # [C,H,W] -> [B,C,H,W]

                ### Upsample All to same size: ###
                upsample_layer_bilinear = torch.nn.Upsample(size=final_crop_size, mode='bilinear')
                upsample_layer_nearest = torch.nn.Upsample(size=final_crop_size, mode='nearest')
                current_image_crop_upsampled = upsample_layer_bilinear(current_image_crop)

                ### Append final outputs to list: ###
                image_crops_list.append(current_image_crop_upsampled)
            crops_tensor_final = torch.cat(image_crops_list)
            input_as_torch = crops_tensor_final

    elif video_path is not None:
        numpy_video = video_read_video_to_numpy_tensor(video_path, frame_index_to_start, frame_index_to_end)
        input_as_torch = numpy_to_torch(numpy_video)

    else:
        return

    input_as_torch = input_as_torch / 256
    torch.cuda.empty_cache()

    ### Perform crop use_roi image
    if use_roi is True and base_directory_of_cropped_images is None:
        ### Avialibale only for video path
        ###  ROI crop image | Get relevant for ceop video ###

        first_frame = numpy_video[0]

        image_cropper = ImageCropper(first_frame)

        ### Get from user crop image
        first_image_cropped_image = image_cropper.get_tensor_crop()

        ### Set into members
        x1, y1, x2, y2 = image_cropper.get_coordinates_by_ROI_area()

        ### the new video crop acording to the ROI
        # [:, y1:y2, x1:x2, :]
        # input_as_torch = input_as_torch[:, :, y1:y2, x1:x2]

        # ### ROI crop image | Get closess 16 power ###
        w, h = math.floor((x2 - x1) / 16) * 16, math.floor((y2 - y1) / 16) * 16

        input_as_torch = input_as_torch[:, :, y1:(y1 + h), x1:(x1 + w)]

    input_as_torch = input_as_torch.unsqueeze(0)

    ## Set the limit loop
    if loop_over == "all_video":
        loop_over = input_as_torch.shape[1]

    list_of_denoised_video = []
    for i in range(0, loop_over - 1 - window_size_temporal, stride):
        torch.cuda.empty_cache()
        relevant_frames_area = input_as_torch[:, i:i + window_size_temporal, :, :]

        relevant_frames_area = relevant_frames_area.cuda()
        denoised_reference, warped_frames = denoise_main(relevant_frames_area)

        list_of_denoised_video.append(denoised_reference)
        print(len(list_of_denoised_video))

    denoised_tensor_video = torch.cat(list_of_denoised_video)
    denoised_tensor_video = denoised_tensor_video / denoised_tensor_video.abs().max()

    ### Save video ####
    video_torch_array_to_video(input_tensor=denoised_tensor_video, video_name=denoise_save_video_file_name)
    input_as_torch = input_as_torch.squeeze(0)
    video_torch_array_to_video(input_tensor=input_as_torch, video_name=noise_save_video_file_name)

    if show_video is True:
        imshow_torch_video(denoised_tensor_video)

    return True





# Main is here below

# main_denoise(use_roi=True, window_size_temporal=4, stride=1,
#              video_path="/home/nehoray/PycharmProjects/Shaback/output/VIDEO_EDITOR/video_example_from_security_camera.mp4")




# main_denoise(use_roi=True, window_size_temporal=4, stride=1,
#              base_directory_of_cropped_images="/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/ex/crops/car/1")


































# video_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/TSOMET__HERTSOG_ZALMAN_SHNEOR/ch01_00000000212000000.mp4"
# numpy_video = video_read_video_to_numpy_tensor(video_path, 0, 10)
# # video_tensor = numpy_to_torch(video_tensor)
# # (T,C,H,W)
# tensor_video = torch.from_numpy(numpy_video).permute(0,3,1,2)
# display_video(tensor_video)
#

# #
# ### Reade video from crop images(Of object tracker for example)
# base_directory = r'/home/dudy/Nehoray/SHABACK_POC_NEW/ecc_segmantion_layer/data/minivan_1'
# final_crop_size = (256, 256)
# crops_filenames_list = path_get_files_from_folder(base_directory, string_pattern_to_search='*.jpg*')
#
# ### Make all crops the same dimensions: ###
# image_crops_list = []
# for file_index in range(len(crops_filenames_list)):
#     print(file_index)
#     ### Get current cropfilename: ###
#     current_crop_filename = crops_filenames_list[file_index]
#
#     ### Read crop file: ###
#     current_image_crop = mpimg.imread(current_crop_filename)
#
#     ### Turn crop image to tensor: ###
#     current_image_crop = numpy_to_torch(current_image_crop)
#     current_image_crop = current_image_crop.unsqueeze(0)  # [C,H,W] -> [B,C,H,W]
#
#     ### Upsample All to same size: ###
#     upsample_layer_bilinear = torch.nn.Upsample(size=final_crop_size, mode='bilinear')
#     upsample_layer_nearest = torch.nn.Upsample(size=final_crop_size, mode='nearest')
#     current_image_crop_upsampled = upsample_layer_bilinear(current_image_crop)
#
#     ### Append final outputs to list: ###
#     image_crops_list.append(current_image_crop_upsampled)
#
# crops_tensor_final = torch.cat(image_crops_list)
# crops_tensor_final = crops_tensor_final.unsqueeze(0)
# input_as_torch = crops_tensor_final /256
#
# # imshow_torch_video(input_as_torch)
#
# # # ###  ROI crop image | Get relevant for ceop video ###
# # # image_cropper = ImageCropper(torch_to_numpy(input_as_torch[0][0]))
# # # cropped_image = image_cropper.get_tensor_crop()
# # # print(cropped_image)
# #
# torch.cuda.empty_cache()
# list_of_denoised_video = []
# window_size_temporal = 6
# stride = 1
# #
# #
# # # ## For ROI crop image
# # # crop_size_x = [32, 128]
# # # crop_size_y = [32, 128]
# #
# #
# # # crop_area_tensor = input_as_torch[:, :, :,10:100, 10:100]
# # # for i in range(0, input_as_torch.shape[1] - 1 - window_size_temporal, stride):
# for i in range(0, 25 - 1 - window_size_temporal, stride):
#     torch.cuda.empty_cache()
#     relevant_frames_area = input_as_torch[:, i:i + window_size_temporal, :, :]
# #
# #     # #### ROI crop image | Get ROI cropped image ###
# #     # x1, y1, x2, y2 = image_cropper.get_coordinates_by_ROI_area()
# #
# #
# #
# #     # ### ROI crop image | Get closess 16 power ###
# #     # w, h = math.floor((x2 - x1) / 16) * 16, math.floor((y2 - y1) / 16) * 16
# #     # relevant_frames_and_crop_window = relevant_frames_area[:, :, :, y1:(y1 + h), x1:(x1 + w)]
# #
# #
# #     # ###ROI crop image| Perform denois###
# #     ### Move to same device
# #     # relevant_frames_and_crop_window.cuda()
# #     # denoised_reference, warped_frames = denoise_main(relevant_frames_and_crop_window)
# #
#     ### Perform denois (without sub crop in video ) ###
#     ### Move to same device
#     relevant_frames_area = relevant_frames_area.cuda()
#     denoised_reference, warped_frames = denoise_main(relevant_frames_area)
#
#     list_of_denoised_video.append(denoised_reference)
#     print(len(list_of_denoised_video))
# #
# denoised_tensor_video = torch.cat(list_of_denoised_video) / 256
# denoised_tensor_video = denoised_tensor_video / denoised_tensor_video.abs().max()
# #
# #
# #
# # #### Display the video ####
# imshow_torch_video(denoised_tensor_video, FPS=1)
# #
# #
# # ### Save video ####
# # VIDEO_NAME_TO_SAVE = "denoise_video.mp4"
# # video_torch_array_to_video(input_tensor=denoised_tensor_video, video_name=VIDEO_NAME_TO_SAVE)
# #
# #
# # print(list_of_denoised_video)
# # print(denoised_reference.shape)
# # print(warped_frames[0].shape)

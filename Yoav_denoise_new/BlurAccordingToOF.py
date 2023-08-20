"""
compute optical flow between 2 frames:
1) 2 consecutive frames in a video using video_mode
2) 2 of the same image, one of them being noised/transformed
"""
import os.path
import sys
import torch.nn
# sys.path.append('/home/yoav/rdnd')
# sys.path.append('/home/yoav')
# sys.path.append("/home/yoav/Anvil")

from RDND_proper.models.FlowFormer.core.FlowFormer.LatentCostFormer.transformer import FlowFormer
from RDND_proper.models.FlowFormer.configs.things import get_cfg as get_cfg_things
from RapidBase.TrainingCore.trainer import *
from RapidBase.TrainingCore.clip_gradients import *
from RapidBase.TrainingCore.datasets import get_default_IO_dict
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

### Import Models: ###
### Initialize Network State Dict: ###


def get_specific_model_stuff(base_name_restore, device):
    args = EasyDict()

    specific_restore = 'models/FlowFormer/check_points/things_kitti.pth'
    args.restore_ckpt = os.path.join(base_name_restore, specific_restore)
    args.name = 'flowformer'
    args.stage = 'kitti'
    args.validation = 'true'
    args.mixed_precision = True
    cfg = get_cfg_things()
    cfg.update(vars(args))
    flow_model = torch.nn.DataParallel(FlowFormer(cfg.latentcostformer), device_ids=[device]).to(device)
    state_dict = training_utils.load_state_dict_into_module(flow_model, torch.load(args.restore_ckpt), strict=False, loading_prefix=None)
    flow_model.load_state_dict(state_dict)
    flow_model.eval()
    return flow_model
# for i in range(net_input.shape[0]):
#     imshow_torch(torch.abs(flow[i, 0]))


def merge_to_video_same_folder(video_dir="/raid/yoav/temp_garbage/blur_according_to_flow", output_video_name="traj_video.mp4", output_dir_video="/raid/yoav/temp_garbage"):
    """

    Args:
        video_dir: where is it taking the frames from
        output_video_name: name for video file
        output_dir_video: dir to video file

    Returns:

    """
    fps = 6  # 5
    flags = "-y"  # overwrite
    # output_dir_video = "/raid/yoav/temp_garbage"
    # output_video_name = "traj_video.mp4"

    command = f'ffmpeg -framerate {fps} -pattern_type glob -i "{video_dir}/*.png" {flags} -vcodec libx264 -acodec aac {os.path.join(output_dir_video, output_video_name)}'
    os.system(command)


def get_mask_frame(folder, index):
    folders = glob(os.path.join(folder, '*'))
    f = folders[index]
    mask = torch.load(os.path.join(f, "car_biggest", "segmentation.pt"))
    return mask


def get_blurred_frames(frames, flows, device=0):
    """
    blur the frame i according to flow from i to i+1
    Args:
        frames: [B, 2, C, H, W]
        flows: [B-1, 2 ,H, W]
        sigma: blur sigma factor
        threshold: big enough flow to consider object moving
    Returns:
    """
    B, C, H, W = frames.shape
    blurred_frames = torch.zeros_like(frames)
    for i, (frame, flow) in enumerate(zip(frames[:-1], flows)):
        print('FRAME NUMBER: ' + str(i))
        mask = get_mask_frame("/raid/yoav/temp_garbage/temp_masking/output", i)
        # mask = torchvision.transforms.Resize((256, 256))(mask).repeat(3, 1, 1)
        mask = torch.nn.Upsample(size=(256,256))(mask.unsqueeze(0).float())
        average_optical_flow_x = ((torch.abs(flow[0:1].sum(0))) * mask.to(device)).sum() / mask.sum()  # average optical flow in segmentation mask per pixel
        average_optical_flow_y = ((torch.abs(flow[1:2].sum(0))) * mask.to(device)).sum() / mask.sum()  # average optical flow in segmentation mask per pixel
        # blur_factor = (mask.to(train_devices[0])).sum() / mask.sum()  # average optical flow in segmentation mask per pixel
        # print(blur_factor)
        # imshow_torch(mask * 1.0)
        # blur_frame = torchvision.transforms.functional.gaussian_blur(frame, sigma=[blur_factor.cpu()*sigma], kernel_size=[kernel_size, kernel_size])
        warp_object = Warp_Object()
        # blur_image_motion_blur_torch(frame.unsqueeze(0), average_optical_flow_x, average_optical_flow_y, N=10, warp_object=warp_object)
        blurry_image = torch.zeros_like(frame.unsqueeze(0))
        N = 50
        blur_factor = 1
        for blur_index in np.arange(N):
            # print(blur_index)
            blurry_image += warp_object.forward(frame.unsqueeze(0),
                                               -average_optical_flow_x*blur_factor*blur_index/N,
                                               -average_optical_flow_y*blur_factor*blur_index/N)/N

        # imshow_torch(frame/255)
        # imshow_torch(blurry_image/255)
        composite_image = frame.unsqueeze(0)
        composite_image[mask.bool().repeat(1,3,1,1)] = blurry_image[mask.bool().repeat(1,3,1,1)]
        # imshow_torch(composite_image/255)
        # imshow_torch(frame/255)
        # blurred_frames[i][mask] = blur_frame[mask]
        # imshow_torch(blurred_frames[i]/255)
        blurred_frames[i:i+1] = composite_image
    return blurred_frames


def special_function_for_nehoray():
    ### IO: ###
    IO_dict = get_default_IO_dict()
    train_devices = [0]
    test_devices = [0]
    IO_dict.device = torch.device(train_devices[0])
    print('Starting to train on GPUs: {}'.format(train_devices + test_devices))

    if os.getcwd() == '/home/yoav':
        print('running from screen')
        base_name_restore = 'rdnd'
    elif os.path.basename(os.getcwd()) == 'Deblur_Folder':  # todo: change this to fit your directories
        print('running from pycharm')
        base_name_restore = '../../../..' # todo: change this to fit your directories
    else:
        print('shouldnt get here')
        assert False, 'where the fuck are you running the script from'

    model = get_specific_model_stuff(base_name_restore, train_devices[0])

    """
    read video of shape [B, T, C, H, W] !!!!!!!!!
    """
    # load demo images
    video = read_video_default_torch(n=30, stride=1)
    video = torchvision.transforms.Resize((256, 256))(video)
    # net_input = torch.cat((video[0:-1], video[1:]), 1).to(train_devices[0])
    net_input = torch.cat((video[0:-1].unsqueeze(1), video[1:].unsqueeze(1)), 1).to(train_devices[0])

    with torch.no_grad():
        model_output = model(net_input)

    flows = model_output[0]

    blurred_frames = get_blurred_frames(video, flows, device=train_devices[0])
    # todo: save folder, frames and video
    folder = "/raid/yoav/temp_garbage/blur_according_to_flow"
    save_video_torch(blurred_frames, folder, flag_convert_bgr2rgb=True)
    merge_to_video_same_folder(video_dir=folder, output_video_name="blurred_car.avi", output_dir_video=os.path.dirname(folder))


if __name__ == "__main__":
    special_function_for_nehoray()


import time
import numpy as np
import io
import os
from PIL import Image
import cv2
from matplotlib import image as mpimg
import torchvision.transforms.functional as TF
from PIL import Image

import PARAMETER
import saverloader
import imageio.v2 as imageio
from nets.pips import Pips
import pip_utils.improc
import random
import glob
from pip_utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from RapidBase.import_all import *
from RDND_proper.models.tapnet.tapnet.tapnet_pipeline import inference

random.seed(125)
np.random.seed(125)
DEVICE = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

import matplotlib.patches as patches
# from bbox_util import *
from matplotlib.widgets import RectangleSelector

import sys
import torch.nn

sys.path.append(f"{PARAMETER.BASE_PROJECT}/Yoav_denoise_new")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Yoav_denoise_new/CoTracker")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Yoav_denoise_new/CoTracker/cotracker")
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/models/tapnet")
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/models")
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}models/tapnet/tapnet")
#
from RapidBase.import_all import *
from RDND_proper.models.tapnet.tapnet.utils import viz_utils
from Yoav_denoise_new.CoTracker.cotracker.predictor import CoTrackerPredictor



#ToDo make it general
# base_directory = "/home/nehoray/PycharmProjects/Shaback/yolo_tracking/examples/runs/track/exp/crops/2/2"
# files = sorted(glob(f"{base_directory}/*.png"))

base_directory = None
files = None
h = None
w = None
insert_any_mask_here = None

def pre_start_scrip(base_directory_for_global):
    global base_directory
    global files
    global insert_any_mask_here
    global h
    global w

    base_directory = base_directory_for_global
    files = sorted(glob(f"{base_directory}/*.png"))
    _, _, h, w = read_image_torch(files[0]).shape
    insert_any_mask_here = torch.zeros((h, w), dtype=torch.float32).transpose(0, 1)


def get_image_dimensions(image_path):
    image = Image.open(image_path)
    width, height = image.size
    return width, height

def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    global insert_any_mask_here
    x1, y1 = eclick.xdata.astype(int), eclick.ydata.astype(int)
    x2, y2 = erelease.xdata.astype(int), erelease.ydata.astype(int)
    plt.close()
    # print(x1, x2, y1, y2)
    # print("bla bla ")
    insert_any_mask_here[(np.minimum(x1, x2)):(np.maximum(x1, x2)), (np.minimum(y1, y2)):(np.maximum(y1, y2))] = 1.0
    insert_any_mask_here = insert_any_mask_here.transpose(0, 1)
    # plt.imshow(insert_any_mask_here.numpy().cpu().detach()); plt.show();




# Global variables
drawing = False
# ix, iy = -1, -1
# fx, fy = -1, -1
rectangle_roi = []
first_image = None


# todo : bug, only supports top right to bottom left roi
# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global fx, fy, ix, iy, drawing, rectangle_roi, first_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img = first_image.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0,255,0), 1)
            cv2.imshow('image', img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(first_image, (ix, iy), (x, y), (0,255,0), 1)
        rectangle_roi = [(ix, iy), (x, y)]

    # fx, fy = x, y


def get_mask_and_demo_video():
    global h
    global w
    global insert_any_mask_here
    global files
    global rectangle_roi
    global drawing
    global first_image  # Add this line

    starting_frame = 0
    files = files[starting_frame:]

    # load first image
    first_image = cv2.imread(files[0])
    first_image = cv2.normalize(first_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while(1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(first_image, 'Please select ROI using mouse', (10, 30), font, 0.4, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(first_image, 'and press enter to confirm', (10, 60), font, 0.4, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow('image', first_image)
        if cv2.waitKey(20) & 0xFF == 13:  # Press 'Enter' to stop
            break

    cv2.destroyAllWindows()
    ix, iy, fx, fy = rectangle_roi[0][0], rectangle_roi[0][1], rectangle_roi[1][0], rectangle_roi[1][1]
    insert_any_mask_here[np.minimum(ix, fx):np.maximum(ix, fx), np.minimum(iy, fy):np.maximum(iy, fy)] = 1
    # insert_any_mask_here = insert_any_mask_here.transpose(0, 1)
    print(insert_any_mask_here.sum())

    points_to_track = torch.argwhere(insert_any_mask_here)
    print(points_to_track.shape)
    normlized_points_to_track = points_to_track / torch.tensor([w, h]).unsqueeze(0)
    print(normlized_points_to_track.shape)
    return files, normlized_points_to_track.unsqueeze(0), points_to_track, h, w



def draw_trajectory_on_image(image, traj):
    """
    Args:
        image:
        traj: shape [N ,2]
    Returns:
    """
    image_pil = torchvision.transforms.ToPILImage()(image[0])
    # Now we can display the image and plot the points
    # plt.imshow(image_pil)
    for point in traj:
        # print(point[0], point[1])
        plt.scatter(point[0], point[1], s=100, c='r') # 'ro' means red circle
    plt.show()


# draw_trajectory_on_image(read_image_torch(files[0]).squeeze(), torch.tensor([1063, 578]).unsqueeze(0))
import numpy as np


def mask_builder(size, indices_list):
    masks = []

    for idx, indices in enumerate(indices_list):
        print(f"mask {idx}")
        mask = np.zeros(size, dtype=bool)
        xy_array = np.array(indices)
        indices
        mask[indices[1]: indices[3], indices[0]: indices[2]] = 1
        mask[:, 0] = 0
        mask[0, :] = 0
        masks.append(mask)

    return masks


def return_xyxy(indices):
    xy_array = np.array(indices)
    x_coords, y_coords = xy_array[:, :, 0], xy_array[:, :, 1]
    xyxy = np.zeros((xy_array.shape[0], 4))  # Create an array to store the results

    xyxy[:, 0] = np.min(x_coords, axis=1)  # Minimum x-coordinate
    xyxy[:, 1] = np.min(y_coords, axis=1)  # Minimum y-coordinate
    xyxy[:, 2] = np.max(x_coords, axis=1)  # Maximum x-coordinate
    xyxy[:, 3] = np.max(y_coords, axis=1)  # Maximum y-coordinate

    return xyxy


def find_min_max_coordinates(masks):
    masks = np.array(masks)
    xyxy_list =[]
    for mask in masks:
        indices = np.argwhere(mask == 1)

        min_x = np.min(indices[:, 1], axis=0)
        max_x = np.max(indices[:, 1], axis=0)
        min_y = np.min(indices[:, 0], axis=0)
        max_y = np.max(indices[:, 0], axis=0)
        xyxy_list.append([min_x, min_y, max_x, max_y])
    return xyxy_list


def run_model_batched(model, rgbs, N, sw, points_to_track=None, batch_size=100):#, points_to_track=torch.tensor([450.0 / 640.0, 100.0 / 360.0])):
    rgbs = rgbs.to(DEVICE).float()  # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B * S, C, H, W)
    H_, W_ = 360, 640
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)

    # point_to_track *= torch.tensor([W_, H_])
    # x = torch.ones((1, N), device=torch.device(DEVICE)) * point_to_track[0]
    # y = torch.ones((1, N), device=torch.device(DEVICE)) * point_to_track[1]
    # xy0 = torch.stack([x, y], dim=-1)  # B, N, 2
    # points_to_track = torch.ones((1, 20, 2))
    xy0_original = points_to_track.to(DEVICE) * torch.tensor([W_, H_]).unsqueeze(0).to(DEVICE) # of shape (1, N, 2)
    N = xy0_original.shape[1]# of shape (1, N, 2)
    reduce = False
    if reduce:
        N = 50
        xy0_original = xy0_original[:, :N, :]# of shape (1, N, 2)

    _, S, C, H, W = rgbs.shape
    trajs_e = torch.zeros((1, S, N, 2)).to(DEVICE)
    if N > batch_size:  # divide into batches of size batch_size
        for n in range(N // batch_size):
            print(f"reached {n} out of {N // batch_size}")
            xy0 = xy0_original
            low, high = n * batch_size, (n + 1) * batch_size
            for image_seq_id in range(int(ceil(rgbs.shape[1] / 7))):
                starting_seq = image_seq_id * 7
                try:
                    preds, preds_anim, vis_e, stats = model(xy0[:, low:high, :], rgbs[:, starting_seq:starting_seq+8], iters=6)
                    trajs_e[0, starting_seq:starting_seq+8, low:high, :] = preds[-1]
                    xy0[:, low:high, :] = trajs_e[:, starting_seq+7, low:high]
                except:  # meaning we didnt have 8 frames from starting_seq till the end
                    frames_left = rgbs.shape[1] - starting_seq
                    xy0[:, low:high, :] = trajs_e[:, -8, low:high]
                    preds, preds_anim, vis_e, stats = model(xy0[:, low:high, :], rgbs[:, -8:], iters=6)
                    trajs_e[0, -8:, low:high, :] = preds[-1]
                    # we dont have to update the initial points since this is the last iteration
    else:   # run on entire data
        xy0 = xy0_original
        for image_seq_id in range(int(ceil(rgbs.shape[1] / 7))):
            starting_seq = image_seq_id * 7
            try:
                preds, preds_anim, vis_e, stats = model(xy0[:, :, :], rgbs[:, starting_seq:starting_seq + 8], iters=6)
                trajs_e[0, starting_seq:starting_seq + 8, :, :] = preds[-1]
                xy0[:, :, :] = trajs_e[:, starting_seq + 7, :]
            except:  # meaning we didnt have 8 frames from starting_seq till the end
                frames_left = rgbs.shape[1] - starting_seq  # 40 - 35 = 5
                xy0 = trajs_e[:, -8, :]  # meaning i take the 35 traj.
                preds, preds_anim, vis_e, stats = model(xy0[:, :, :], rgbs[:, -8:], iters=6)
                trajs_e[0, -8:, :] = preds[-1]

    # draw_trajectory_on_image(rgbs[0, 0].detach().cpu(), trajs_e[0, 0, :, :].detach().cpu())

    if rgbs.shape[1] % 8 != 0: # clip values not calculated
        trajs_e = trajs_e[0, :-(rgbs.shape[1] % 8), :, :]

    trajs_e /= torch.tensor([W_, H_]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return trajs_e


def create_indexing_from_traj(trajs_e=None):
    """
    Args:
        traj: traj of shapes [1, S, N, 2], such that traj[0, i, j, c] is i frame, j point, channel x or y
    Returns:

    optical flow [S, H, W, 2], meaning we transform the N points into the relevent part of the H,W image and then fill the rest with zeros
    """

    if trajs_e is None:
        trajs_e = torch.load(f"{PARAMETER.BASE_PROJECT}/Yoav_denoise_new/output1/traj_demo_file.pt")
    # _, _ , origin_points, h, w = get_mask_and_demo_video()
    # trajs_e = torch.ones((1, 10, 1250, 2)).to(DEVICE) / 2
    trajs_e = trajs_e * torch.tensor([w, h]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    # trajs_e = trajs_e * torch.tensor([w, h]).unsqueeze(0).unsqueeze(0).to(DEVICE)
    origin_points = torch.argwhere(insert_any_mask_here)
    _, S, N, _ = trajs_e.shape
    # indexing_video = torch.zeros((S, w, h, 2))
    flows_video = torch.zeros((S, w, h, 2))
    for i in range(S):
        print(f"done creating flows with frame {i} out of {S}")
        for j, p in enumerate(origin_points):
            # flows_video[i, int(trajs_e[0, i, j, 0]), int(trajs_e[0, i, j, 1]), :] = trajs_e[0, i, j, :] - trajs_e[0, 0, j, :]
            flows_video[i, p[0], p[1], :] = trajs_e[0, i, j, :] - trajs_e[0, 0, j, :]

    flows_video = flows_video.transpose(1, 2)

    # for i in range(0, S):
    #     imshow_torch(flows_video[i, :, :, 0])

    return flows_video


def plot_traj_on_seq(trajs_e, video_seq=None, frame_count=64, hw=None):
    if video_seq is None:
        files, _, points, h, w = get_mask_and_demo_video()
        hw = [h, w]
        files = files[:frame_count]
        imgs = [read_image_torch(path) for path in files]
        video_seq = torch.cat(imgs)  # [T, 3, H, W]

    if hw is not None:
        trajs_e = trajs_e * torch.tensor([hw[1], hw[0]]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # for i, (im, points) in enumerate(zip(video_seq, trajs_e[0, :, :])):
        # if i % 8 == 0:
        # draw_trajectory_on_image(im.cpu().detach(), points.cpu().detach())


def warp_image_with_indexing_seq(video_seq=None, flows_video=None, frame_count=64, origin_points=None):
    """
    for each origin_point p, i want to go to video_seq[i] at indexing_video[i][p]
    Args:
        video_seq: [N, C, H, W]
        indexing_video: [N, H, W, 2]

    Returns:

    """
    warp_layer = Warp_Object()
    if video_seq is None:
        files, _, points, _, _ = get_mask_and_demo_video()
        files = files[:frame_count]
        imgs = [read_image_torch(path) for path in files]
        video_seq = torch.cat(imgs)  # [T, 3, H, W]

    background_image = video_seq[0].clone()
    averaged_image = torch.zeros((3, video_seq.shape[2], video_seq.shape[3]))  # [T, 3, H, W]
    warped_tensor = torch.zeros((frame_count, 3, video_seq.shape[2], video_seq.shape[3]))
    for i in range(frame_count):
        flow_x, flow_y = flows_video[i, :, :, 0], flows_video[i, :, :, 1]
        # x, y = indexing_video[i, :, :, 0], indexing_video[i, :, :, 1]
        warped_image = warp_layer.forward(video_seq[i].unsqueeze(0), flow_x.unsqueeze(0).unsqueeze(-1), flow_y.unsqueeze(0).unsqueeze(-1)).squeeze()
        warped_tensor[i] = warped_image
        # todo: interpolate or properly warp
        averaged_image += warped_image
    #
    # for i in range(frame_count):
    #     imshow_torch(warped_tensor[i]/255)
    #     # imshow_torch(indexing_video[i, :, :, 0:1])
    #     # imshow_torch(indexing_video[i, :, :, 1:2])
    # todo: get real location of stating flow

    final_image = background_image.clone()
    final_image[:, origin_points[:, 1], origin_points[:, 0]] = (averaged_image[:, origin_points[:, 1], origin_points[:, 0]] / frame_count)
    # imshow_torch(final_image / 255, title_str="averaged image")
    # imshow_torch(background_image / 255, title_str="original image")
    # imshow_torch((final_image - background_image), title_str="diff image")
    return final_image / 255


def main_old(frame_count, batch_size):
    # the idea in this file is to chain together pips from a long sequence, and return some visualizations

    exp_name = 'extracting_single_path_exp'  # (exp_name is used for logging notes that correspond to different runs)

    init_dir = 'reference_model'

    # ## choose hyps
    # B = 1
    # S = 50
    # N = 2  # number of points to track
    #
    # filenames = glob('/home/yoav/rdnd/models/pips_main/demo_images/*.jpg')
    # filenames = sorted(filenames)

    filenames, points_to_track, _, _, _ = get_mask_and_demo_video()
    # points_to_track = points_to_track[:, -50, :]
    B = 1
    S = frame_count

    if len(filenames) <= S:
        print(f"There are too few files, Less then{S} ")
        raise ValueError("TooFewFilesLess")

    N = points_to_track.shape[1]
    filenames = sorted(filenames)[:S]

    print('filenames', filenames)
    max_iters = len(filenames) // (S // 2) - 1  # run slightly overlapping subseqs

    log_freq = 1  # when to produce visualizations

    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    log_dir = 'logs_chain_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    model = Pips(stride=4).to(DEVICE)
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()


    # while global_step < max_iters:
    read_start_time = time.time()
    global_step += 1
    sw_t = pip_utils.improc.Summ_writer(
        writer=writer_t,
        global_step=global_step,
        log_freq=log_freq,
        fps=12,
        scalar_freq=int(log_freq / 2),
        just_gif=True)


    try:
        rgbs = []
        for s in range(S):
            fn = filenames[(global_step - 1) * S // 2 + s]
            if s == 0:
                print('start frame', fn)
            im = imageio.imread(fn)
            im = im.astype(np.uint8)
            try: # rgb image
                rgbs.append(torch.from_numpy(im).permute(2, 0, 1))
            except: # bw image
                rgbs.append(torch.from_numpy(im).unsqueeze(-1).repeat(1, 1, 3).permute(2, 0, 1))
        rgbs = torch.stack(rgbs, dim=0).unsqueeze(0)  # 1, S, C, H, W
        read_time = time.time() - read_start_time
        iter_start_time = time.time()

        with torch.no_grad():
            # trajs_e = run_model(model, rgbs, N, sw_t, points_to_track)
            print(points_to_track)
            trajs_e = run_model_batched(model, rgbs, N, sw_t, points_to_track, batch_size=batch_size)
            torch.save(trajs_e, f"{PARAMETER.BASE_PROJECT}/Yoav_denoise_new/output1/traj_demo_file.pt")
            return trajs_e, rgbs

        iter_time = time.time() - iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
            model_name, global_step, max_iters, read_time, iter_time))
    except FileNotFoundError as e:
        print('error', e)



    writer_t.close()



def main_new(frame_count, batch_size):
    # the idea in this file is to chain together pips from a long sequence, and return some visualizations

    exp_name = 'extracting_single_path_exp'  # (exp_name is used for logging notes that correspond to different runs)

    init_dir = 'reference_model'

    # ## choose hyps
    # B = 1
    # S = 50
    # N = 2  # number of points to track
    #
    # filenames = glob('/home/yoav/rdnd/models/pips_main/demo_images/*.jpg')
    # filenames = sorted(filenames)

    filenames, points_to_track, _, _, _ = get_mask_and_demo_video()
    # points_to_track = points_to_track[:, -50, :]
    B = 1
    S = frame_count
    N = points_to_track.shape[1]
    filenames = sorted(filenames)[:S]

    print('filenames', filenames)
    max_iters = len(filenames) // (S // 2) - 1  # run slightly overlapping subseqs

    log_freq = 1  # when to produce visualizations

    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    log_dir = 'logs_chain_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    rgbs = []
    for s in range(S):
        fn = filenames[(global_step - 1) * S // 2 + s]
        if s == 0:
            print('start frame', fn)
        im = imageio.imread(fn)
        im = im.astype(np.uint8)
        try: # rgb image
            rgbs.append(torch.from_numpy(im).permute(2, 0, 1))
        except: # bw image
            rgbs.append(torch.from_numpy(im).unsqueeze(-1).repeat(1, 1, 3).permute(2, 0, 1))
    rgbs = torch.stack(rgbs, dim=0).unsqueeze(0)  # 1, S, C, H, W

    # should get rgbs of shape [n, h, w] or [n, h, w, c] and points to track of shape [m, 3] where dimension [:, 0] is all zeros and the rest is x, y
    print(rgbs.shape, points_to_track.shape)
    new_points = np.zeros((points_to_track.squeeze().shape[0], 3))
    new_points[:, 1:] = points_to_track
    tracks, visibles = inference(rgbs.squeeze().permute(0, 2, 3, 1).cpu().detach().numpy(), new_points)


    writer_t.close()
    return tracks, rgbs



def convert_to_binary_masks(masks):
    binary_masks = [mask.astype(int) for mask in masks]
    return binary_masks


def cotracker_wrapper():
    cp = f"{PARAMETER.BASE_PROJECT}/Yoav_denoise_new/checkpoints/cotracker_stride_4_wind_8.pth"
    cotracker = CoTrackerPredictor(checkpoint=cp)
    cotracker = cotracker.to(DEVICE).eval()
    filenames, points_to_track, _, _, _ = get_mask_and_demo_video()
    # correct since co tracker gets not non normalized points
    points_to_track = points_to_track * torch.tensor([w, h]).unsqueeze(0)


    rgbs = []
    for fn in filenames:
        im = imageio.imread(fn)
        im = im.astype(np.uint8)
        try:  # rgb image
            rgbs.append(torch.from_numpy(im).permute(2, 0, 1))
        except:  # bw image
            rgbs.append(torch.from_numpy(im).unsqueeze(-1).repeat(1, 1, 3).permute(2, 0, 1))
    rgbs = torch.stack(rgbs, dim=0).unsqueeze(0)


    # should get video of shape [1, N, C, H, W]
    # video = rgbs.permute(0, 3, 1, 2)[None].to(DEVICE).float()
    video = rgbs.to(DEVICE).float()
    points_to_track = points_to_track.to(DEVICE).float()
    t = torch.randint(0, 1, (points_to_track.shape[1], 1)).unsqueeze(0).to(DEVICE)
    points_to_track = torch.cat((t, points_to_track), dim=-1)


    tracks, visibles = cotracker(video, queries=points_to_track)
    tracks = tracks / torch.tensor([h, w]).to(DEVICE)
    # back to expected shape, legacy
    # tracks of shape [num_points, frames, 2], (x, y) points
    # visibles of shape [num_points, frames] with bool for occluded
    points_to_track = points_to_track[0]
    tracks = np.array(tracks.permute(0, 2, 1, 3).cpu().detach())[0]
    visibles = np.array(visibles.permute(0, 2, 1).cpu().detach())[0]
    rgbs = np.array(rgbs.cpu().detach())

    return rgbs, tracks, points_to_track, visibles



def main_interface_cotracker(base_directory_of_images, base_output,output_full_frame="frames",output_seg="segmentation", output_crop="crop"):
    pre_start_scrip(base_directory_of_images)
    output_full_frame = f"{base_output}/{output_full_frame}"
    output_seg = f"{base_output}/{output_seg}"
    output_crop = f"{base_output}/{output_crop}"
    try:
        if not os.path.exists(base_output):
            os.makedirs(base_output)
        if not os.path.exists(output_full_frame):
            os.makedirs(output_full_frame)
        if not os.path.exists(output_seg):
            os.makedirs(output_seg)
        if not os.path.exists(output_crop):
            os.makedirs(output_crop)
    except Exception as e:
        raise RuntimeError(f"got Exception - e - {e}")

    rgbs, trajs_e, query_points, visibles = cotracker_wrapper()
    # visualize_all(frames, tracks, visibles)
    # trajs_e of shape [1, frames, points, 2], rgbs of shape [1, frames, c, h, w]
    # trajs_e = torch.tensor(trajs_e)
    # trajs_e = trajs_e[:, :-T+1]
    # trajs_e_not_normalized = trajs_e.cpu().squeeze()
    trajs_e_not_normalized = torch.tensor(trajs_e) * torch.tensor([w, h]).unsqueeze(0)
    trajs_e_not_normalized = trajs_e_not_normalized.to(torch.int32)

    # todo : make it better
    # get rid of ending zeros
    # zero_count = (trajs_e_not_normalized[0, :, 0] == 0).sum()
    # trajs_e_not_normalized = trajs_e_not_normalized[:, :-zero_count, :]

    roi_xyxy_list = [[min(traj[:, 0]), min(traj[:, 1]), max(traj[:, 0]), max(traj[:, 1])] for traj in trajs_e_not_normalized]

    # roi_xyxy_list = [[min(traj[:, 1]), min(traj[:, 0]), max(traj[:, 1]), max(traj[:, 0])] for traj in trajs_e_not_normalized]
    mask_list = mask_builder((h, w), roi_xyxy_list)

    images_list = rgbs[0].transpose(0, 2, 3, 1)
    center_points = []
    #### Plot the desired crop base on the original points and the optical flow during the images ####
    for hi in range(trajs_e.shape[1]):

        ###Actual display
        # plt.figure()
        # plt.title(hi)
        # plt.imshow(images_list[hi, roi_xyxy_list[hi][1]:roi_xyxy_list[hi][3], roi_xyxy_list[hi][0]:roi_xyxy_list[hi][2]])
        plt.show()

        ###Save as pt file

        torch.save(mask_list[hi], f"{output_seg}/{hi}.pt")

        ### Save as jpg - full frame
        pil_image = TF.to_pil_image(images_list[hi, roi_xyxy_list[hi][1]:roi_xyxy_list[hi][3], roi_xyxy_list[hi][0]:roi_xyxy_list[hi][2]])
        pil_image.save(f"{output_crop}/{hi}.png")

        ### Save as jpg - crop
        pil_image = TF.to_pil_image(images_list[hi])
        pil_image.save(f"{output_full_frame}/{hi}.png")

        # calculate the center of bounding box
        roi = roi_xyxy_list[hi]
        center_point = [(roi[0].item() + roi[2].item()) / 2, (roi[1].item() + roi[3].item()) / 2]
        center_points.append(center_point)
        print(f"Frame {hi}: Center = {center_point}")


    return center_points
    # insert ecc
    #### Avarge image and other
    # flows_video = create_indexing_from_traj()
    # origin_points = torch.argwhere(insert_any_mask_here)
    # averaged_image = warp_image_with_indexing_seq(video_seq=rgbs.squeeze().type(torch.float32), flows_video=flows_video, frame_count=rgbs.shape[0], origin_points=origin_points)
    # imshow_torch(averaged_image, title_str="averaged_img")




# if __name__ == '__main__':
#     #ToDo there are some limitations on selecting roi
#     base_output = "/home/nehoray/PycharmProjects/Shaback/Yoav_denoise_new/output1"
#     # base_directory_of_images = "/home/nehoray/PycharmProjects/Shaback/output/video_example_from_security_camera/crops/2/7"
#     base_directory_of_images = "/home/nehoray/PycharmProjects/Shaback/remove_object/temp_directory_for_desired_frame_to_remove_object"
#     base_directory_of_images = "/home/nehoray/PycharmProjects/Shaback/output/toem/crop"
#     base_directory_of_images = "/home/nehoray/PycharmProjects/Shaback/Inpaint-Anything/data/images/drone_follows_a_sports_car_from_left_"
#     base_directory_of_images = "/home/nehoray/PycharmProjects/Shaback/Inpaint-Anything/data/images/drone_follows_a_sports_car_from_right_and_back"
#     base_directory_of_images = "/home/nehoray/PycharmProjects/Shaback/Inpaint-Anything/data/images/car_road_5"
#     # try:
#     center_point_of_bounding_box = main_interface_cotracker(base_directory_of_images, base_output)
#     print(center_point_of_bounding_box)
#     # except Exception as e:
#     #     print(f"Got an error {e}")


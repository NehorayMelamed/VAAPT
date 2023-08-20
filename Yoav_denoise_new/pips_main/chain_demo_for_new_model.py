import time
import numpy as np
import io
import os
from PIL import Image
import cv2
from matplotlib import image as mpimg
import torchvision.transforms.functional as TF
from PIL import Image
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

import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

import matplotlib.patches as patches
# from bbox_util import *
from matplotlib.widgets import RectangleSelector



#ToDo make it general
# base_directory = r"/home/dudy/Nehoray/Yoav_denoise_new/pips_main/input_data/png_new/scene_0"
base_directory = r"/home/nehoray/PycharmProjects/Shaback/yolo_tracking/examples/runs/track/exp/crops/2/5"



# files = sorted(glob(f"{base_directory}/*.png"))
files = sorted(glob(f"{base_directory}/*.png"))

# Get the last directory name
last_directory_name = os.path.basename(base_directory)



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


# def get_mask_and_demo_video():
#     global h
#     global w
#     global insert_any_mask_here
#     global files
#     # files = sorted(glob("/home/dudy/Nehoray/Yoav_denoise_new/pips_main/input_data/demo_images/*.jpg"))
#     # files = sorted(glob("/home/dudy/Nehoray/SHABACK_POC_NEW/util/output_directory/scene_4_images/*.jpg"))
#     # files = sorted(glob("/home/dudy/Nehoray/SHABACK_POC_NEW/util/output_directory/scene_12_images/*.jpg"))
#     # files = sorted(glob("/home/dudy/Nehoray/Yoav_denoise_new/pips_main/input_data/scene_14_o_images/*.jpg"))
#     # files = sorted(glob("/home/dudy/Nehoray/Yoav_denoise_new/pips_main/input_data/scene_8_o_images/*.jpg"))
#     # files = sorted(glob("/home/dudy/Nehoray/Yoav_denoise_new/pips_main/input_data/scene_18_o_full_size_images/*.jpg"))
#     # files = sorted(glob("/home/dudy/Nehoray/Yoav_denoise_new/pips_main/input_data/scene_107_images/*.jpg"))
#     # files = sorted(glob("/home/dudy/Nehoray/SHABACK_POC_NEW/Omer/output/vid_18/images/*.jpg"))
#     # files = sorted(glob("/home/dudy/Nehoray/Yoav_denoise_new/pips_main/input_data/demo_sc_0/*.jpg"))
#
#
#     # files = sorted(glob("/home/dudy/Nehoray/SHABACK_POC_NEW/ecc_segmantion_layer/data/ar_shaback_4/*.jpg"))
#     # files = path_get_files_from_folder(string_pattern_to_search="*raw_image.jpg",path="/home/dudy/Nehoray/SHABACK_POC_NEW/ecc_segmantion_layer/data/vehicle_shaback/vehicle_shaback_1/raw_image*.jpg")
#
#
#     # w, h = get_image_dimensions(files[0])
#     starting_frame = 0
#     files = files[starting_frame:]
#
#     # load first image
#     first_image = read_image_cv2(files[0])/255
#
#     fig, ax = plt.subplots()
#     ax.imshow(first_image)
#
#     rs = RectangleSelector(ax, line_select_callback,
#                            useblit=True,
#                            button=[1, 3],  # disable middle button
#                            minspanx=5, minspany=5,
#                            spancoords='pixels',
#                            interactive=True)
#
#     plt.show()
#     # plt.close()
#
#     print(insert_any_mask_here.sum())
#     points_to_track = torch.argwhere(insert_any_mask_here)
#     print(points_to_track.shape)
#     normlized_points_to_track = points_to_track / torch.tensor([w, h]).unsqueeze(0)
#     print(normlized_points_to_track.shape)
#     return files, normlized_points_to_track.unsqueeze(0), points_to_track, h, w



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
        cv2.imshow('image', first_image)
        if cv2.waitKey(20) & 0xFF == 27: # Press ESC to stop
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



def create_indexing_from_traj(trajs_e=None):
    """
    Args:
        traj: traj of shapes [1, S, N, 2], such that traj[0, i, j, c] is i frame, j point, channel x or y
    Returns:

    optical flow [S, H, W, 2], meaning we transform the N points into the relevent part of the H,W image and then fill the rest with zeros
    """

    if trajs_e is None:
        trajs_e = torch.load("/home/nehoray/PycharmProjects/Shaback/Yoav_denoise_new/output1/traj_demo_file.pt")
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
    new_points = (points_to_track - 0.5) * 2
    p = np.zeros((points_to_track.squeeze().shape[0], 3))
    p[:, 1:] = new_points.squeeze()
    tracks, visibles = inference(rgbs.squeeze().permute(0, 2, 3, 1).cpu().detach().numpy(), p)


    writer_t.close()
    return tracks, rgbs



def convert_to_binary_masks(masks):
    binary_masks = [mask.astype(int) for mask in masks]
    return binary_masks

if __name__ == '__main__':
    base_output = f"/home/nehoray/PycharmProjects/Shaback/Yoav_denoise_new/output1"
    output_full_frame = f"{base_output}/full_frame"
    output_seg = f"{base_output}/segmentation"
    output_crop = f"{base_output}/crop"

    if not os.path.exists(base_output):
        os.makedirs(base_output)
    if not os.path.exists(output_full_frame):
        os.makedirs(output_full_frame)
    if not os.path.exists(output_seg):
        os.makedirs(output_seg)
    if not os.path.exists(output_crop):
        os.makedirs(output_crop)


    dont_change = 8
    T = 1
    frames = T*dont_change
    batch_size = 500  # multiple of number of points

    # trajs_e, rgbs = main_new(frames, batch_size)
    trajs_e, rgbs = main_new(frames, batch_size)
    # trajs_e of shape [1, frames, points, 2], rgbs of shape [1, frames, c, h, w]
    trajs_e = torch.tensor(trajs_e)
    trajs_e = trajs_e.permute(1, 0, 2).unsqueeze(0)
    trajs_e_not_normalized = trajs_e.cpu().squeeze().unsqueeze(0).to(torch.int32)


    # todo : make it better
    # get rid of ending zeros
    # zero_count = (trajs_e_not_normalized[0, :, 0] == 0).sum()
    # trajs_e_not_normalized = trajs_e_not_normalized[:, :-zero_count, :]

    # roi_xyxy_list = [[min(traj[:, 1]), min(traj[:, 0]), max(traj[:, 1]), max(traj[:, 0])] for traj in trajs_e_not_normalized.squeeze()]
    roi_xyxy_list = [[min(traj[:, 0]), min(traj[:, 1]), max(traj[:, 0]), max(traj[:, 1])] for traj in trajs_e_not_normalized.squeeze()]
    roi_xyxy_list = roi_xyxy_list[1:]

    mask_list = mask_builder((h, w), roi_xyxy_list)

    images_list = torch_to_numpy(rgbs[0])
    #### Plot the desired crop base on the original points and the optical flow during the images ####
    for hi in range(0, trajs_e.shape[1]-1):

        di = hi + 1
        ###Actual display
        # plt.figure()
        # plt.title(hi)
        # plt.imshow(images_list[hi, roi_xyxy_list[hi][1]:roi_xyxy_list[hi][3], roi_xyxy_list[hi][0]:roi_xyxy_list[hi][2]])
        plt.show()

        ###Save as pt file

        torch.save(mask_list[hi], f"{output_seg}/{hi}.pt")

        ### Save as jpg - full frame
        pil_image = TF.to_pil_image(images_list[di, roi_xyxy_list[hi][1]:roi_xyxy_list[hi][3], roi_xyxy_list[hi][0]:roi_xyxy_list[hi][2]])
        pil_image.save(f"{output_crop}/{hi}.png")

        ### Save as jpg - crop
        pil_image = TF.to_pil_image(images_list[hi])
        pil_image.save(f"{output_full_frame}/{hi}.png")

    # insert ecc

    # draw_trajectory_on_image(read_image_torch("/home/dudy/Nehoray/SHABACK_POC_NEW/Omer/output/vid_18/images/frame_0011.jpg"), 770*trajs_e[0, 0])

    # flows_video = create_indexing_from_traj()
    # origin_points = torch.argwhere(insert_any_mask_here)
    # averaged_image = warp_image_with_indexing_seq(video_seq=rgbs.squeeze().type(torch.float32), flows_video=flows_video, frame_count=frames, origin_points=origin_points)
    # imshow_torch(averaged_image, title_str="averaged_img")
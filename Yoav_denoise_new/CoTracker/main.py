import sys

import torch.nn

sys.path.append("/home/yoav/rdnd")
sys.path.append("/home/yoav")
sys.path.append("/home/yoav/rdnd/models/tapnet/tapnet")
sys.path.append("/home/yoav/rdnd/models/tapnet")

from RapidBase.import_all import *
from models.tapnet.tapnet.utils import viz_utils
from models.CoTracker.cotracker.predictor import CoTrackerPredictor

DEVICE = 0


def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.astype(np.float32)
  frames = frames / 255 * 2 - 1
  return frames


def sample_random_points(frame_max_idx, height, width, num_points):
  """Sample random points with (time, height, width) order."""
  y = np.random.randint(0, height, (num_points, 1))
  x = np.random.randint(0, width, (num_points, 1))
  t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
  points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
  return points


def get_frames():
    frames = read_video_defualt_torch(n=90, stride=1).permute(0, 2, 3, 1)
    frames = torch.nn.Upsample(scale_factor=0.25)(frames.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    # two_frames = orig_frames[:2]
    n, h, w, c = frames.shape

    # orig_frames = # [n, h, w, c] or [n, h, w]
    width, height = frames.shape[1:3]
    num_points = 500
    query_points = sample_random_points(0, height, width, num_points)
    query_points = torch.from_numpy(query_points)[None].to(DEVICE).float()

    to_filter = True
    if to_filter:
        # filter car location
        # filter x axis points
        query_points = query_points[query_points[:, :, 1] > 350].unsqueeze(0)
        query_points = query_points[query_points[:, :, 1] < 400].unsqueeze(0)
        # filter y axis points
        # query_points = query_points[query_points[:, :, 2] < 150].unsqueeze(0)
        query_points = query_points[query_points[:, :, 2] > 50].unsqueeze(0)

    return frames, query_points


def main():
    frames, query_points = get_frames()

    cp = "/home/yoav/rdnd/models/CoTracker/checkpoints/cotracker_stride_4_wind_8.pth"
    cotracker = CoTrackerPredictor(checkpoint=cp)
    cotracker = cotracker.to(DEVICE).eval()
    # should get video of shape [1, N, C, H, W]
    video = frames.permute(0, 3, 1, 2)[None].to(DEVICE).float()
    tracks, visibles = cotracker(video, queries=query_points)

    # back to expected shape, legacy
    # tracks of shape [num_points, frames, 2], (x, y) points
    # visibles of shape [num_points, frames] with bool for occluded
    query_points = np.array(query_points.cpu().detach())[0]
    tracks = np.array(tracks.permute(0, 2, 1, 3).cpu().detach())[0]
    visibles = np.array(visibles.permute(0, 2, 1).cpu().detach())[0]
    frames = np.array(frames.cpu().detach())

    return frames, tracks, query_points, visibles


def draw_trajectory_on_image(image, traj):
    """
    Args:
        image:
        traj: shape [N ,2]
    Returns:
    """
    image_pil = torchvision.transforms.ToPILImage()(image)
    # Now we can display the image and plot the points
    plt.imshow(image_pil)
    for point in traj:
        # print(point[0], point[1])
        plt.scatter(point[0], point[1], s=10, c='r') # 'ro' means red circle
    plt.show()


def save_trajectory_image(image, traj, path="/raid/yoav/temp_garbage/traj_videos", index=0):
    """
    Args:
        image:
        traj: shape [N ,2]
    Returns:
    """
    image_pil = torchvision.transforms.ToPILImage()(image)
    # Now we can display the image and plot the points
    plt.figure()
    plt.imshow(image_pil)
    for point in traj:
        # print(point[0], point[1])
        plt.scatter(point[0], point[1], s=10, c='r')  # 'ro' means red circle
    plt.savefig(os.path.join(path, f"{str(index).zfill(3)}.png"))


def plot_traj_on_seq(trajs_e, video_seq, hw=None):
    if hw is not None:
        trajs_e = trajs_e * torch.tensor([hw[1], hw[0]]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(DEVICE)

    for i, (im, points) in enumerate(zip(video_seq, trajs_e[0, :, :])):
        draw_trajectory_on_image(im.cpu().detach(), points.cpu().detach())


def save_trajectory_on_video(trajs_e, video_seq, hw=None, save_path="/raid/yoav/temp_garbage/traj_videos_2"):
    os.makedirs(save_path, exist_ok=True)

    if hw is not None:
        trajs_e = trajs_e * torch.tensor([hw[1], hw[0]]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(DEVICE)

    for i, (im, points) in enumerate(zip(video_seq, trajs_e[0, :, :])):
        save_trajectory_image(im.cpu().detach(), points.cpu().detach(), path=save_path, index=i)

    merge_to_video_same_folder(save_path)


def merge_to_video_same_folder(video_dir="/raid/yoav/temp_garbage/traj_videos"):
    fps = 6  # 5
    crf = 20
    flags = "-y"  # overwrite
    output_dir_video = "/raid/yoav/temp_garbage"
    output_video_name = "traj_video.mp4"

    command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx264 -x264-params "crf={}" {}'.format(
        fps, video_dir, flags, crf, os.path.join(output_dir_video, output_video_name))
    os.system(command)


def visualize_all(frames, tracks, visibles):
    use_defulat_vis = True
    if use_defulat_vis:
        # Visualize sparse point tracks
        # tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
        # video should be numpy [n, h, w, 3]
        video_viz = viz_utils.paint_point_track(frames, tracks, visibles)
        folder = "/raid/yoav/temp_garbage/traj_videos_4"
        save_video_torch(torch.tensor(video_viz).permute(0, 3, 1, 2), path=folder, flag_convert_bgr2rgb=True,
                         flag_scale_by_255=True)
        merge_to_video_same_folder(folder)
    else:
        tracks_ready = torch.tensor(tracks).permute(1, 0, 2).unsqueeze(0)
        visibles = np.asarray(visibles)
        visibles_ready = torch.tensor(visibles).permute(1, 0).unsqueeze(0)

        video_seq = torch.tensor(frames).permute(0, 3, 1, 2)
        video_seq -= video_seq.min()
        video_seq /= video_seq.max()

        # plot_traj_on_seq(tracks_ready, video_seq, 5, None)
        save_trajectory_on_video(tracks_ready, video_seq, save_path="/raid/yoav/temp_garbage/traj_videos_4")
        # todo: create a pipeline that knows what to do with occlusions:
        #       1) dont count them in averaging


frames, tracks, query_points, visibles = main()
visualize_all(frames, tracks, visibles)

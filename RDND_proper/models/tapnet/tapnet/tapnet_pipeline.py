import sys
import os

import torch

import PARAMETER
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from RapidBase.import_all import *
import haiku as hk
import jax
import mediapy as media
import numpy as np
import tree

sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/models/tapnet/tapnet")
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/models/tapnet")
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/models")
import RDND_proper.models.tapnet.tapnet.tapir_model
from RDND_proper.models.tapnet.tapnet.configs import tapir_config
from RDND_proper.models.tapnet.tapnet.utils import transforms
from RDND_proper.models.tapnet.tapnet.utils import viz_utils
from RDND_proper.models.tapnet.tapnet.tapir_model import TAPIR


# Build Model

def build_model(frames, query_points):
  """Compute point tracks and occlusions given frames and query points."""
  model = TAPIR()
  outputs = model(
      video=frames,
      is_training=False,
      query_points=query_points,
      query_chunk_size=64,
  )
  return outputs

# Utility Functions

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


def postprocess_occlusions(occlusions, expected_dist):
  """Postprocess occlusions to boolean visible flag.

  Args:
    occlusions: [num_points, num_frames], [-inf, inf], np.float32
    expected_dist: [num_points, num_frames], [-inf, inf], np.float32

  Returns:
    visibles: [num_points, num_frames], bool
  """
  # visibles = occlusions < 0
  visibles = (1 - jax.nn.sigmoid(occlusions)) * (1 - jax.nn.sigmoid(expected_dist)) > 0.5
  return visibles


def  inference(frames, query_points):
    gpus = jax.devices('gpu')
    # gpus = jax.devices('cpu')
    defualt_device = gpus[0]
    # defualt_device = gpus[0]
    jax.default_device = defualt_device
    jax.config.update("jax_default_device", defualt_device)
    # Load Checkpoint
    checkpoint_path = f"{PARAMETER.RDND_BASE_PATH}/models/tapnet/tapnet/checkpoints/tapir_checkpoint.npy"
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']

    model = hk.transform_with_state(build_model)
    model_apply = jax.jit(model.apply, device=defualt_device)

    """Inference on one video.
    
    Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8
    query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]
    
    Returns:
    tracks: [num_points, 3], [-1, 1], [t, y, x]
    visibles: [num_points, num_frames], bool
    """
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    num_frames, height, width = frames.shape[0:3]
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension

    # Model inference
    rng = jax.random.PRNGKey(42)
    outputs, _ = model_apply(params, state, rng, frames, query_points)
    outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
    tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']

    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles


def sample_random_points(frame_max_idx, height, width, num_points):
  """Sample random points with (time, height, width) order."""
  y = np.random.randint(0, height, (num_points, 1))
  x = np.random.randint(0, width, (num_points, 1))
  t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
  points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
  return points


# # Load an Examplar Video
#
# video_id = 'horsejump-high'  # @param
#
# ds, ds_info = tfds.load('davis', split='validation', with_info=True)
# davis_dataset = tfds.as_numpy(ds)
#
# for sample in davis_dataset:
#   video_name = sample['metadata']['video_name'].decode()
#   if video_name == video_id:
#     break  # stop at particular video id
#
#
# # @title Predict Sparse Point Tracks {form-width: "25%"}
# #
# resize_height = 256  # @param {type: "integer"}
# resize_width = 256  # @param {type: "integer"}
# num_points = 50  # @param {type: "integer"}
# #
# orig_frames = sample['video']['frames']

# orig_frames = read_5_frames_defualt_torch().permute(0, 2, 3, 1)
# n, h, w, c = orig_frames.shape
#
# # orig_frames = # [n, h, w, c] or [n, h, w]
# height, width = orig_frames.shape[1:3]
# frames = media.resize_video(orig_frames, (resize_height, resize_width))
# query_points = sample_random_points(0, frames.shape[1], frames.shape[2], num_points)
#
# # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# # tracks of shape [num_points, frames, 2], (x, y) points
# # visibles of shape [num_points, frames] with bool for occluded
# tracks, visibles = inference(frames, query_points)
#
# # todo: we already have a good pipeline that receives [1, #frames, #points, 2]
# #       so we can just reshape and pass on
#
# tracks_ready = torch.tensor(tracks).permute(1, 0, 2).unsqueeze(0)
# visibles = np.asarray(visibles)
# visibles_ready = torch.tensor(visibles).permute(1, 0).unsqueeze(0)
#
#
# DEVICE = 15
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


def plot_traj_on_seq(trajs_e, video_seq, frame_count=64, hw=None):
    if hw is not None:
        trajs_e = trajs_e * torch.tensor([hw[1], hw[0]]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(DEVICE)

    for i, (im, points) in enumerate(zip(video_seq, trajs_e[0, :, :])):
        draw_trajectory_on_image(im.cpu().detach(), points.cpu().detach())

#
# video_seq = torch.tensor(frames).permute(0, 3, 1, 2)
# video_seq -= video_seq.min()
# video_seq /= video_seq.max()
#
# plot_traj_on_seq(tracks_ready, video_seq, 5, None)
# todo: create a pipeline that knows what to do with occlusions:
#       1) dont count them in averaging

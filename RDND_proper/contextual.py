import torch
import torch.nn.functional as F

# from .config import LOSS_TYPES

__all__ = ['contextual_loss', 'contextual_bilateral_loss']


def contextual_loss(x: torch.Tensor,
                    y: torch.Tensor,
                    band_width: float = 0.5,
                    loss_type: str = 'cosine'):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.
    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """

    assert x.size() == y.size(), 'input tensor must have the same size.'
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    N, C, H, W = x.size()

    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(x, y)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(x, y)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(x, y)

    dist_tilde = compute_relative_distance(dist_raw)
    cx = compute_cx(dist_tilde, band_width)
    cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5))  # Eq(5)

    return cx_loss


# TODO: Operation check
def contextual_bilateral_loss(x: torch.Tensor,
                              y: torch.Tensor,
                              weight_sp: float = 0.1,
                              band_width: float = 1.,
                              loss_type: str = 'cosine'):
    """
    Computes Contextual Bilateral (CoBi) Loss between x and y,
        proposed in https://arxiv.org/pdf/1905.05169.pdf.
    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper).
    k_arg_max_NC : torch.Tensor
        indices to maximize similarity over channels.
    """

    assert x.size() == y.size(), 'input tensor must have the same size.'
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    # spatial loss
    grid = compute_meshgrid(x.shape).to(x.device)
    dist_raw = compute_l2_distance(grid, grid)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_sp = compute_cx(dist_tilde, band_width)

    # feature loss
    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(x, y)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(x, y)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(x, y)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_feat = compute_cx(dist_tilde, band_width)

    # combined loss
    cx_combine = (1. - weight_sp) * cx_feat + weight_sp * cx_sp

    k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)

    cx = k_max_NC.mean(dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5))

    return cx_loss


def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
    return cx


def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde


def compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                           y_normalized)  # (N, H*W, H*W)

    # convert to distance
    dist = 1 - cosine_sim

    return dist


def patchify_torch(input_tensor, patch_size, stride=None):
    if stride is None:
        stride = patch_size

    ### Get Shape: ###
    B,C,H,W = input_tensor.shape

    ### Patchify by adding doubling the amount of spatial dimensions (B,C,H,W) -> (B,C,H/P,W/P,P,P): ###
    input_tensor_patches = input_tensor.unfold(2, patch_size, stride).unfold(3, patch_size, stride)

    ### Stack N_patches on to Batch dimension (later on unpack it!!!) (B,C,H/P,W/P,P,P) -> (B*H*W/P^2,C,P,P): ###
    input_tensor_patches = input_tensor_patches.reshape(B, C, -1, patch_size, patch_size).reshape(-1, C ,patch_size, patch_size)

    return input_tensor_patches

def fuck_it():
    x = torch.rand(6, 3, 192, 192)
    y = torch.rand(6, 3, 192, 192)
    N, C, H, W = x.size()

    # Tensor -> patches
    size = 16  # patch size
    stride = 16  # patch stride
    x_patches = x.unfold(2, size, stride).unfold(3, size, stride)
    x_patches = x_patches.reshape(6, 3, -1, 16, 16).reshape(6, -1, 16, 16)
    y_patches = y.unfold(2, size, stride).unfold(3, size, stride)
    y_patches = y_patches.reshape(6, 3, -1, 16, 16).reshape(6, -1, 16, 16)

    x_patches_vec = x_patches.view(N, 432, -1)
    y_patches_vec = y_patches.view(N, 432, -1)
    x_patches_s = torch.sum(x_patches_vec ** 2, dim=1)
    y_patches_s = torch.sum(y_patches_vec ** 2, dim=1)

    y_vec = y.view(N, C, -1)
    A = y_vec.transpose(1, 2).repeat(1, 1, 144) @ x_patches.flatten(-2, -1)
    dist = y_patches_s - 2 * A + x_patches_s.transpose(0, 1)
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.)

    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1)
    y_s = torch.sum(y_vec ** 2, dim=1)
    #

def compute_cosine_distance_patches(x, y, patch_size):

    ### patchify x,y (B,C,H,W) -> (B*H*W/P^2,C,P,P): ###
    x_patches = patchify_torch(x, patch_size, stride=1)
    y_patches = patchify_torch(y, patch_size, stride=1)
    BP,CP,HP,WP = x_patches.shape
    B,C,H,W = x.shape

    ### Flatten all patch pixels (B*H*W/P^2,C,P,P) -> (B*H*W/P^2,C,P^2): ###
    x_patches_vec = x_patches.view(BP,CP,-1)  #(B*H*W/P^2,C,P^2)
    y_patches_vec = y_patches.view(BP,CP,-1)  #(B*H*W/P^2,C,P^2)
    x_patches_s = torch.sum(x_patches_vec ** 2, dim=1)  #(B*H*W,P^2)     #sum over channels
    y_patches_s = torch.sum(y_patches_vec ** 2, dim=1)  #(B*H*W,P^2)

    x_vec = x.view(B, C, -1)  #(B,C,H*W)
    y_vec = y.view(B, C, -1)  #(B,C,H*W)
    x_s = torch.sum(x_vec ** 2, dim=1)  #(B,H*W)
    y_s = torch.sum(y_vec ** 2, dim=1)  #(B,H*W)

    ### Calculate Distance Matrix: ###
    y_transposed = y_vec.transpose(1, 2)  #(B,C,H*W) -> (B,H*W,C)
    y_transposed_stacked = y_transposed.repeat(H*W/patch_size**2, 1, 1)  #(B,H*W,C) -> (B*H*W/P^2,H*W,C)
    A = torch.bmm(y_transposed_stacked, x_patches_vec)   #(B*H*W/P^2,H*W,C) X (B*H*W/P^2,C,P^2)  -> (B*H*W/P^2,H*W,P^2)

    y_vec = y.view(N, C, -1)
    A = y_vec.transpose(1, 2).repeat(1, 1, 144) @ x_patches.flatten(-2, -1)
    dist = y_patches_s - 2 * A + x_patches_s.transpose(0, 1)
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.)

    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                           y_normalized)  # (N, H*W, H*W)

    # convert to distance
    dist = 1 - cosine_sim

    return dist

# TODO: Considering avoiding OOM.
def compute_l1_distance(x: torch.Tensor, y: torch.Tensor):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)

    dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    dist = dist.sum(dim=1).abs()
    dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
    dist = dist.clamp(min=0.)

    return dist


# TODO: Considering avoiding OOM.
def compute_l2_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1)
    y_s = torch.sum(y_vec ** 2, dim=1)

    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s - 2 * A + x_s.transpose(0, 1)
    dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
    dist = dist.clamp(min=0.)

    return dist


def compute_meshgrid(shape):
    N, C, H, W = shape
    rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
    cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

    return feature_grid

import torch
import time


import contextual_loss as cl

# input features
img1 = torch.rand(1, 3, 92, 92)
img2 = torch.rand(1, 3, 92, 92)

# contextual loss
criterion = cl.ContextualLoss()

start_time = time.time()

loss = criterion(img1, img2)

print("--- %s seconds ---" % (time.time() - start_time))

# functional call
loss = cl.functional.contextual_loss(img1, img2, band_width=0.1, loss_type='cosine2')

# comparing with VGG features
# if `use_vgg` is set, VGG model will be created inside of the criterion
criterion = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4')
loss = criterion(img1, img2)
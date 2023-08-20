import math
from typing import Tuple

import torch
from torch import Tensor


def transformation_matrix_2D(center: Tuple[int, int], angle: float = 0, scale: float = 1, shifts: Tuple[float, float] = (0, 0)) -> Tensor:
    alpha = scale * math.cos(angle)
    beta = scale * math.sin(angle)
    #Issue
    affine_matrix = Tensor([[alpha, beta, (1-alpha)*center[1] - beta * center[0]],
                   [-beta, alpha, beta*center[1]+(1-alpha)*center[0]]])
    affine_matrix[1, 2] += float(shifts[0])  # shift_y
    affine_matrix[0, 2] += float(shifts[1])  # shift_x
    transformation_matrix = torch.zeros((3, 3))
    transformation_matrix[2, 2] = 1
    transformation_matrix[0:2, :] = affine_matrix
    return transformation_matrix


def param2theta(transformation: Tensor, H: int, W: int):
    param = torch.linalg.inv(transformation)
    theta = torch.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * H / W
    theta[0, 2] = param[0, 2] * 2 / W + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = param[1, 0] * W / H
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / H + theta[1, 0] + theta[1, 1] - 1
    return theta


def affine_transformation_matrix(dims: Tuple[int, int], angle: float = 0, scale: float = 1, shifts: Tuple[float, float] = (0, 0)) -> Tensor:
    H, W = dims
    invertable_transformation = transformation_matrix_2D((H/2, W/2), angle, scale, shifts)  #TODO: dudy changed from H//2,W//2
    transformation = param2theta(invertable_transformation, H, W)
    return transformation


def identity_transforms(N: int, angles: Tensor, scales: Tensor , shifts: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
    # returns identity angles, scale, shifts when they are not already defined
    if angles is None:
        angles = torch.zeros(N)
    if scales is None:
        scales = torch.ones(N)
    if shifts is None:
        shifts = (Tensor([0 for _ in range(N)]), Tensor([0 for _ in range(N)]))
    return angles, scales, shifts


def batch_affine_matrices(dims: Tuple[int, int], N: int = 1, angles: Tensor = None, scales: Tensor = None, shifts: Tuple[Tensor, Tensor] = None) -> Tensor:
    angles, scales, shifts = identity_transforms(N, angles, scales, shifts)
    affine_matrices = torch.zeros((N, 2, 3))
    for i in range(N):
        affine_matrices[i] = affine_transformation_matrix(dims, angles[i], scales[i], (shifts[0][i], shifts[1][i]))
    return affine_matrices



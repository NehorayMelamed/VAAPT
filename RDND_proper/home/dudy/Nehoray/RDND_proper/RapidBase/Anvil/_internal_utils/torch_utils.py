from typing import Tuple, List, Union

import numpy as np
import torch
from torch import Tensor
from functools import reduce

from RapidBase.Anvil._internal_utils.test_exceptions import raise_if_not


class MatrixOrigami:
    # expands matrix to 5 dimensions, then folds it back to original size at the end
    def __init__(self, matrix: Tensor):
        self.matrix = matrix
        self.original_dims = matrix.shape

    def expand_matrix(self, num_dims: int = 5) -> Tensor:
        # makes matrix requisite number of dimensions
        return fix_dimensionality(matrix=self.matrix, num_dims=num_dims)

    def expand_other_matrix(self, matrix: Tensor, num_dims = 5) -> Tensor:
        return fix_dimensionality(matrix, num_dims=num_dims)

    def squeeze_to_original_dims(self, matrix: Tensor) -> Tensor:
        # TODO Elisheva added this loop instead of squeeze()
        for dim in matrix.shape:
            if dim != 1:
                break
            matrix = matrix.squeeze(0)
        for i in range(len(self.original_dims) - len(matrix.shape)):
             matrix = matrix.unsqueeze(0)
        return matrix

# TODO Elisheva added this
class NoMemoryMatrixOrigami:
    # expands matrix to 5 dimensions, then folds it back to original size at the end
    def __init__(self, matrix: Tensor):
        self.original_dims = matrix.shape

    def expand_matrix(self, matrix: Tensor, num_dims: int = 5) -> Tensor:
        # makes matrix requisite number of dimensions
        return fix_dimensionality(matrix=matrix, num_dims=num_dims)

    def squeeze_to_original_dims(self, matrix: Tensor) -> Tensor:
        for dim in matrix.shape:
            if dim != 1:
                break
            matrix = matrix.squeeze(0)
        for i in range(len(self.original_dims) - len(matrix.shape)):
            matrix = matrix.unsqueeze(0)
        return matrix


def fix_dimensionality(matrix: Tensor, num_dims: int) -> Tensor:
    # makes matrix requisite number of dimensions
    current_dims = len(matrix.shape)
    if current_dims <= num_dims:
        for i in range(num_dims - len(matrix.shape)):
            matrix = matrix.unsqueeze(0)
        return matrix
    else:
        raise RuntimeError("Tried to expand a Tensor to a size smaller than itself")


def roll_n(matrix: Tensor, axis: int, n: int) -> Tensor:
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(matrix.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(matrix.dim()))
    front = matrix[f_idx]
    back = matrix[b_idx]
    return torch.cat([back, front], axis)


def faster_fft(matrix: Tensor, matrix_fft: Tensor, dim=[-2,-1]) -> Tensor:
    if matrix_fft is None:
        return torch.fft.fftn(matrix, dim=dim)
    else:
        return matrix_fft


def pad_matrix_to_size(matrix: Tensor, sizes: Tuple[int, int], resqueeze: bool = True) -> Tensor:
    # resqueeze is whether to bring back to less than 5 dims
    # adds padding to create Tensor of size sizes
    raise_if_not(len(sizes) == 2, message="Sizes must be of size H,W")
    B, T, C, old_H, old_W = matrix.shape
    H, W = sizes
    h_diff = H - old_H
    w_diff = W - old_W
    padded_matrix = torch.zeros((B,T,C,H,W))
    dimension_memory = MatrixOrigami(padded_matrix)
    expanded_padded_matrix = dimension_memory.expand_matrix()
    expanded_padded_matrix[:,:,:,h_diff//2:-h_diff//2, w_diff//2:-w_diff//2] = matrix
    if resqueeze:
        return dimension_memory.squeeze_to_original_dims(expanded_padded_matrix)
    else:
        return expanded_padded_matrix


def torch_int_types() -> List[torch.dtype]:
    return [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]


def true_dimensionality(matrix: Tensor) -> int:
    first_dim = len(matrix.shape)
    for num_dim, dim in enumerate(matrix.shape):
        if dim != 1:
            first_dim = num_dim
            break
    return len(matrix.shape)-first_dim


def dimension_N(matrix: Tensor, dimension: int) -> int:
    # returns size of dimension -dimension, or 1 if it doesn't exist
    if len(matrix.shape) >= dimension:
        return matrix.shape[-dimension]
    else:
        return 1


def extend_tensor_length_N(vector: Tensor, N: int) -> Tensor:
    """
    :param vector: 1D vector of size 1, or size N
    :param N: length the vector is, or will be extended to
    :return: vector of size N containing the element/s of input vector
    """
    if len(vector) == N:
        return vector
    else:
        return torch.ones(N) * vector


def compare_unequal_dimensionality_tensors(greater_dim_vector: Tensor, lesser_dim_vector: Tensor) -> bool:
    # returns if the tensors are the same size, consdiering dimensions of size 1 to be irrelevant
    # can also be used to compare equal sized vectors
    for i in range(-1, -len(greater_dim_vector.shape)-1, -1):
        try:
            if greater_dim_vector.shape[i] != lesser_dim_vector.shape[i]:
                return False
        except IndexError:
            if greater_dim_vector.shape[i] != 1:
                return False
    return True


# TODO Elisheva added
def compare_unequal_outer_dimensions(greater_dim_vector: Tensor, lesser_dim_vector: Tensor) -> bool:
    # A general test for cases where given tensors of different B, T dimensions
    # return true if lesser_dim_vector is a legitimate reference matrix for greater_dim_vector
    # Phase 1: CHW dimensions. must both exist and equal
    for i in range(-1, max(-len(greater_dim_vector.shape) - 1, -4), -1):
        try:
            if greater_dim_vector.shape[i] != lesser_dim_vector.shape[i]:
                return False
        except IndexError:
            return False
    # Phase 2: BT dimensions. False if lesser_dim_vector is not "private case" of greater_dim_vector.
    for i in range(-4, -len(greater_dim_vector.shape) - 1, -1):
        try:
            if greater_dim_vector.shape[i] != lesser_dim_vector.shape[i]:
                if lesser_dim_vector.shape[i] > 1:
                    return False
        except IndexError:
            return True
    return True


def equal_size_tensors(a: Tensor, b: Tensor) -> bool:
    # returns if tensors are of equal size
    if true_dimensionality(a) != true_dimensionality(b):
        return False
    elif len(a.shape) > len(b.shape):
        return compare_unequal_dimensionality_tensors(a, b)
    else: # need to know which tensor is greater dimensionality before comparing unequal dimensions
        return compare_unequal_dimensionality_tensors(b, a)


def extend_vector_length_n(vector: Tensor, N: int) -> Tensor:
    vector_length = len(vector)
    if vector_length == 1:
        return torch.ones(N) * vector
    elif int(N) == vector_length:
        return vector
    elif int(N) % vector_length == 0:
        return vector.repeat(int(N) // vector_length)
    else:
        raise RuntimeError("Cannot extend Tensor not factor length of N")


def non_interpolated_center_crop(matrix: Tensor, target_H: int, target_W: int) -> Tensor:
    B,T,C,H,W = matrix.shape
    excess_rows = H - target_H
    excess_columns = W - target_W
    start_y = int(excess_rows/2)
    start_x = int(excess_columns/2)
    stop_y = start_y+target_H
    stop_x = start_x+target_W
    return matrix[:, :, :, start_y:stop_y, start_x:stop_x]


def center_logical_mask(matrix: Tensor, zero_edges: int) -> Tensor:
    if len(matrix.shape) == 2:
        valid_mask = torch.zeros_like(matrix)
        H, W = matrix.shape
        valid_mask[zero_edges:H - zero_edges, zero_edges:W - zero_edges] = 1
    elif len(matrix.shape) == 3:
        valid_mask = torch.zeros_like(matrix)
        C, H, W = matrix.shape
        valid_mask[:, zero_edges:H - zero_edges, zero_edges:W - zero_edges] = 1
    elif len(matrix.shape) == 4:
        valid_mask = torch.zeros_like(matrix)
        T, C, H, W = matrix.shape
        valid_mask[:, :, zero_edges:H - zero_edges, zero_edges:W - zero_edges] = 1
    elif len(matrix.shape) == 5:
        valid_mask = torch.zeros_like(matrix)
        B, T, C, H, W = matrix.shape
        valid_mask[:, :, :, zero_edges:H - zero_edges, zero_edges:W - zero_edges] = 1
    else:
        raise RuntimeError("Matrix is not 2-5D")
    return valid_mask


# TODO Elisheva changed to allow parallelism and also named with torch for now
def unravel_index(index: Union[int, Tensor, np.array], dimensions: Union[List[int], Tuple[int]]) -> Tuple:
    # turns raw index into dimensioned index
    dimensioned_index = []
    dimensions = list(dimensions)
    dimensions.append(1)  # for determining last dimension
    for i in range(len(dimensions)-1):
        remaining_space = reduce(lambda x, y: x*y, dimensions[i+1:])
        dimensioned_index.append(index//remaining_space)
        index -= dimensioned_index[-1]*remaining_space
    return tuple(dimensioned_index)


# TODO Elisheva changed to allow parallelism and also named with torch for now
def unravel_index_torch(index: Union[int, Tensor, np.array], dimensions: Union[List[int], Tuple[int]]) -> Tuple:
    # turns raw index into dimensioned index
    remaining_index = index.clone()
    dimensioned_index = []
    dimensions = list(dimensions)
    dimensions.append(1)  # for determining last dimension
    for i in range(len(dimensions)-1):
        remaining_space = reduce(lambda x, y: x*y, dimensions[i+1:])
        dimensioned_index.append(remaining_index//remaining_space)
        remaining_index -= dimensioned_index[-1]*remaining_space
    return tuple(dimensioned_index)


# TODO Elisheva added
def ravel_multi_index(multi_index: tuple,
                      dimensions: Union[List[int], Tuple[int]]) -> Union[Tensor, np.array, int]:
    # turns dimensioned index into raw index
    flattened_index = multi_index[0] - multi_index[0]  # to keep the element type yet get zero
    for i in range(len(dimensions) - 1):
        dim_block_size = reduce(lambda x, y: x*y, dimensions[i+1:])
        flattened_index = flattened_index + multi_index[i] * dim_block_size
    return flattened_index


def convert_iterable_tensor(elements: Union[list, tuple]) -> Tensor:
    # recursively converts a list of any form into Tensor.
    if type(elements[0]) == Tensor:
        return torch.stack(elements)
    elif type(elements[0]) in [float, int, np.float, np.float64, np.float32, np.int, np.uint8, np.int16]:
        return Tensor(elements)
    else:
        return torch.stack([convert_iterable_tensor(iterable) for iterable in elements])


# def construct_tensor(elements: Union[Tensor, list, tuple, float, int, np.array], device: Union[str, torch.device] = "cpu"):
#     if type(elements) in [float, int]:
#         return Tensor([elements], device=device)
#     elif type(elements) == Tensor and str(elements.device) == str(device):
#         if len(elements.shape) == 0:  # is tensor of form tensor(float), as opposed to tensor([float]). Problem since former type is not iterable
#             return Tensor([float(elements)])
#         return elements
#     elif type(elements) in [list, tuple]:
#         return convert_iterable_tensor(elements).to(device)
#     else:
#         return Tensor(elements, device=device)

# TODO Elisheva changed to new version for construct tensor, notice and check carefully
def construct_tensor(elements: Union[Tensor, list, tuple, float, int, np.array], device: Union[str, torch.device] = "cpu"):
    if type(elements) in [float, int]:
        return torch.tensor([elements], device=device)
    # we have a device issue here - if tensor, leave the original device or change?
    elif type(elements) == Tensor:
        if len(elements.shape) == 0:  # is tensor of form torch.tensor(float), as opposed to torch.tensor([float]). Problem since former type is not iterable
            return torch.tensor([float(elements)]).to(elements.device)
        return elements.to(elements.device)
    elif type(elements) in [list, tuple]:
        return convert_iterable_tensor(elements).to(device)
    else:
        return torch.tensor(elements, device=device)


def stepped_arange(start: Union[int, float], stop: Union[int, float], steps: int) -> Tensor:
    """
    :param start: lower bound on arange (inclusive)
    :param stop: upper bound on arange (non-inclusive)
    :param steps: number of elements to be in the arange
    :return: arange from start to stopp in step steps
    """
    if stop-start%steps == 0:
        return torch.arange(start, stop, (stop-start)//steps)
    else:
        return torch.arange(start, stop, (stop-start)/steps)



def RGB2BW(input_image):
    if len(input_image.shape) == 2:
        return input_image

    if len(input_image.shape) == 3:
        if type(input_image) == torch.Tensor and input_image.shape[0] == 3:
            grayscale_image = 0.299 * input_image[0:1, :, :] + 0.587 * input_image[1:2, :, :] + 0.114 * input_image[2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1] + 0.587 * input_image[:, :, 1:2] + 0.114 * input_image[:, :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 4:
        if type(input_image) == torch.Tensor and input_image.shape[1] == 3:
            grayscale_image = 0.299 * input_image[:, 0:1, :, :] + 0.587 * input_image[:, 1:2, :, :] + 0.114 * input_image[:, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, 0:1] + 0.587 * input_image[:, :, :, 1:2] + 0.114 * input_image[:, :, :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 5:
        if type(input_image) == torch.Tensor and input_image.shape[2] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1, :, :] + 0.587 * input_image[:, :, 1:2, :, :] + 0.114 * input_image[:, :, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, :, 0:1] + 0.587 * input_image[:, :, :, :, 1:2] + 0.114 * input_image[:, :, :, :, 2:3]
        else:
            grayscale_image = input_image

    return grayscale_image

def RGB2BW_interleave_C(input_image):  #Only for tensors for now, and if you know(!) they're RGB
    #(*). Unless given another input of C or T, this function cannot know whether it was really given a RGB image, SO IT ASSUMES IT!!! YOU NEED TO MAKE SURE!!!!!
    if len(input_image.shape)==3:
        TxC,H,W = input_image.shape
        new_C = TxC//3
        input_image = input_image.reshape(3, new_C, H, W)
        grayscale_image = 0.299 * input_image[0:1, :, :, :] + 0.587 * input_image[1:2, :, :, :] + 0.114 * input_image[2:3, :, :, :]

    elif len(input_image.shape)==4:
        B, TxC, H, W = input_image.shape
        new_C = TxC // 3
        input_image = input_image.reshape(B, 3, new_C, H, W)
        grayscale_image = 0.299 * input_image[:, 0:1, :, :, :] + 0.587 * input_image[:, 1:2, :, :, :] + 0.114 * input_image[:, 2:3, :, :, :]

    return grayscale_image

def RGB2BW_interleave_T(input_image):  #Only for tensors for now, and if you know(!) they're RGB
    if len(input_image.shape)==4:    #[T,C,H,W]
        T,C,H,W = input_image.shape
        if C == 3:
            grayscale_image = 0.299 * input_image[:, 0:1, :, :] + 0.587 * input_image[:, 1:2, :, :] + 0.114 * input_image[:, 2:3, :, :]
        else:
            grayscale_image = input_image

    elif len(input_image.shape)==5:  #[B,T,C,H,W]
        B, T, C, H, W = input_image.shape
        if C == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1, :, :] + 0.587 * input_image[:, :, 1:2, :, :] + 0.114 * input_image[:, :, 2:3, :, :]
        else:
            grayscale_image = input_image

    return grayscale_image


def BW2RGB_interleave_C(input_image):
    ### For Torch Tensors!: ###
    if len(input_image.shape) == 3:
        RGB_image = torch.repeat_interleave(input_image, 3, 0)

    elif len(input_image.shape) == 4:
        RGB_image = torch.repeat_interleave(input_image, 3, 1)

    elif len(input_image.shape) == 5:
        RGB_image = torch.cat([input_image]*3, 2)

    return RGB_image

def BW2RGB_interleave_T(input_image):
    ### For Torch Tensors!: ###
    if len(input_image.shape) == 4:  #[T,C,H,W]
        RGB_image = torch.repeat_interleave(input_image, 3, 1)

    elif len(input_image.shape) == 5:  #[B,T,C,H,W]
        RGB_image = torch.repeat_interleave(input_image, 3, 2)

    return RGB_image


def BW2RGB_MultipleFrames(input_image, flag_how_to_concat='C'):
    if flag_how_to_concat == 'C':
        RGB_image = BW2RGB_interleave_C(input_image)
    else:
        RGB_image = BW2RGB_interleave_T(input_image)
    return RGB_image

def RGB2BW_MultipleFrames(input_image, flag_how_to_concat='C'):
    if flag_how_to_concat == 'C':
        grayscale_image = RGB2BW_interleave_C(input_image)
    else:
        grayscale_image = RGB2BW_interleave_T(input_image)
    return grayscale_image

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




def multiplicable(a: Tensor, b: Tensor) -> bool:
    for i in range(min(len(a.shape), len(b.shape))+1):
        if a.shape[-i] != b.shape[-i] and not (a.shape[-i] == 1 or b.shape[-i]==1):
            return False
    return True


def multiplicable_tensors(tensors: List[Tensor]) -> bool:
    # returns whether all tensors in list can multiply each other
    for i in range(len(tensors)):
        for j in range(len(tensors)):
            if i == j:
                continue  # same tensor
            else:
                if not multiplicable(tensors[i], tensors[j]):
                    return False
    return True

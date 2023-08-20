import cv2, torch
from Palantir_utils import center_crop
# from RapidBase.Anvil._internal_utils.torch_utils import RGB2BW
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import scale_array_stretch_hist, crop_torch_batch


a = torch.zeros((4, 1, 450, 450))
b = center_crop(a, (460, 460))
c = 1+1


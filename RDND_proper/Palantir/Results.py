from torch import Tensor

from Palantir.Palantir_utils import find_outliers, pad_torch_batch, center_crop
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import scale_array_stretch_hist

from icecream import ic
ic.configureOutput(includeContext=True)


class Results:
    # this class calculates all metrics and moves everything to the CPU
    def __init__(self, background: Tensor, sub_aligned_tensor, original_tensor, H_matrix, params):
        # crops all tensors to same size
        final_h, final_w = params.algo.final_crop_h, params.algo.final_crop_w
        actual_h = min(final_h, sub_aligned_tensor.shape[-2])
        actual_w = min(final_w, sub_aligned_tensor.shape[-1])
        self.background = center_crop(background, (actual_h, actual_w))
        self.background = pad_torch_batch(self.background, (final_h, final_w))
        aligned_tensor = center_crop(sub_aligned_tensor, (final_h, final_w))
        self.aligned_tensor = pad_torch_batch(aligned_tensor, (final_h, final_w))
        original_tensor = center_crop(original_tensor, (final_h, final_w))
        self.original_frames = pad_torch_batch(original_tensor, (final_h, final_w))

        self.H_matrix = H_matrix
        self.params = params
        if self.params.results.use_outliers:
            # not really good code. Results should rely on resultsSaver to tell it what to save. It is the way it is now to allow for easier debugging
            self.outliers = find_outliers(self.aligned_tensor, self.background)
            self.outliers = self.outliers.cpu()
        if self.params.results.use_diff_tensor:
            diff_tensor = (self.aligned_tensor - self.background).abs()
            diff_tensor = scale_array_stretch_hist(diff_tensor, (0.1, 0.99))
            self.diff_tensor = diff_tensor.cpu()
        self.aligned_tensor = self.aligned_tensor.cpu()
        self.original_frames = self.original_frames.cpu()

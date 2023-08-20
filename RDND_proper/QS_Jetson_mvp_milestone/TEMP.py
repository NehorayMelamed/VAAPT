
from RapidBase.import_all import *
from RapidBase.PyTorchSteerablePyramid.steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import RapidBase.PyTorchSteerablePyramid.steerable.utils as SCFpyr_utils

input_tensor = RGB2BW(read_image_default_torch())
pyr = SCFpyr_PyTorch(height=5, nbands=4, scale_factor=1, device=input_tensor.device)
coeff = pyr.build(input_tensor)
# im_batch_reconstructed = pyr.reconstruct(coeff)

# Visualization
coeff_single = SCFpyr_utils.extract_from_batch(coeff, 0)
coeff_grid = SCFpyr_utils.make_grid_coeff(coeff, normalize=True)
cv2.imshow('Complex Steerable Pyramid', coeff_grid)
cv2.waitKey(0)

#TODO: use the Riesz transform (both fourier and rubinstein real-space versions) to get an amplitude+phase for the network: ###

bla = 1
imshow_torch(input_tensor)
imshow_torch(coeff[0].abs())
imshow_torch(coeff[-1].abs())
imshow_torch(coeff[1][0].abs())


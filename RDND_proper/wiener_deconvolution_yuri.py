import torch.nn
import torchvision.transforms

from RapidBase.import_all import *

### Get psf tensor: ###
psf_tensor = torch.randn(1,1,33,33)
kernel_full_filename = r'/home/dudy/Nehoray/RDND_proper/RapidBase/Data/TestData/kernelImage/1.png'
psf_image = read_image_cv2(kernel_full_filename)
psf_tensor = numpy_to_torch(psf_image)

### Get input image: ###
input_tensor = torch.randn(1,3,256,256)
image_full_filename = r'/home/dudy/Nehoray/RDND_proper/RapidBase/Data/TestData/blurredImage/1.png'
input_image = read_image_cv2(image_full_filename)
input_tensor = numpy_to_torch(input_image)

# ### Perform Affine TRansform: ###
# angles_vec_degrees = [-10,-5,0,5,10]
# scale_vec = [0.8,0.9,1,1.1,1.2]
# final_psfs_list = []
# for angle_index in np.arange(len(angles_vec_degrees)):
#     for scale_index in np.arange(len(scale_vec)):
#         ### Get current parameters: ###
#         current_angle = angles_vec_degrees[angle_index]
#         current_scale = scale_vec[scale_index]
#
#         ### Affine transform: ###
#         #TODO: at the end use our transforms instead of torchvision
#         final_psf = torchvision.transforms.functional.affine(psf_tensor, angle=current_angle, scale=current_scale, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
#
#         ### Append to list: ###
#         final_psfs_list.append(final_psf)

### Upsample Kernel Homogeneously: ###
scale_vec = [0.8,0.9,1,1.1,1.2]
final_psfs_list = []
for scale_index in np.arange(len(scale_vec)):
    ### Initialize upsample layer: ###
    upsample_layer = torch.nn.Upsample(scale_factor=current_scale)

    ### Get final psf: ###
    final_psf = upsample_layer.forward(psf_tensor)

    ### Append to list: ###
    final_psfs_list.append(final_psf)


### Perform FFT: ###
H,W = input_tensor.shape[-2:]
for psf_index in np.arange(final_psfs_list):
    ### Get current image FFT: ###
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=(-2,-1))

    ### Get current psf: ###
    current_psf = final_psfs_list[psf_index]

    ### Get PSF fft: ###
    current_psf_fft = torch.fft.fftn(current_psf, s=(H,W), dim=(-2,-1))

    ### Perform Wiener Deconvolution: ###
    SNR_vec = [0.001, 0.1, 1, 2, 5]  #we'll think about reasonable values later
    final_results_list = []
    for current_SNR in SNR_vec:
        #TODO: make sure my implementation of weiner deconvolution is correct or use another implementation

        ### Filter in FFT domain using multiplication: ###
        input_tensor_fft_filtered = input_tensor_fft * current_psf_fft.conj() / (current_psf_fft.abs()**2 + 1/SNR)

        ### Inverse FFT to get the real-space result of the wiener deconvolution: ###
        input_tensor_filtered = torch.fft.ifftn(input_tensor_fft_filtered, dim=(-2,-1)).real

        ### Append to list: ###
        final_results_list.append(input_tensor_filtered)

        ### Present final results with proper titles: ###









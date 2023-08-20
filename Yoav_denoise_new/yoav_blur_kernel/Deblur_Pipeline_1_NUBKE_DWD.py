import sys
import os

import PARAMETER

sys.path.append('../../../')
sys.path.append('../..')
sys.path.append('../')

### current work directory - '/home/nehoray/PycharmProjects/Shaback/Yoav_denoise_new/yoav_blur_kernel'


from util.image_segmenter import ImageSegmenter
from skimage.util import img_as_ubyte


sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/models/NonUniformBlurKernelEstimation/NonUniformBlurKernelEstimation/utils_NUBKE") #ToDo
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/RDND_proper/models/NonUniformBlurKernelEstimation/NonUniformBlurKernelEstimation/utils_NUBKE")
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/RDND_proper/models/NonUniformBlurKernelEstimation/NonUniformBlurKernelEstimation")
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/RDND_proper")
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/models/dwdn/dwdn")
# sys.path.append('RDND_proper/models/NonUniformBlurKernelEstimation/NonUniformBlurKernelEstimation')
# sys.path.append('RDND_proper/models/NonUniformBlurKernelEstimation/NonUniformBlurKernelEstimation/utils_NUBKE')

from skimage.io import imsave
from RDND_proper.models.NonUniformBlurKernelEstimation.NonUniformBlurKernelEstimation.models.TwoHeadsNetwork import TwoHeadsNetwork



###Load the bas imports, they couse to some issues whne trying to use cv2
from RDND_proper.models.NonUniformBlurKernelEstimation.NonUniformBlurKernelEstimation.utils_NUBKE.visualization import \
    save_kernels_grid, get_kernels_grid

from RDND_proper.models.NonUniformBlurKernelEstimation.NonUniformBlurKernelEstimation.utils_NUBKE.restoration import \
    RL_restore, combined_RL_restore
from RapidBase.import_all import *




DEVICE = 0
# model_file = "RDND_proper/models/NonUniformBlurKernelEstimation/NonUniformBlurKernelEstimation/models/TwoHeads.pkl"

model_file = "TwoHeads.pkl"
output_dir = "output"


def my_own_make_grid(kernels):
    ker_count = int(math.sqrt(kernels.shape[0]))
    ks = int(kernels.shape[1])
    kernel_image = torch.zeros((ker_count*ks, ker_count*ks))
    for kernel_id, kernel in enumerate(kernels):
        i = kernel_id // ker_count
        j = kernel_id % ker_count
        kernel_image[i*ks:(i+1)*ks, j*ks:(j+1)*ks] = kernel

    return kernel_image


def my_own_make_grid_indices(kernels, kernel_id):
    ker_count = int(math.sqrt(kernels.shape[0]))
    ks = int(kernels.shape[1])
    kernel_image = torch.zeros((ker_count*ks, ker_count*ks))
    i = kernel_id // ker_count
    j = kernel_id % ker_count
    kernel_image[i*ks:(i+1)*ks, j*ks:(j+1)*ks] = 1

    return kernel_image


def get_avg_kernel_from_segmentation_masks_and_kernels(blurry_image, basis_kernels, masks, seg_mask=None, c=None):
    """
    function with n**2 and i want the avg kernel inside seg_mask
    todo: add the kernel * original_mask to get correct kernels per location
    Args:
        kernels:
        seg_mask:
    Returns:
    """
    M, N = masks.shape[-2:]


    # if seg_mask is None:
    #     seg_mask = torch.zeros((1, 1, M, N))
    #     seg_mask[:, :, 100:200, 100:200] = 1
    if seg_mask is not None:
        if len(seg_mask.shape) == 3:
            seg_mask = seg_mask.unsqueeze(0)
        if len(seg_mask.shape) != 4 :#((1, 1, M, N))
            raise ValueError("Segmentation masks invalid shape")
    else:
        seg_mask = torch.zeros((1,1,M,N))


    imshow_torch_temp(seg_mask[0], name="seg_mask_before_scale")

    # get the kernel corrosponding to image locations based on masks and kernel basis
    corrected_kernels = get_kernels_grid(blurry_image, basis_kernels.squeeze(), masks.squeeze())
    # todo: think about removing the reshape if i flatten anyway
    corrected_kernels = corrected_kernels.flatten(0, 1)
    # for i in range(corrected_kernels.shape[0]):
    #     imshow_torch_temp(corrected_kernels[i], f"corrected_{str(i).zfill(2)}")
    # imshow_torch_temp(basis_kernels[0, 5] * 255)

    kernel_grids = torch.cat([my_own_make_grid_indices(corrected_kernels.squeeze(), i).unsqueeze(0) for i in range(corrected_kernels.shape[0])])
    # create one image per kernel basis, then multiply each wth seg mask and sum the result to see how much fits
    # imshow_torch_temp(kernel_grids[3000] * 255)

    seg_mask = torch.nn.functional.upsample(seg_mask, size=kernel_grids.shape[1:]).squeeze()
    imshow_torch_temp(seg_mask.unsqueeze(0), name="seg_mask_after_scale")
    # [ks, h, w] has weight of kernel at each location
    kernel_weights = kernel_grids * seg_mask.to(kernel_grids.device)
    # for i, k in enumerate(kernel_weights):
    #     if k.sum() != 0:
    #         imshow_torch_temp(k, f"kernel_weights_{str(i).zfill(2)}")
    #         imshow_torch_temp(corrected_kernels[i], f"kernel_{str(i).zfill(2)}")
    # imshow_torch_temp(kernel_weights[6])
    kernel_weights = kernel_weights.sum(-1).sum(-1)
    avg_kernel = (corrected_kernels.squeeze() * kernel_weights.unsqueeze(-1).unsqueeze(-1).to(corrected_kernels.device)).sum(0)


    imshow_torch_temp(avg_kernel.unsqueeze(0), "avg_kernel")
    # average kernels per location
    return avg_kernel


def find_kernels_NUBKE(blurred_image, output_dir):
    K = 25  # number of elements en the base
    #Todo check with Yoav the correct pkl file !!!
    gamma_factor = 2.2


    if path.exists(output_dir) is False:
        os.mkdir(output_dir)

    two_heads = TwoHeadsNetwork(K).to(DEVICE)
    print('loading weight\'s model')
    two_heads.load_state_dict(torch.load(model_file, map_location='cuda:%d' % DEVICE))

    two_heads.eval()
    # todo: blurred_image to range [0, 1]
    blurred_image = blurred_image.squeeze()
    # Kernels and masks are estimated
    blurry_tensor_to_compute_kernels = blurred_image ** gamma_factor - 0.5
    kernels_estimated, masks_estimated = two_heads(blurry_tensor_to_compute_kernels[None, :, :, :])

    # kernels_grid = my_own_make_grid(kernels_estimated.squeeze())
    # imshow_torch_temp(255 * kernels_grid)


    kernels_val_n = kernels_estimated[0, :, :, :]
    kernels_val_n_ext = kernels_val_n[:, np.newaxis, :, :]

    blur_kernel_val_grid = make_grid(kernels_val_n_ext, nrow=K,
                                     normalize=True, scale_each=True, pad_value=1)
    mask_val_n = masks_estimated[0, :, :, :]
    mask_val_n_ext = mask_val_n[:, np.newaxis, :, :]
    blur_mask_val_grid = make_grid(mask_val_n_ext, nrow=K, pad_value=1)

    imsave(os.path.join(output_dir, '_kernels.png'),
           img_as_ubyte(blur_kernel_val_grid.detach().cpu().numpy().transpose((1, 2, 0))))

    imsave(os.path.join(output_dir, '_masks.png'),
           img_as_ubyte(blur_mask_val_grid.detach().cpu().numpy().transpose((1, 2, 0))))

    win_kernels_grid = save_kernels_grid(blurred_image, torch.flip(kernels_estimated[0], dims=(1, 2)),
                                         masks_estimated[0],
                                         os.path.join(output_dir, '_kernels_grid.png'))

    # for i in range(K):
    #     fig = imshow_torch(kernels_val_n[i])
    #     plt.savefig(f"/raid/yoav/temp_garbage/base_kernel_{str(i).zfill(2)}.png")

    return kernels_val_n



def find_kernel_NUBKE(blurred_image, seg_mask=None):
    K = 25  # number of elements en the base
    gamma_factor = 2.2

    two_heads = TwoHeadsNetwork(K).to(DEVICE)
    print('loading weight\'s model')
    two_heads.load_state_dict(torch.load(model_file, map_location='cuda:%d' % DEVICE))

    with torch.no_grad():
        blurry_tensor_to_compute_kernels = blurred_image ** gamma_factor - 0.5
        kernels, masks = two_heads(blurry_tensor_to_compute_kernels.unsqueeze(0))

    if seg_mask is None:
        seg_mask = torch.ones_like(blurred_image)

    avg_kernel = get_avg_kernel_from_segmentation_masks_and_kernels(blurry_tensor_to_compute_kernels, kernels, masks, seg_mask)
    return avg_kernel


def deblur_image_pipline_NUBKE(blurred_image, seg_mask=None):
    K = 25  # number of elements en the base
    gamma_factor = 2.2

    two_heads = TwoHeadsNetwork(K).to(DEVICE)
    print('loading weight\'s model')
    two_heads.load_state_dict(torch.load(model_file, map_location='cuda:%d' % DEVICE))

    initial_restoration_tensor = blurred_image.clone()

    with torch.no_grad():
        blurry_tensor_to_compute_kernels = blurred_image ** gamma_factor - 0.5
        kernels, masks = two_heads(blurry_tensor_to_compute_kernels.unsqueeze(0))


    avg_kernel = get_avg_kernel_from_segmentation_masks_and_kernels(blurry_tensor_to_compute_kernels, kernels, masks, seg_mask)
    # imshow_torch_temp(avg_kernel, name="weighted_kernel_according_to_mask")

    # imshow_torch_temp(masks[0, 1], name="mask_0")
    # for i in np.arange(25):
    #     imshow_torch_temp(kernels[0, i], name="kernel_" + str(i).zfill(2))

    output = initial_restoration_tensor

    with torch.no_grad():

        if True:
            blurred_image = blurred_image.unsqueeze(0)
            output = output.unsqueeze(0)
            # print(blurred_image.shape)
            # print(output.shape)
            # print(kernels.shape)
            # print(masks.shape)
            # print(blurred_image.max())
            # print(output.max())
            # print(kernels.max())
            # print(masks.max())
            output = combined_RL_restore(blurred_image, output, kernels, masks, 30, blurred_image.device,
                                         SAVE_INTERMIDIATE=True, saturation_threshold=0.99,
                                         reg_factor=1e-3, optim_iters=1e-6, gamma_correction_factor=2.2,
                                         apply_dilation=False, apply_smoothing=True, apply_erosion=True)
        else:
            output = RL_restore(blurred_image.unsqueeze(0), output, kernels, masks, 30, blurred_image.device)

    imshow_torch_temp(output[0].clip(0, 1), name="NUBKE_estimation")

    return output


def imshow_torch_temp(image, name="temp"):
    #ToDo rename it
    plt.imshow(image.permute(1, 2, 0).cpu().detach().numpy())
    plt.savefig(f"{output_dir}/{name}.png")


def main_interface(image_path, output_dir_for_global):
    global output_dir
    output_dir = output_dir_for_global
    if path.exists(output_dir_for_global) is False:
        os.mkdir(output_dir_for_global)

    #### load_from_segmentation file
    # seg_path = "/home/dudy/Nehoray/segment_anything_base_dir/Grounded-Segment-Anything/outputs/output_minivan_1/27/masks/27_mask_car_3.pt"
    # seg_mask = torch.load(seg_path)

    ###Load image and let user to draw segmentation mask
    img_segmenter = ImageSegmenter(image_path)
    seg_mask, image_base_segment = img_segmenter.get_mask()


    seg_mask_torch = numpy_to_torch(seg_mask)
    blurred_image = read_image_torch(image_path).to(DEVICE)
    imshow_torch_temp(blurred_image.squeeze() / 255, "input")
    find_kernels_NUBKE(blurred_image.squeeze() / 255, output_dir)
    deblurred = deblur_image_pipline_NUBKE(blurred_image.squeeze() / 255, seg_mask=seg_mask_torch)
    return True


# if __name__ == "__main__":
#     image_path = "/home/nehoray/PycharmProjects/Shaback/Grounded_Segment_Anything/output_car/raw_image.jpg"
#     main_interface(image_path=image_path, output_dir_for_global="output4")



#     ### Load from pt file
#     if seg_mask_pt_file_path is not None:
#         pt_file_mask_data = torch.load(seg_mask_pt_file_path)
#         seg_mask = pt_file_mask_data
#     else:
#         raise ValueError("No mask provided")
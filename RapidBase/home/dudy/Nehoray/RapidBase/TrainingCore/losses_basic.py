import torch.nn as nn
from easydict import EasyDict

from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_get_valid_center
from RapidBase.Utils.MISCELENEOUS import update_dict
import torch
import torch.functional as F
from numpy import arange
import numpy as np

# from contextual_loss.modules.vgg import VGG19
# from contextual_loss import functional as F
# from contextual_loss .config import LOSS_TYPES

def gradient_x(img):
    gx = torch.nn.functional.pad((img[:, :, :, :-1] - img[:, :, :, 1:]), (0, 1, 0, 0), mode="replicate")
    return gx
def gradient_y(img):
    gy = torch.nn.functional.pad((img[:, :, :-1, :] - img[:, :, 1:, :]), (0, 0, 0, 1), mode="replicate")
    return gy
def second_order_gradient_x(img):
    return gradient_x(gradient_x(img))
def second_order_gradient_y(img):
    return gradient_y(gradient_y(img))

class MaskedL1LossConfidence(nn.Module):
    def __init__(self):
        super(MaskedL1LossConfidence, self).__init__()

    def forward(self, pred, target, valid_mask, confidence=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target - pred
        diff = diff[valid_mask]

        diff_abs = diff.abs()
        if confidence is None:
            confidence = torch.ones_like(valid_mask)
        conf_diff_abs = confidence[valid_mask] * diff_abs
        Masked_L1_Loss = conf_diff_abs.mean()

        return Masked_L1_Loss

class MaskedL2LossConfidence(nn.Module):
    def __init__(self):
        super(MaskedL2LossConfidence, self).__init__()

    def forward(self, pred, target, valid_mask, confidence=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target - pred
        diff = diff[valid_mask]

        diff_abs = diff.abs()**2
        if confidence is None:
            confidence = torch.ones_like(valid_mask)

        conf_diff_abs = confidence[valid_mask] * diff_abs
        Masked_L1_Loss = conf_diff_abs.mean()

        return Masked_L1_Loss

class MaskedSpillingLossLoss(nn.Module):
    def __init__(self):
        super(MaskedSpillingLossLoss, self).__init__()
    def forward(self, pred, target, valid_mask, weight):
        assert pred.dim() == target.dim(), "inconsistent dimensions"


        loss_metric = 0*(pred[valid_mask]*weight[valid_mask]).mean()

        criterion = torch.nn.BCELoss()
        loss_fill_factor = criterion(pred[valid_mask],valid_mask[valid_mask].float())

        Masked_metric_Loss = loss_metric + loss_fill_factor
        return Masked_metric_Loss

class MaskedConfidenceBCELoss(nn.Module):
    def __init__(self):
        super(MaskedConfidenceBCELoss, self).__init__()

    def forward(self, confidence, valid_mask, conf_gt):
        criterion = torch.nn.BCEWithLogitsLoss()

        Masked_confidence_loss = criterion(confidence[valid_mask], conf_gt[valid_mask].float())

        return Masked_confidence_loss

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        Masked_MSE_Loss = (diff ** 2).mean() #L2 basically....
        return Masked_MSE_Loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, valid_mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target - pred
        diff = diff[valid_mask]
        Masked_L1_Loss = diff.abs().mean()
        return Masked_L1_Loss

class L1Loss_TargetValue(nn.Module):
    def __init__(self):
        super(L1Loss_TargetValue, self).__init__()
    #TODO: what eventually will need to happen is a good estimation and definition of the
    # noise level (not just in the case of AGWN) and then sigma->sigma/sqrt(T) for predicted T
    def forward(self, pred, target, target_value):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff_abs = (target - pred).abs()
        Masked_L1_Loss = (diff_abs - target_value).mean()  #(*). can be one number or a complete tensor
        return Masked_L1_Loss

class MaskedL1Loss_TargetValue(nn.Module):
    def __init__(self):
        super(MaskedL1Loss_TargetValue, self).__init__()
    #TODO: what eventually will need to happen is a good estimation and definition of the
    # noise level (not just in the case of AGWN) and then sigma->sigma/sqrt(T) for predicted T
    def forward(self, pred, target, target_value, valid_mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff_abs = (target - pred).abs()
        Masked_L1_Loss = ((diff_abs - target_value)[valid_mask]).mean()  #(*). can be one number or a complete tensor
        return Masked_L1_Loss

class MaskedL1Loss_TargetMean(nn.Module):
    def __init__(self):
        super(MaskedL1Loss_TargetMean, self).__init__()
    def forward(self, pred, target, target_value, valid_mask):
        #TODO: this is the global version of the above MaskedL1Loss_TargetValue, i can also do a "mid-level"/local version which asks local statistics to be the same
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff_abs = (target - pred).abs()
        Masked_L1_Loss = (diff_abs[valid_mask].mean() - target_value[valid_mask].mean()).abs()
        return Masked_L1_Loss


class MaskedL2Loss(nn.Module):
    def __init__(self):
        super(MaskedL2Loss, self).__init__()

    def forward(self, pred, target, valid_mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target - pred
        diff = diff[valid_mask]
        Masked_L2_Loss = (diff.abs()**2).mean()
        return Masked_L2_Loss

class MaskedGradLossConfidence(nn.Module):
    def __init__(self):
        super(MaskedGradLossConfidence, self).__init__()

    def forward(self, pred, target, valid_mask, confidence=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diffx = (target[:, :, 1:] - target[:, :, :-1]) - (pred[:, :, 1:] - pred[:, :, :-1])
        diffx = diffx[valid_mask[:, :, 1:]]

        diffy = (target[:, :, :, 1:]-target[:, :, :, :-1]) - (pred[:, :, :, 1:]-pred[:, :, :, :-1])
        diffy = diffy[valid_mask[:, :, :, 1:]]

        if confidence is None:
            Grad_Loss = (diffx.abs().mean() + diffy.abs().mean())
        else:
            confidence_x = confidence[:, :, :, 0:1]
            confidence_y = confidence[:, :, :, 1:2]
            Grad_Loss = (diffx.abs() * confidence_x[valid_mask[:, :, 1:]]).mean() + \
                (diffy.abs() * confidence_y[valid_mask[:, :, :, 1:]]).mean()

        return Grad_Loss


class Masked_RMSE_log(nn.Module):
    def __init__(self):
        super(Masked_RMSE_log, self).__init__()

    def forward(self, fake, real):
        valid_mask = (real > 0)
        ### Take Care Of Shape Misalignement: ###
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        loss = torch.sqrt(torch.mean(torch.abs(torch.log(real[valid_mask]) - torch.log(fake[valid_mask])) ** 2))
        return loss


class Masked_L1_log(nn.Module):  # change name!: this is Masked_L1_Logg !!!!!
    def __init__(self):
        super(Masked_L1_log, self).__init__()

    def forward(self, fake, real):
        assert fake.dim() == real.dim(), "inconsistent dimensions"
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear', align_corners=True)
        ### Get valid mask for comparison: ###
        valid_mask = (real > 0).detach()
        real = real[valid_mask]
        fake = fake[valid_mask]
        loss = torch.mean(torch.abs(torch.log(real) - torch.log(fake)))
        return loss


class Masked_berHuLoss1(nn.Module):  # Huber Smooth Loss: basically L2 with L1 for small values
    def __init__(self):
        super(Masked_berHuLoss1, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        ### Get huber threshold for L1->L2 transition: ###
        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c
        ### Get "Base" L1 Loss: ###
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()
        ### For Large Loss Values Have L2 Loss: ###
        huber_mask = (diff > huber_c).detach()
        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2
        ### Get Mean Between L1&L2 Parts: ###
        # TODO: make sure this is correct because i think that with diff2=diff[huber_mask] we are pointing to diff[huber_mask] values and changing them...than why the .cat().mean()?
        Huber_Loss = torch.cat((diff, diff2)).mean()
        return Huber_Loss


class Masked_berHuLoss2(nn.Module):  # Masked Huber Loss ... but i need to understand it completely......
    def __init__(self, threshold=0.2):
        super(Masked_berHuLoss2, self).__init__()
        self.threshold = threshold

    def forward(self, real, fake):
        valid_mask = real > 0
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.upsample(fake, size=(H, W), mode='bilinear')
        fake = fake * valid_mask
        diff = torch.abs(real - fake)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()[0]

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 - delta ** 2, 0., -delta ** 2.) + delta ** 2
        part2 = part2 / (2. * delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss


class Masked_GradLoss(nn.Module):  # Basically L1 but in the code we apply it to the gradient of the depth maps!!!!
    def __init__(self):
        super(Masked_GradLoss, self).__init__()

    def forward(self, pred, target):
        valid_mask = target > 0
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        return torch.mean(torch.abs(target - pred)[valid_mask])


class Masked_NormalLoss(nn.Module):  # New "Normal" Loss why penalizes incorrect normal direction(?????)
    def __init__(self):
        super(Masked_NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        assert grad_fake.dim() == grad_real.dim(), "inconsistent dimensions"
        # prod = (grad_fake[:, :, None, :] @ grad_real[:, :, :, None]).squeeze(-1).squeeze(-1)
        valid_mask = grad_real > 0
        prod = grad_fake * grad_real
        # first of all.....if we sum and don't unsqueeze then dimensions don't match for later.... moreove why only so along the last dimension?!?!?!...maybe the last dimension = 2 for x and y grads?)
        fake_norm = torch.sqrt(torch.sum(grad_fake ** 2, dim=-1))
        real_norm = torch.sqrt(torch.sum(grad_real ** 2, dim=-1))
        ### Apply Masks: ###
        prod = prod[valid_mask]
        fake_norm = fake_norm[valid_mask]
        real_norm = real_norm[valid_mask]
        return 1 - torch.mean(prod / (fake_norm * real_norm))


def max_of_two(y_over_z, z_over_y):
    return torch.max(y_over_z, z_over_y)

def SSIM1(x, y):
    ### Returns a Map: ###
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ### Pad inputs to be able to return the same spatial dimensions: ###
    x_padded = nn.ReflectionPad2d(1)(x)
    y_padded = nn.ReflectionPad2d(1)(y)

    mu_x = nn.functional.avg_pool2d(x_padded, 3, 1, padding=0)
    mu_y = nn.functional.avg_pool2d(y_padded, 3, 1, padding=0)

    sigma_x = nn.functional.avg_pool2d(x_padded ** 2, 3, 1, padding=0) - mu_x ** 2
    sigma_y = nn.functional.avg_pool2d(y_padded ** 2, 3, 1, padding=0) - mu_y ** 2
    sigma_xy = nn.functional.avg_pool2d(x_padded * y_padded, 3, 1, padding=0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def SSIM2(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ### Pad inputs to be able to return the same spatial dimensions: ###
    x = nn.ReflectionPad2d(1)(x)
    y = nn.ReflectionPad2d(1)(y)

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def scale_pyramid_(self, img, num_scales):
    ### Image To Gray Scale: ###
    img = torch.mean(img, 1)
    img = torch.unsqueeze(img, 1)
    ### Initialize Image Pyramid: ###
    image_pyramid_list = [img]
    image_shape = img.size()
    h = int(image_shape[2])
    w = int(image_shape[3])
    ### Loop over scales and build image pyramid by downsampling: ###
    for i in range(num_scales):
        ### Downsample dimensions: ###
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        ### Downsample image: ###
        temp = nn.functional.upsample(img, [nh, nw], mode='nearest')
        image_pyramid_list.append(temp)
    return image_pyramid_list

### Same as the above but without converting image to GreyScale: ####
def scale_pyramid(self, img, num_scales):  # Without Turning To GreyScale
    scaled_imgs = [img]
    s = img.size()
    h = int(s[2])
    w = int(s[3])
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        temp = nn.functional.upsample(img, [nh, nw], mode='bilinear')
        scaled_imgs.append(temp)
    return scaled_imgs


# Transfer to pytorch: ###
# def Get_EdgeAware_Disparity_Smoothness_Maps(disp_list, input_img_list):
#     ### Get Gradients from each scale/output of the disparity map: ###
#     disp_gradients_x_list = [gradient_x(d) for d in disp_list]
#     disp_gradients_y_list = [gradient_y(d) for d in disp_list]
#     ### Get Gradients from each pyramid scale of the input (Left) Image: ###
#     image_gradients_x_list = [gradient_x(img) for img in input_img_list]
#     image_gradients_y_list = [gradient_y(img) for img in input_img_list]
#     ### Get Edge-Aware Image Gradients Weights: ###
#     image_gradients_weights_x_list = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
#                                       for g in image_gradients_x_list]
#     image_gradients_weights_y_list = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
#                                       for g in image_gradients_y_list]
#     ### Get Edge-Aware Disparity Smoothness Scores for each scale/output of the disparity maps: ###
#     smoothness_x_list = [disp_gradients_x_list[i] * image_gradients_weights_x_list[i]
#                          for i in range(len(image_gradients_weights_x_list))]
#     smoothness_y_list = [disp_gradients_y_list[i] * image_gradients_weights_y_list[i]
#                          for i in range(len(image_gradients_weights_y_list))]
#     # ### Pad Smoothness Score Maps (because the first order gradient removes one valid pixel, and had it been second order it would be 2 elements): ###
#     # smoothness_x_list = [torch.nn.functional.pad(k, (0, 1, 0, 0, 0, 0, 0, 0), mode='constant') for k in smoothness_x_list]
#     # smoothness_y_list = [torch.nn.functional.pad(k, (0, 0, 0, 1, 0, 0, 0, 0), mode='constant') for k in smoothness_y_list]
#     ### Concatenate(???) Smoothness Score Maps Lists: ###
#     return smoothness_x_list + smoothness_y_list  # as i understand it we are adding lists....which means CONCATENATION!!!

def Get_EdgeAware_Disparity_Smoothness_Maps(optical_flow, input_image):
    ### Get Gradients from each scale/output of the disparity map: ###
    optical_flow_gradients_x = gradient_x(optical_flow)
    optical_flow_gradients_y = gradient_y(optical_flow)
    ### Get Gradients from each pyramid scale of the input (Left) Image: ###
    image_gradients_x = gradient_x(input_image)
    image_gradients_y = gradient_y(input_image)
    ### Get Edge-Aware Image Gradients Weights: ###
    image_gradients_weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    image_gradients_weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))
    ### Get Edge-Aware Disparity Smoothness Scores for each scale/output of the disparity maps: ###
    smoothness_x = optical_flow_gradients_x * image_gradients_weights_x
    smoothness_y = optical_flow_gradients_y * image_gradients_weights_y
    return [smoothness_x, smoothness_y]


# # TODO: not used yet but is said to help
# def Get_EdgeAware_Disparity_2nd_order_Smoothness_Maps(disp_list, input_img_list):
#     ### Get Gradients from each scale/output of the disparity map: ###
#     disp_gradients_x_list = [second_order_gradient_x(d) for d in disp_list]
#     disp_gradients_y_list = [second_order_gradient_y(d) for d in disp_list]
#     ### Get Gradients from each pyramid scale of the input (Left) Image: ###
#     image_gradients_x_list = [gradient_x(img) for img in input_img_list]
#     image_gradients_y_list = [gradient_y(img) for img in input_img_list]
#     ### Get Edge-Aware Image Gradients Weights: ###
#     image_gradients_weights_x_list = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
#                                       for g in image_gradients_x_list]
#     image_gradients_weights_y_list = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
#                                       for g in image_gradients_y_list]
#     ### Get Edge-Aware Disparity Smoothness Scores for each scale/output of the disparity maps: ###
#     smoothness_x_list = [disp_gradients_x_list[i] * image_gradients_weights_x_list[i]
#                          for i in range(len(image_gradients_weights_x_list))]
#     smoothness_y_list = [disp_gradients_y_list[i] * image_gradients_weights_y_list[i]
#                          for i in range(len(image_gradients_weights_y_list))]
#     # ### Pad Smoothness Score Maps (because the first order gradient removes one valid pixel, and had it been second order it would be 2 elements): ###
#     # smoothness_x_list = [torch.nn.functional.pad(k, (0, 1, 0, 0, 0, 0, 0, 0), mode='constant') for k in smoothness_x_list]
#     # smoothness_y_list = [torch.nn.functional.pad(k, (0, 0, 0, 1, 0, 0, 0, 0), mode='constant') for k in smoothness_y_list]
#     ### Concatenate(???) Smoothness Score Maps Lists: ###
#     return smoothness_x_list + smoothness_y_list  # as i understand it we are adding lists....which means CONCATENATION!!!


def Get_EdgeAware_Disparity_2nd_order_Smoothness_Maps(optical_flow, input_image):
    ### Get Gradients from each scale/output of the disparity map: ###
    optical_flow_gradients_x = second_order_gradient_x(optical_flow)
    optical_flow_gradients_y = second_order_gradient_y(optical_flow)
    ### Get Gradients from each pyramid scale of the input (Left) Image: ###
    image_gradients_x = gradient_x(input_image)
    image_gradients_y = gradient_y(input_image)
    ### Get Edge-Aware Image Gradients Weights: ###
    image_gradients_weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    image_gradients_weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))
    ### Get Edge-Aware Disparity Smoothness Scores for each scale/output of the disparity maps: ###
    smoothness_x = optical_flow_gradients_x * image_gradients_weights_x
    smoothness_y = optical_flow_gradients_y * image_gradients_weights_y
    return [smoothness_x, smoothness_y]



# def get_occlusion_map1(left_disparity_pyramid_list, right_disparity_pyramid_list, left_disparity_pyramid_warped_list, right_disparity_pyramid_warped_list):
#     ### This occlusions map is built basically on flow field consistency: ###
#     alpha1 = 0.01
#     alpha2 = 0.5
#     left_occlusions_map_list = []
#     right_occlusions_map_list = []
#     for i in range(len(left_disparity_pyramid_list)):
#         W_f = left_disparity_pyramid_list[i]
#         W_b = right_disparity_pyramid_list[i]
#         W_f_warped = left_disparity_pyramid_warped_list[i]
#         W_b_warped = right_disparity_pyramid_warped_list[i]
#         # notation here is: where occlusions_map==1 this is a VALID PIXEL!...TODO: maybe i should call it occlusions_mask
#         left_occlusions_map_list.append(torch.abs(W_f-W_b_warped) < alpha1*torch.abs(W_f**2+W_b_warped**2) + alpha2)
#         right_occlusions_map_list.append(torch.abs(W_b-W_f_warped) < alpha1*torch.abs(W_b**2+W_f_warped**2) + alpha2)
#     return left_occlusions_map_list, right_occlusions_map_list

def get_occlusion_map1(left_image_optical_flow, right_image_optical_flow, left_image_optical_flow_warped, right_image_optical_flow_warped):
    ### This occlusions map is built basically on flow field consistency: ###
    alpha1 = 0.01
    alpha2 = 0.5
    W_f = left_image_optical_flow
    W_b = right_image_optical_flow
    W_f_warped = left_image_optical_flow_warped
    W_b_warped = right_image_optical_flow_warped
    left_occlusions_map = torch.abs(W_f - W_b_warped) < alpha1 * torch.abs(W_f ** 2 + W_b_warped ** 2) + alpha2
    right_occlusions_map = torch.abs(W_b - W_f_warped) < alpha1 * torch.abs(W_b ** 2 + W_f_warped ** 2) + alpha2

    return left_occlusions_map, right_occlusions_map


def get_torch_masked_mean_list(inputs_list, mask_list):
    outputs_list = []
    for i in range(len(inputs_list)):
        current_input_valid = inputs_list[i][mask_list[i].repeat([1, inputs_list[i].shape[1], 1, 1])]
        outputs_list.append(torch.mean(current_input_valid))
    return outputs_list

def get_torch_masked_mean_uns(input_tensor, mask_input):
    output_tensor = torch.mean(input_tensor[mask_input.repeat([1, input_tensor.shape[1], 1, 1])])
    return output_tensor


#For optical flow / disparity estimation
from RapidBase.Utils.Registration.Warp_Layers import Warp_Object, Warp_Object_OpticalFlow
class Unsupervised_Loss_Layer(nn.Module):
    def __init__(self):
        super(Unsupervised_Loss_Layer, self).__init__()
        ### Lambdas: ###
        self.alpha_image_reconstruction_SSIM = 0
        self.lambda_image_reconstruction_loss = 1
        self.lambda_disparity_first_order_smoothness = 0  # 0.1
        self.lambda_disparity_second_order_smoothness = 0.1  # 0.1
        self.lambda_disparity_left_right_consistency = 0  # 1
        self.flag_deduce_occlusions_map = False

        ### Warp Objects: ###
        self.number_of_scales = 6
        self.warp_interpolator_object = Warp_Object_OpticalFlow()  #TODO: create an object which accepts one tensor of optical flow instead of (X,Y) pairs

    def forward(self, left_image, right_image, left_optical_flow, right_optical_flow, interpolation_mode='bilinear'):

        ### Warp Left and Right Images According To Respective Estimated Disparity Maps: ###
        right_image_warped = self.warp_interpolator_object(right_image, -left_optical_flow, interpolation_mode)
        left_image_warped = self.warp_interpolator_object(right_image, -right_optical_flow, interpolation_mode)

        ### Warp Left and Right Disparity Maps Themselves According To Respective Estimated Disparity Maps: ###
        right_optical_flow_warped = self.warp_interpolator_object(right_optical_flow, -left_optical_flow, interpolation_mode)
        left_optical_flow_warped = self.warp_interpolator_object(left_optical_flow, -right_optical_flow, interpolation_mode)

        ### Get Disparity Maps Smoothness Score Maps: ###
        left_optical_flow_smoothness = Get_EdgeAware_Disparity_Smoothness_Maps(left_optical_flow, left_image)
        right_optical_flow_smoothness = Get_EdgeAware_Disparity_Smoothness_Maps(right_optical_flow, right_image)
        left_optical_flow_smoothness_2nd_order = Get_EdgeAware_Disparity_2nd_order_Smoothness_Maps(left_optical_flow, left_image)
        right_optical_flow_smoothness_2nd_order = Get_EdgeAware_Disparity_2nd_order_Smoothness_Maps(right_optical_flow, right_image)

        #################################### Get Occlusion Masks: ####################################
        if self.flag_deduce_occlusions_map:
            left_non_occluded_map, right_non_occluded_map = get_occlusion_map1(left_optical_flow,
                                                                               right_optical_flow,
                                                                               left_optical_flow_warped,
                                                                               right_optical_flow_warped)
        else:
            left_non_occluded_map = torch.ones_like(left_optical_flow).byte()
            right_non_occluded_map = torch.ones_like(right_optical_flow).byte()

        ##############################################################################################################################################
        ### Accumulate Losses: ###
        # (1). Image Reconstruction Loss:
        #   (a). L1 Image Reconstruction Loss:
        l1_left = torch.abs(left_image - right_image_warped)
        l1_right = torch.abs(right_image - left_image_warped)
        l1_reconstruction_loss_left = get_torch_masked_mean_uns(l1_left, left_non_occluded_map)
        l1_reconstruction_loss_right = get_torch_masked_mean_uns(l1_right, right_non_occluded_map)
        #   (b). SSIM Image Reconstruction Loss:
        ssim_loss_left = SSIM2(right_image_warped, left_image)
        ssim_loss_right = SSIM2(left_image_warped, right_image)
        ssim_loss_left_mean = get_torch_masked_mean_uns(ssim_loss_left, left_non_occluded_map)
        ssim_loss_right_mean = get_torch_masked_mean_uns(ssim_loss_right, right_non_occluded_map)
        #   (c). Combining L1 & SSIM Losses:
        image_loss_right = self.alpha_image_reconstruction_SSIM * ssim_loss_right_mean + (1-self.alpha_image_reconstruction_SSIM) * l1_reconstruction_loss_left
        image_loss_left = self.alpha_image_reconstruction_SSIM * ssim_loss_left_mean + (1-self.alpha_image_reconstruction_SSIM) * l1_reconstruction_loss_right
        image_reconstruction_loss = image_loss_left + image_loss_right

        # (2). Disparity Maps Consistency:
        left_optical_flow_consistency_map = torch.abs(right_optical_flow_warped - left_optical_flow)
        right_optical_flow_consistency_map = torch.abs(left_optical_flow_warped - right_optical_flow)
        left_optical_flow_consistency_loss = get_torch_masked_mean_uns(left_optical_flow_consistency_map, left_non_occluded_map)
        right_optical_flow_consistency_loss = get_torch_masked_mean_uns(right_optical_flow_consistency_map, right_non_occluded_map)
        optical_flow_left_right_consistency_loss = left_optical_flow_consistency_loss + right_optical_flow_consistency_loss

        # (3). Disparity 1st Order Smoothness:
        left_optical_flow_smoothness_loss = torch.abs(left_optical_flow_smoothness)
        right_optical_flow_smoothness_loss = torch.abs(right_optical_flow_smoothness)
        optical_flow_first_order_smoothness_loss = left_optical_flow_smoothness_loss + right_optical_flow_smoothness_loss

        # (4). Disparity 2nd Order Smoothness:
        left_optical_flow_smoothness_2nd_order_loss = torch.abs(left_optical_flow_smoothness_2nd_order)
        right_optical_flow_smoothness_2nd_order_loss = torch.abs(right_optical_flow_smoothness_2nd_order)
        optical_flow_second_order_smoothness_loss = left_optical_flow_smoothness_2nd_order_loss + right_optical_flow_smoothness_2nd_order_loss

        ### Combine Losses: ###
        total_loss = (self.lambda_image_reconstruction_loss * image_reconstruction_loss +
                      self.lambda_disparity_first_order_smoothness * optical_flow_first_order_smoothness_loss +
                      self.lambda_disparity_second_order_smoothness * optical_flow_second_order_smoothness_loss +
                      self.lambda_disparity_left_right_consistency * optical_flow_left_right_consistency_loss)

        return total_loss


class Unsupervised_OnlyForward_Loss_Layer(nn.Module):
    def __init__(self):
        super(Unsupervised_OnlyForward_Loss_Layer, self).__init__()
        ### Lambdas: ###
        self.alpha_image_reconstruction_SSIM = 0
        self.lambda_image_reconstruction_loss = 1
        self.lambda_disparity_first_order_smoothness = 0  # 0.1
        self.lambda_disparity_second_order_smoothness = 0.1  # 0.1
        self.lambda_disparity_left_right_consistency = 0  # 1
        self.flag_deduce_occlusions_map = False

        ### Warp Objects: ###
        self.number_of_scales = 6
        self.warp_interpolator_object = Warp_Object_OpticalFlow()  # TODO: create an object which accepts one tensor of optical flow instead of (X,Y) pairs

    def forward(self, left_image, right_image, left_optical_flow, valid_mask, interpolation_mode='bilinear'):

        ### Warp Left and Right Images According To Respective Estimated Disparity Maps: ###
        #TODO: one has to be EXTREMELY careful in notation here. what is left_image, right_image, left_optical_flow sign and right_optical_flow sign
        right_image_warped = self.warp_interpolator_object(right_image, +1*left_optical_flow, interpolation_mode)

        ### Get Disparity Maps Smoothness Score Maps: ###
        left_optical_flow_smoothness = Get_EdgeAware_Disparity_Smoothness_Maps(left_optical_flow, left_image)
        left_optical_flow_smoothness_2nd_order = Get_EdgeAware_Disparity_2nd_order_Smoothness_Maps(left_optical_flow, left_image)
        left_optical_flow_smoothness_x, left_optical_flow_smoothness_y = left_optical_flow_smoothness
        left_optical_flow_smoothness_2nd_order_x, left_optical_flow_smoothness_2nd_order_y = left_optical_flow_smoothness_2nd_order

        #################################### Get Occlusion Masks: ####################################
        # left_non_occluded_map = torch.ones_like(left_optical_flow[:,0:1,:,:]).bool()

        ##############################################################################################################################################
        ### Accumulate Losses: ###
        # (1). Image Reconstruction Loss:
        #   (a). L1 Image Reconstruction Loss:
        l1_left = torch.abs(left_image - right_image_warped)
        l1_reconstruction_loss_left = get_torch_masked_mean_uns(l1_left, valid_mask)
        image_reconstruction_loss = l1_reconstruction_loss_left

        # (3). Disparity 1st Order Smoothness:
        left_optical_flow_smoothness_loss = 0.5 * (torch.abs(left_optical_flow_smoothness_x) + torch.abs(left_optical_flow_smoothness_y))
        optical_flow_first_order_smoothness_loss = get_torch_masked_mean_uns(left_optical_flow_smoothness_loss, valid_mask)

        # (4). Disparity 2nd Order Smoothness:
        left_optical_flow_smoothness_2nd_order_loss = 0.5 * (torch.abs(left_optical_flow_smoothness_2nd_order_x) + torch.abs(left_optical_flow_smoothness_2nd_order_y))
        optical_flow_second_order_smoothness_loss = get_torch_masked_mean_uns(left_optical_flow_smoothness_2nd_order_loss, valid_mask)

        ### Combine Losses: ###
        total_loss = (self.lambda_image_reconstruction_loss * image_reconstruction_loss +
                      self.lambda_disparity_first_order_smoothness * optical_flow_first_order_smoothness_loss +
                      self.lambda_disparity_second_order_smoothness * optical_flow_second_order_smoothness_loss)

        return total_loss



# class Unsupervised_Loss_Layer(nn.Module):
#     def __init__(self):
#         super(Unsupervised_Loss_Layer, self).__init__()
#         ### Lambdas: ###
#         self.alpha_image_reconstruction_SSIM = 0
#         self.lambda_image_reconstruction_loss = 1
#         self.lambda_disparity_first_order_smoothness = 0  # 0.1
#         self.lambda_disparity_second_order_smoothness = 0.1  # 0.1
#         self.lambda_disparity_left_right_consistency = 0  # 1
#         self.flag_deduce_occlusions_map = False
#
#         ### Warp Objects: ###
#         self.number_of_scales = 6
#         self.warp_interpolator_objects_list = []
#         for i in arange(self.number_of_scales):
#             self.warp_interpolator_objects_list.append(Warp_Object())
#
#     def forward(self, left_image_pyramid_list, right_image_pyramid_list, left_disparity_pyramid_list, right_disparity_pyramid_list, warp_interpolator_object, GT_left_occlusions_map_list=None, GT_right_occlusions_map_list=None):
#         ### Default Parameters: ###
#
#         ### If we only inserted a Tensor (as opposed to a list representing the different scales of prediction), then turn the appropriate variable to a 1 element list: ###
#         if type(left_disparity_pyramid_list) is not list:
#             left_disparity_pyramid_list = [left_disparity_pyramid_list]
#             right_disparity_pyramid_list = [right_disparity_pyramid_list]
#             number_of_pyramid_scales = 1
#         else:
#             number_of_pyramid_scales = len(left_disparity_pyramid_list)
#
#         # ### Get Image Pyramids if needed (if we only inserted the original left and right images): ###
#         # if type(left_image_pyramid_list) is not list:
#         #     left_image_pyramid_list = scale_pyramid_(input_left, number_of_pyramid_scales)
#         #     right_image_pyramid_list = scale_pyramid_(input_right, number_of_pyramid_scales)
#
#         ### Warp Left and Right Images According To Respective Estimated Disparity Maps: ###
#         number_of_scales = len(left_image_pyramid_list)
#         right_image_pyramid_warped_list = [self.warp_interpolator_objects_list[i](
#             right_image_pyramid_list[i], -1*left_disparity_pyramid_list[i], torch.zeros_like(left_disparity_pyramid_list[i]), 'bilinear') for i in range(number_of_scales)]
#         left_image_pyramid_warped_list = [self.warp_interpolator_objects_list[i](
#             left_image_pyramid_list[i], +1*right_disparity_pyramid_list[i], torch.zeros_like(right_disparity_pyramid_list[i]), 'bilinear') for i in range(number_of_scales)]
#
#         ### Warp Left and Right Disparity Maps Themselves According To Respective Estimated Disparity Maps: ###
#         right_disparity_pyramid_warped_list = [self.warp_interpolator_objects_list[i](
#             right_disparity_pyramid_list[i], -1*left_disparity_pyramid_list[i], torch.zeros_like(left_disparity_pyramid_list[i]), 'bilinear') for i in range(number_of_scales)]
#         left_disparity_pyramid_warped_list = [self.warp_interpolator_objects_list[i](
#             left_disparity_pyramid_list[i], +1*right_disparity_pyramid_list[i], torch.zeros_like(right_disparity_pyramid_list[i]), 'bilinear') for i in range(number_of_scales)]
#
#         ### Get Disparity Maps Smoothness Score Maps: ###
#         left_disparity_pyramid_smoothness_list = Get_EdgeAware_Disparity_Smoothness_Maps(
#             left_disparity_pyramid_list, left_image_pyramid_list)
#         right_disparity_pyramid_smoothness_list = Get_EdgeAware_Disparity_Smoothness_Maps(
#             right_disparity_pyramid_list, right_image_pyramid_list)
#         left_disparity_pyramid_smoothness_2nd_order_list = Get_EdgeAware_Disparity_2nd_order_Smoothness_Maps(
#             left_disparity_pyramid_list, left_image_pyramid_list)
#         right_disparity_pyramid_smoothness_2nd_order_list = Get_EdgeAware_Disparity_2nd_order_Smoothness_Maps(
#             right_disparity_pyramid_list, right_image_pyramid_list)
#
#         #################################### Get Occlusion Masks: ####################################
#         ### if we have GT occlusions maps use them: ###
#         if GT_left_occlusions_map_list is not None:
#             left_non_occluded_map_list = GT_left_occlusions_map_list
#         else:
#             ### if GT occlusions are NOT available then either deduce them or simply make all pixels valid and deal with it: ###
#             if self.flag_deduce_occlusions_map:
#                 # TODO: input self.flag_deduce_occlusions_map as input to function because at the beginning pretty much everything is occluded because network output is random
#                 # left_occlusions_map, right_occlusions_map = get_occlusion_map2(left_disparity_pyramid_list, right_disparity_pyramid_list, left_disparity_pyramid_warped_list, right_disparity_pyramid_warped_list)  # left_occlusions_map, right_occlusions_map = get_occlusion_map3(left_disparity_pyramid_list, right_disparity_pyramid_list, left_disparity_pyramid_warped_list, right_disparity_pyramid_warped_list, left_image_pyramid_list, right_image_pyramid_list, left_image_pyramid_warped_list, right_image_pyramid_warped_list)
#                 left_non_occluded_map_list, right_non_occluded_map_list = get_occlusion_map1(
#                     left_disparity_pyramid_list, right_disparity_pyramid_list, left_disparity_pyramid_warped_list, right_disparity_pyramid_warped_list)
#             else:
#                 left_non_occluded_map_list = [torch.ones_like(l).byte() for l in left_disparity_pyramid_list]
#                 right_non_occluded_map_list = [torch.ones_like(l).byte() for l in right_disparity_pyramid_list]
#         if GT_right_occlusions_map_list is not None:
#             right_non_occluded_map_list = GT_right_occlusions_map_list
#
#         ##############################################################################################################################################
#         ### Accumulate Losses: ###
#         # (1). Image Reconstruction Loss:
#         #   (a). L1 Image Reconstruction Loss:
#         l1_left = [torch.abs(left_image_pyramid_list[i] - right_image_pyramid_warped_list[i])
#                    for i in range(number_of_scales)]
#         l1_right = [torch.abs(right_image_pyramid_list[i] - left_image_pyramid_warped_list[i])
#                     for i in range(number_of_scales)]
#         l1_reconstruction_loss_left_list = get_torch_masked_mean_list(l1_left, left_non_occluded_map_list)
#         l1_reconstruction_loss_right_list = get_torch_masked_mean_list(l1_right, right_non_occluded_map_list)
#         #   (b). SSIM Image Reconstruction Loss:
#         ssim_loss_left_map_list = [SSIM2(right_image_pyramid_warped_list[i],
#                                          left_image_pyramid_list[i]) for i in range(number_of_scales)]
#         ssim_loss_right_map_list = [SSIM2(left_image_pyramid_warped_list[i],
#                                           right_image_pyramid_list[i]) for i in range(number_of_scales)]
#         ssim_loss_left_mean_list = get_torch_masked_mean_list(ssim_loss_left_map_list, left_non_occluded_map_list)
#         ssim_loss_right_mean_list = get_torch_masked_mean_list(ssim_loss_right_map_list, right_non_occluded_map_list)
#         #   (c). Combining L1 & SSIM Losses:
#         image_loss_right = [self.alpha_image_reconstruction_SSIM * ssim_loss_right_mean_list[i] +
#                             (1 - self.alpha_image_reconstruction_SSIM) * l1_reconstruction_loss_right_list[i] for i in range(number_of_scales)]
#         image_loss_left = [self.alpha_image_reconstruction_SSIM * ssim_loss_left_mean_list[i] +
#                            (1 - self.alpha_image_reconstruction_SSIM) * l1_reconstruction_loss_left_list[i] for i in range(number_of_scales)]
#         image_loss1 = [(image_loss_left[i] + image_loss_right[i]) for i in range(number_of_scales)]
#         image_reconstruction_loss = sum(image_loss1)
#
#         # current_input_valid = l1_left[0][left_non_occluded_map_list[0].repeat([1,l1_left[0].shape[1],1,1])]
#         # outputs_list.append(torch.mean(current_input_valid))
#
#         # (2). Disparity Maps Consistency:
#         left_disparity_pyramid_consistency_map_list = [
#             torch.abs(right_disparity_pyramid_warped_list[i] - left_disparity_pyramid_list[i]) for i in range(number_of_scales)]
#         left_disparity_pyramid_consistency_loss_list = get_torch_masked_mean_list(
#             left_disparity_pyramid_consistency_map_list, left_non_occluded_map_list)
#         right_disparity_pyramid_consistency_map_list = [
#             torch.abs(left_disparity_pyramid_warped_list[i] - right_disparity_pyramid_list[i]) for i in range(number_of_scales)]
#         right_disparity_pyramid_consistency_loss_list = get_torch_masked_mean_list(
#             right_disparity_pyramid_consistency_map_list, right_non_occluded_map_list)
#         disparity_left_right_consistency_loss = sum(
#             left_disparity_pyramid_consistency_loss_list + right_disparity_pyramid_consistency_loss_list)
#
#         # (3). Disparity 1st Order Smoothness:
#         left_disparity_pyramid_smoothness_loss_list = [torch.mean(
#             torch.abs(left_disparity_pyramid_smoothness_list[i])) / 2 ** i for i in range(number_of_scales)]
#         right_disparity_pyramid_smoothness_loss_list = [torch.mean(
#             torch.abs(right_disparity_pyramid_smoothness_list[i])) / 2 ** i for i in range(number_of_scales)]
#         disparity_first_order_smoothness_loss = sum(
#             left_disparity_pyramid_smoothness_loss_list + right_disparity_pyramid_smoothness_loss_list)
#
#         # (5). Disparity 2nd Order Smoothness:
#         left_disparity_pyramid_smoothness_2nd_order_loss_list = [torch.mean(
#             torch.abs(left_disparity_pyramid_smoothness_2nd_order_list[i])) / 2 ** i for i in range(number_of_scales)]
#         right_disparity_pyramid_smoothness_2nd_order_loss_list = [torch.mean(
#             torch.abs(right_disparity_pyramid_smoothness_2nd_order_list[i])) / 2 ** i for i in range(number_of_scales)]
#         disparity_second_order_smoothness_loss = sum(
#             left_disparity_pyramid_smoothness_2nd_order_loss_list + right_disparity_pyramid_smoothness_2nd_order_loss_list)
#
#         ### Combine Losses: ###
#         total_loss = (self.lambda_image_reconstruction_loss * image_reconstruction_loss +
#                       self.lambda_disparity_first_order_smoothness * disparity_first_order_smoothness_loss +
#                       self.lambda_disparity_second_order_smoothness * disparity_second_order_smoothness_loss +
#                       self.lambda_disparity_left_right_consistency * disparity_left_right_consistency_loss)
#
#         return total_loss



def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)



# (1). Define GAN Validity loss: [vanilla | lsgan | wgan-gp]
class GAN_validity_loss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0): #TODO: what about using -1 for fake label and changing the loss terms accordingly?
        super(GAN_validity_loss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        #TODO: understand whether there's any benefit in incorporating MSELoss (maybe even somehow incorporate the mixed targets trick for classification) and WGAN validity loss

        # Choose wanted gan Validity Loss:
        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            def wgan_loss(input, target):
                # target is boolean:
                # if target=1 then we want the discriminator output to be 1 and want to minimize loss so we use -1*input.mean()
                # if target=0 then again wanting to minimize loss we will use a loss of +1*input.mean()
                #(*). NOTE: in this case there is no real insentive as far as i can see for a certain range of outputs as the target value doesn't enter the loss calculation explicitely.
                #           the insentive is basically to output as much as possible for real and as low as possible for fake
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))


    def get_target_label_in_correct_form(self, discriminator_output, flag_is_target_real):
        # If gan type is wgan-gp then we simply return the boolean itself
        if self.gan_type == 'wgan-gp':
            return flag_is_target_real

        # If gan type is NOT wgan-gp then we return a matrix the size of the discriminator output filled
        # with the correct label according to flag_is_target_real:
        if flag_is_target_real:
            return torch.empty_like(discriminator_output).fill_(self.real_label_val)
        else:
            return torch.empty_like(discriminator_output).fill_(self.fake_label_val)


    def forward(self, discriminator_output, target_is_real):
        target_label_correct_size = self.get_target_label_in_correct_form(discriminator_output, target_is_real)
        loss = self.loss(discriminator_output, target_label_correct_size)
        return loss




# (2). Relativistic GAN validity Loss:
class Relativistic_GAN_validity_loss(nn.Module):
    def __init__(self):
        super(Relativistic_GAN_validity_loss, self).__init__()
        self.current_discriminator_real_images_validity_loss = 0
        self.current_discriminator_fake_images_validity_loss = 0
    def forward(self, GAN_validity_loss_function, D_output_real_image_batch_validity, D_output_fake_image_batch_validity):
        self.current_discriminator_real_images_validity_loss = GAN_validity_loss_function(D_output_real_image_batch_validity - torch.mean(D_output_fake_image_batch_validity), True) #TODO: i think this should be switched between False & True
        self.current_discriminator_fake_images_validity_loss = GAN_validity_loss_function(D_output_fake_image_batch_validity - torch.mean(D_output_real_image_batch_validity), False)
        return (self.current_discriminator_real_images_validity_loss + self.current_discriminator_fake_images_validity_loss) / 2



# (3). Gradient Penalty Loss:
class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()

        # TODO: what is register_buffer() property of the nn.Module Class???
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)  #self.grad_outputs pre-exist as part of the nn.Module class!

    # Get Gradient Outputs:
    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0) #make sure that the wanted functionality here actually works
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        # interp = a random alpha is assigned and interp is a linear combination of alpha*fake+(1-alpha)*real.
        current_grad_outputs_resized_if_needed = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit,
                                          inputs=interp,
                                          grad_outputs=current_grad_outputs_resized_if_needed,
                                          create_graph=True, #Why True?
                                          retain_graph=True, #Why True?
                                          only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)
        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss



#(5). Contextual Loss:
class TensorAxis:
    N = 0
    H = 1
    W = 2
    C = 3

class CSFlow:
    def __init__(self, sigma=float(1.0), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization=TensorAxis.C):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = torch.exp((self.b - scaled_distances) / self.sigma)
        # self.cs_weights_before_normalization = 1 / (1 + scaled_distances)
        # self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)
        self.reduce_sum = torch.sum(self.cs_weights_before_normalization, TensorAxis.C)
        self.cs_NHWC = self.cs_weights_before_normalization / torch.unsqueeze(self.reduce_sum, 3)

    # def reversed_direction_CS(self):
    #     cs_flow_opposite = CSFlow(self.sigma, self.b)
    #     cs_flow_opposite.raw_distances = self.raw_distances
    #     work_axis = [TensorAxis.H, TensorAxis.W]
    #     relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
    #     cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
    #     return cs_flow_opposite

    # --
    @staticmethod
    def create_using_L2(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        input2_pixels_flattened = torch.reshape(I_features, (sI[0], -1, sI[3]))
        input1_pixels_flattened = torch.reshape(T_features, (sI[0], -1, sT[3]))
        r_Ts = torch.sum(input1_pixels_flattened * input1_pixels_flattened, 2)
        r_Is = torch.sum(input2_pixels_flattened * input2_pixels_flattened, 2)
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec, r_T, r_I = input2_pixels_flattened[i], input1_pixels_flattened[i], r_Ts[i], r_Is[i]
            A = Tvec @ torch.transpose(Ivec, 0, 1)  # (matrix multiplication)
            cs_flow.A = A
            # A = tf.matmul(Tvec, tf.transpose(Ivec))
            r_T = torch.reshape(r_T, [-1, 1])  # turn to column vector
            dist = r_T - 2 * A + r_I
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_L1(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        input2_pixels_flattened = torch.reshape(I_features, (sI[0], -1, sI[3]))
        input1_pixels_flattened = torch.reshape(T_features, (sI[0], -1, sT[3]))
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec = input2_pixels_flattened[i], input1_pixels_flattened[i]
            dist = torch.abs(torch.sum(Ivec.unsqueeze(1) - Tvec.unsqueeze(0), dim=2))
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(1), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        # prepare feature before calculating cosine distance
        T_features, I_features = cs_flow.center_by_T(T_features, I_features)
        T_features = CSFlow.l2_normalize_channelwise(T_features)
        I_features = CSFlow.l2_normalize_channelwise(I_features)

        # work seperatly for each example in dim 1
        cosine_dist_l = []
        N = T_features.size()[0]
        for i in range(N):
            T_features_i = T_features[i, :, :, :].unsqueeze_(0)  # 1HWC --> 1CHW
            I_features_i = I_features[i, :, :, :].unsqueeze_(0).permute((0, 3, 1, 2))
            patches_PC11_i = cs_flow.patch_decomposition(T_features_i)  # 1HWC --> PC11, with P=H*W
            cosine_dist_i = torch.nn.functional.conv2d(I_features_i, patches_PC11_i)
            cosine_dist_1HWC = cosine_dist_i.permute((0, 2, 3, 1))
            cosine_dist_l.append(cosine_dist_i.permute((0, 2, 3, 1)))  # back to 1HWC

        cs_flow.cosine_dist = torch.cat(cosine_dist_l, dim=0)

        cs_flow.raw_distances = - (cs_flow.cosine_dist - 1) / 2  ### why -

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    def calc_relative_distances(self, axis=TensorAxis.C):
        epsilon = 1e-5
        div = torch.min(self.raw_distances, dim=axis, keepdim=True)[0]
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    @staticmethod
    def sum_normalize(cs, axis=TensorAxis.C):
        reduce_sum = torch.sum(cs, dim=axis, keepdim=True)
        cs_normalize = torch.div(cs, reduce_sum)
        return cs_normalize

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size
        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT = T_features.mean(0, keepdim=True).mean(1, keepdim=True).mean(2, keepdim=True)
        self.varT = T_features.var(0, keepdim=True).var(1, keepdim=True).var(2, keepdim=True)
        self.T_features_centered = T_features - self.meanT
        self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = features.norm(p=2, dim=TensorAxis.C, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, T_features):
        # 1HWC --> 11PC --> PC11, with P=H*W
        (N, H, W, C) = T_features.shape
        P = H * W
        patches_PC11 = T_features.reshape(shape=(1, 1, P, C)).permute(dims=(2, 3, 0, 1))
        return patches_PC11

    @staticmethod
    def pdist2(x, keepdim=False):
        sx = x.shape
        x = x.reshape(shape=(sx[0], sx[1] * sx[2], sx[3]))
        differences = x.unsqueeze(2) - x.unsqueeze(1)
        distances = torch.sum(differences**2, -1)
        if keepdim:
            distances = distances.reshape(shape=(sx[0], sx[1], sx[2], sx[3]))
        return distances

    @staticmethod
    def calcR_static(sT, order='C', deformation_sigma=0.05):
        # oreder can be C or F (matlab order)
        pixel_count = sT[0] * sT[1]

        rangeRows = range(0, sT[1])
        rangeCols = range(0, sT[0])
        Js, Is = np.meshgrid(rangeRows, rangeCols)
        row_diff_from_first_row = Is
        col_diff_from_first_col = Js

        row_diff_from_first_row_3d_repeat = np.repeat(row_diff_from_first_row[:, :, np.newaxis], pixel_count, axis=2)
        col_diff_from_first_col_3d_repeat = np.repeat(col_diff_from_first_col[:, :, np.newaxis], pixel_count, axis=2)

        rowDiffs = -row_diff_from_first_row_3d_repeat + row_diff_from_first_row.flatten(order).reshape(1, 1, -1)
        colDiffs = -col_diff_from_first_col_3d_repeat + col_diff_from_first_col.flatten(order).reshape(1, 1, -1)
        R = rowDiffs ** 2 + colDiffs ** 2
        R = R.astype(np.float32)
        R = np.exp(-(R) / (2 * deformation_sigma ** 2))
        return R

# --------------------------------------------------
#           CX loss
# --------------------------------------------------
def CX_loss(T_features, I_features, deformation=False, dis=False, flag_L2_or_dotP=1):
    # since this originally Tensorflow implemntation
    # we modify all tensors to be as TF convention and not as the convention of pytorch.
    def from_pt2tf(Tpt):
        Ttf = Tpt.permute(0, 2, 3, 1)
        return Ttf
    # N x C x H x W --> N x H x W x C
    T_features_tf = from_pt2tf(T_features)
    I_features_tf = from_pt2tf(I_features)

    # Decide whether L2 or dotP:
    if flag_L2_or_dotP == 1:
        cs_flow = CSFlow.create_using_L2(I_features_tf, T_features_tf, sigma=1.0)
    elif flag_L2_or_dotP == 2:
        cs_flow = CSFlow.create_using_dotP(I_features_tf, T_features_tf, sigma=1.0)


    cs = cs_flow.cs_NHWC

    if deformation:
        deforma_sigma = 0.001
        sT = T_features_tf.shape[1:2 + 1]
        R = CSFlow.calcR_static(sT, deformation_sigma=deforma_sigma)
        cs *= torch.Tensor(R).unsqueeze(dim=0).cuda()

    if dis:
        CS = []
        k_max_NC = torch.max(torch.max(cs, dim=1)[1], dim=1)[1]
        indices = k_max_NC.cpu()
        N, C = indices.shape
        for i in range(N):
            CS.append((C - len(torch.unique(indices[i, :]))) / C)
        score = torch.FloatTensor(CS)
    else:
        # reduce_max X and Y dims
        # cs = CSFlow.pdist2(cs,keepdim=True)
        k_max_NC = torch.max(torch.max(cs, dim=1)[0], dim=1)[0]
        # reduce mean over C dim
        CS = torch.mean(k_max_NC, dim=1)

        CX_as_loss = 1 - CS
        CX_loss = -torch.log(1 - CX_as_loss)
        score = torch.mean(CX_loss)
    return score


class Contextual_Loss(nn.Module):
    def __init__(self, device=torch.device('cpu'), flag_L2_or_dotP=1):
        super(Contextual_Loss, self).__init__()
        self.flag_L2_or_dotP = flag_L2_or_dotP
    def forward(self, x, y):
        return CX_loss(x, y, deformation=False, dis=False, flag_L2_or_dotP=self.flag_L2_or_dotP)


class VRT_ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 band_width: float = 0.5,
                 loss_type: str = 'cosine',
                 use_vgg: bool = False,
                 vgg_layer: str = 'relu3_4',
                 device: int = 0):

        super(VRT_ContextualLoss, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        assert loss_type in LOSS_TYPES,\
            f'select a loss type from {LOSS_TYPES}.'

        self.band_width = band_width
        self.device = device

        if use_vgg:
            self.vgg_model = VGG19().to(self.device)
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

            self.vgg_mean = self.vgg_mean.to(self.device)
            self.vgg_std = self.vgg_std.to(self.device)

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3,\
                'VGG model takes 3 chennel images.'

            # normalization
            x = x.sub(self.vgg_mean).div(self.vgg_std)
            y = y.sub(self.vgg_mean).div(self.vgg_std)

            # picking up vgg feature maps
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)

        return F.contextual_loss(x, y, self.band_width)


#(6). Gradient Sensitive Loss:
class Gradient_Sensitive_Loss_1(nn.Module):
    def __init__(self, lambda_p=1):
        super(Gradient_Sensitive_Loss_1, self).__init__()
        self.gradient_x_kernel_numpy = np.array([[[1, 0, -1],
                                      [2, 0, -2],
                                      [1, 0, -1]]])
        self.gradient_y_kernel_numpy = np.array([[[1, 2, 1],
                                      [0, 0, 0],
                                      [-1, -2, -1]]])
        self.flag_already_set_kernel_size = False
        self.gradient_x_kernel_Tensor = torch.Tensor(self.gradient_x_kernel_numpy)
        self.gradient_y_kernel_Tensor = torch.Tensor(self.gradient_y_kernel_numpy)
        self.L_g_x_Loss = nn.L1Loss()
        self.L_g_y_Loss = nn.L1Loss()
        self.L_p_Loss = nn.L1Loss()
        self.Masked_L_p_Loss = MaskedL1Loss()
        self.lambda_p = lambda_p
        self.number_of_kernel_channels = 0


    def to_device(self, device='cpu'):
        self.gradient_x_kernel_Tensor = torch.Tensor(self.gradient_x_kernel_numpy)
        self.gradient_y_kernel_Tensor = torch.Tensor(self.gradient_y_kernel_numpy)
        self.L_g_x_Loss = nn.L1Loss()
        self.L_g_y_Loss = nn.L1Loss()
        self.L_p_Loss = nn.L1Loss()

    def forward(self, x, y):
        x,y = x[0], y[0]
        number_of_input_channels = x.shape[1]
        #Make Tensors
        if self.number_of_kernel_channels != number_of_input_channels:
            self.gradient_x_kernel_numpy_multidim = self.gradient_x_kernel_numpy.repeat(repeats=number_of_input_channels,axis=0)
            self.gradient_y_kernel_numpy_multidim = self.gradient_y_kernel_numpy.repeat(repeats=number_of_input_channels,axis=0)
            self.gradient_x_kernel_Tensor = torch.Tensor(self.gradient_x_kernel_numpy_multidim)
            self.gradient_y_kernel_Tensor = torch.Tensor(self.gradient_y_kernel_numpy_multidim)
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.view((1,number_of_input_channels,3,3))
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.view((1,number_of_input_channels,3,3))
            self.number_of_kernel_channels = number_of_input_channels
            self.flag_already_set_kernel_size = True

        #Make Sure Devices Are The Same:
        if self.gradient_x_kernel_Tensor.device != x.device:
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.to(x.device)
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.to(x.device)

        ### Get Gradient Strength Measure (M): ###
        G_x = torch.nn.functional.conv2d(x, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        G_y = torch.nn.functional.conv2d(x, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        M = (G-G.min()) / (G.max()-G.min())
        #Get Decomposition of x,y into x=M*x+(1-M)*x, y=M*y+(1-M)*y

        ### Emphasis x&y using M on gradients: ###
        M_x = M * x/number_of_input_channels
        M_y = M * y/number_of_input_channels
        M_x_complete = (1-M) * x/number_of_input_channels
        M_y_complete = (1-M) * y/number_of_input_channels

        ### Get Decomposition's Gradients: ###
        M_x_Gx = torch.nn.functional.conv2d(M_x, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        M_x_Gy = torch.nn.functional.conv2d(M_x, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        M_y_Gx = torch.nn.functional.conv2d(M_y, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        M_y_Gy = torch.nn.functional.conv2d(M_y, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)

        ### Get Gradient Sensitive Loss: ###
        L_g = self.L_g_x_Loss(M_x_Gx, M_y_Gx) + self.L_g_y_Loss(M_x_Gy, M_y_Gy)  #make sure emphasized images gradients match
        L_p = self.L_p_Loss(M_x_complete, M_y_complete)  #make sure the (1-M) of the images match
        return (1-self.lambda_p) * L_g + self.lambda_p * L_p


class Gradient_Sensitive_Loss_2(Gradient_Sensitive_Loss_1):
    def __init__(self, lambda_p=1):
        super(Gradient_Sensitive_Loss_2, self).__init__()

    def forward(self, x, y):
        x, y = x[0], y[0]
        number_of_input_channels = x.shape[1]
        # Make Tensors
        if self.number_of_kernel_channels != number_of_input_channels:
            self.gradient_x_kernel_numpy_multidim = self.gradient_x_kernel_numpy.repeat(
                repeats=number_of_input_channels, axis=0)
            self.gradient_y_kernel_numpy_multidim = self.gradient_y_kernel_numpy.repeat(repeats=number_of_input_channels, axis=0)
            self.gradient_x_kernel_Tensor = torch.Tensor(self.gradient_x_kernel_numpy_multidim)
            self.gradient_y_kernel_Tensor = torch.Tensor(self.gradient_y_kernel_numpy_multidim)
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.view((1, number_of_input_channels, 3, 3))
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.view((1, number_of_input_channels, 3, 3))
            self.number_of_kernel_channels = number_of_input_channels
            self.flag_already_set_kernel_size = True

        ### Make Sure Devices Are The Same: ###
        if self.gradient_x_kernel_Tensor.device != x.device:
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.to(x.device)
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.to(x.device)

        ### Get Gradient Strength Measure (M): ###
        G_x = torch.nn.functional.conv2d(x, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        G_y = torch.nn.functional.conv2d(x, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        M = (G - G.min()) / (G.max() - G.min())
        # Get Decomposition of x,y into x=M*x+(1-M)*x, y=M*y+(1-M)*y

        ### Emphasis x&y using M on gradients: ###
        M_x = M * x / number_of_input_channels
        M_y = M * y / number_of_input_channels
        M_x_complete = (1 - M) * x / number_of_input_channels
        M_y_complete = (1 - M) * y / number_of_input_channels

        ### Get Decomposition's Gradients: ###
        x_Gx = torch.nn.functional.conv2d(x, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        x_Gy = torch.nn.functional.conv2d(x, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        y_Gx = torch.nn.functional.conv2d(y, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        y_Gy = torch.nn.functional.conv2d(y, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)

        ### Get Gradient Sensitive Loss: ###
        L_g = self.L_g_x_Loss(x_Gx, y_Gx) + self.L_g_y_Loss(x_Gy, y_Gy)  #make sure gradients match
        L_p = self.L_p_Loss(x, y)  #make sure pixel values match
        return (1 - self.lambda_p) * L_g + self.lambda_p * L_p


class Gradient_Sensitive_Loss_3(Gradient_Sensitive_Loss_1):
    def __init__(self, lambda_p=1):
        super(Gradient_Sensitive_Loss_3, self).__init__()

    def forward(self, x, y):
        x, y = x[0], y[0]
        number_of_input_channels = x.shape[1]
        # Make Tensors
        if self.number_of_kernel_channels != number_of_input_channels:
            self.gradient_x_kernel_numpy_multidim = self.gradient_x_kernel_numpy.repeat(
                repeats=number_of_input_channels, axis=0)
            self.gradient_y_kernel_numpy_multidim = self.gradient_y_kernel_numpy.repeat(repeats=number_of_input_channels, axis=0)
            self.gradient_x_kernel_Tensor = torch.Tensor(self.gradient_x_kernel_numpy_multidim)
            self.gradient_y_kernel_Tensor = torch.Tensor(self.gradient_y_kernel_numpy_multidim)
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.view((1, number_of_input_channels, 3, 3))
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.view((1, number_of_input_channels, 3, 3))
            self.number_of_kernel_channels = number_of_input_channels
            self.flag_already_set_kernel_size = True

        ### Make Sure Devices Are The Same: ###
        if self.gradient_x_kernel_Tensor.device != x.device:
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.to(x.device)
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.to(x.device)

        ### Get Gradient Strength Measure (M): ###
        G_x = torch.nn.functional.conv2d(x, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        G_y = torch.nn.functional.conv2d(x, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        M = (G - G.min()) / (G.max() - G.min())
        # Get Decomposition of x,y into x=M*x+(1-M)*x, y=M*y+(1-M)*y

        ### Emphasis x&y using M on gradients: ###
        M_x = M * x
        M_y = M * y

        ### Get L1 on emphasized components (emphasized according to gradient strength): ###
        loss_value = self.L_p_Loss(M_x, M_y)  #weighted L1 on pixel values, weighted by gradient strength
        return loss_value

# import kornia
class Gradient_Sensitive_Loss_4(Gradient_Sensitive_Loss_1):
    def __init__(self, lambda_p=1):
        super(Gradient_Sensitive_Loss_4, self).__init__()

    def forward(self, x, y):
        x, y = x[0], y[0]
        number_of_input_channels = x.shape[1]
        # Make Tensors
        if self.number_of_kernel_channels != number_of_input_channels:
            self.gradient_x_kernel_numpy_multidim = self.gradient_x_kernel_numpy.repeat(
                repeats=number_of_input_channels, axis=0)
            self.gradient_y_kernel_numpy_multidim = self.gradient_y_kernel_numpy.repeat(
                repeats=number_of_input_channels, axis=0)
            self.gradient_x_kernel_Tensor = torch.Tensor(self.gradient_x_kernel_numpy_multidim)
            self.gradient_y_kernel_Tensor = torch.Tensor(self.gradient_y_kernel_numpy_multidim)
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.view((1, number_of_input_channels, 3, 3))
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.view((1, number_of_input_channels, 3, 3))
            self.number_of_kernel_channels = number_of_input_channels
            self.flag_already_set_kernel_size = True

        ### Make Sure Devices Are The Same: ###
        if self.gradient_x_kernel_Tensor.device != x.device:
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.to(x.device)
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.to(x.device)

        ### Get Gradient Strength Measure (M): ###
        G_x = torch.nn.functional.conv2d(x, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        G_y = torch.nn.functional.conv2d(x, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        M = (G - G.min()) / (G.max() - G.min())
        # Get Decomposition of x,y into x=M*x+(1-M)*x, y=M*y+(1-M)*y

        ### Emphasis x&y using M on gradients: ###
        M_x = M * x
        M_y = M * y
        M_x_complete = (1 - M) * x
        M_y_complete = (1 - M) * y

        ### Get Original Image Canny Edges: ###
        y_edges_magnitude, y_edges_binary = kornia.filters.canny(y)

        ### Get L1 on pixel values only where y has edges: ###
        loss_value = self.Masked_L_p_Loss(x, y, (y_edges_binary>0.5))  #assumes y_edges is binary image of edge/non-edge

        return loss_value


class Gradient_Sensitive_Loss_5(Gradient_Sensitive_Loss_1):
    def __init__(self, lambda_p=1):
        super(Gradient_Sensitive_Loss_5, self).__init__()

    def forward(self, x, y):
        x, y = x[0], y[0]
        number_of_input_channels = x.shape[1]
        # Make Tensors
        if self.number_of_kernel_channels != number_of_input_channels:
            self.gradient_x_kernel_numpy_multidim = self.gradient_x_kernel_numpy.repeat(
                repeats=number_of_input_channels, axis=0)
            self.gradient_y_kernel_numpy_multidim = self.gradient_y_kernel_numpy.repeat(
                repeats=number_of_input_channels, axis=0)
            self.gradient_x_kernel_Tensor = torch.Tensor(self.gradient_x_kernel_numpy_multidim)
            self.gradient_y_kernel_Tensor = torch.Tensor(self.gradient_y_kernel_numpy_multidim)
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.view((1, number_of_input_channels, 3, 3))
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.view((1, number_of_input_channels, 3, 3))
            self.number_of_kernel_channels = number_of_input_channels
            self.flag_already_set_kernel_size = True

        ### Make Sure Devices Are The Same: ###
        if self.gradient_x_kernel_Tensor.device != x.device:
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.to(x.device)
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.to(x.device)

        ### Get Gradient Strength Measure (M): ###
        G_x = torch.nn.functional.conv2d(x, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        G_y = torch.nn.functional.conv2d(x, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        M = (G - G.min()) / (G.max() - G.min())
        # Get Decomposition of x,y into x=M*x+(1-M)*x, y=M*y+(1-M)*y

        ### Emphasis x&y using M on gradients: ###
        M_x = M * x
        M_y = M * y
        M_x_complete = (1 - M) * x
        M_y_complete = (1 - M) * y

        ### Get Original Image Canny Edges: ###
        y_edges_magnitude, y_edges_binary = kornia.filters.canny(y)

        ### Get L1 on pixel values only where y has edges: ###
        loss_value = self.L_p_Loss(x*y_edges_magnitude, y*y_edges_magnitude)

        return loss_value

class Gradient_Sensitive_Loss_6(Gradient_Sensitive_Loss_1):
    def __init__(self, lambda_p=1):
        super(Gradient_Sensitive_Loss_6, self).__init__()

    def forward(self, x, y):
        x, y = x[0], y[0]
        number_of_input_channels = x.shape[1]
        # Make Tensors
        if self.number_of_kernel_channels != number_of_input_channels:
            self.gradient_x_kernel_numpy_multidim = self.gradient_x_kernel_numpy.repeat(
                repeats=number_of_input_channels, axis=0)
            self.gradient_y_kernel_numpy_multidim = self.gradient_y_kernel_numpy.repeat(
                repeats=number_of_input_channels, axis=0)
            self.gradient_x_kernel_Tensor = torch.Tensor(self.gradient_x_kernel_numpy_multidim)
            self.gradient_y_kernel_Tensor = torch.Tensor(self.gradient_y_kernel_numpy_multidim)
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.view((1, number_of_input_channels, 3, 3))
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.view((1, number_of_input_channels, 3, 3))
            self.number_of_kernel_channels = number_of_input_channels
            self.flag_already_set_kernel_size = True

        ### Make Sure Devices Are The Same: ###
        if self.gradient_x_kernel_Tensor.device != x.device:
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.to(x.device)
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.to(x.device)

        ### Get Gradient Strength Measure (M): ###
        x_Gx = torch.nn.functional.conv2d(x, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        x_Gy = torch.nn.functional.conv2d(x, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        y_Gx = torch.nn.functional.conv2d(y, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        y_Gy = torch.nn.functional.conv2d(y, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        G = torch.sqrt(torch.pow(x_Gx, 2) + torch.pow(x_Gy, 2))
        M = (G - G.min()) / (G.max() - G.min())
        # Get Decomposition of x,y into x=M*x+(1-M)*x, y=M*y+(1-M)*y

        ### Emphasis x&y using M on gradients: ###
        M_x = M * x
        M_y = M * y
        M_x_complete = (1 - M) * x
        M_y_complete = (1 - M) * y

        ### Get Original Image Canny Edges: ###
        y_edges_magnitude, y_edges_binary = kornia.filters.canny(y)

        ### Get L1 on pixel values only where y has edges: ###
        loss_value = self.L_p_Loss(x,y) + \
                     self.lambda_p * self.L_p_Loss(x_Gx * y_edges_magnitude, y_Gx * y_edges_magnitude) +\
                     self.lambda_p * self.L_p_Loss(x_Gy * y_edges_magnitude, y_Gy * y_edges_magnitude)

        return loss_value




class Masked_Gradient_Sensitive_Loss(nn.Module):
    def __init__(self, lambda_p=1):
        super(Masked_Gradient_Sensitive_Loss, self).__init__()
        self.gradient_x_kernel_numpy = np.array([[[1, 0, -1],
                                      [2, 0, -2],
                                      [1, 0, -1]]])
        self.gradient_y_kernel_numpy = np.array([[[1, 2, 1],
                                      [0, 0, 0],
                                      [-1, -2, -1]]])
        self.flag_already_set_kernel_size = False
        self.gradient_x_kernel_Tensor = torch.Tensor(self.gradient_x_kernel_numpy)
        self.gradient_y_kernel_Tensor = torch.Tensor(self.gradient_y_kernel_numpy)
        self.L_g_x_Loss = MaskedL1Loss()
        self.L_g_y_Loss = MaskedL1Loss()
        self.L_p_Loss = MaskedL1Loss()
        self.lambda_p = lambda_p
        self.number_of_kernel_channels = 0


    def to_device(self, device='cpu'):
        self.gradient_x_kernel_Tensor = torch.Tensor(self.gradient_x_kernel_numpy)
        self.gradient_y_kernel_Tensor = torch.Tensor(self.gradient_y_kernel_numpy)
        self.L_g_x_Loss = nn.L1Loss()
        self.L_g_y_Loss = nn.L1Loss()
        self.L_p_Loss = nn.L1Loss()

    def forward(self, x, y, valid_mask):
        number_of_input_channels = x.shape[1]
        #Make Tensors
        if self.number_of_kernel_channels != number_of_input_channels:
            self.gradient_x_kernel_numpy_multidim = self.gradient_x_kernel_numpy.repeat(repeats=number_of_input_channels,axis=0)
            self.gradient_y_kernel_numpy_multidim = self.gradient_y_kernel_numpy.repeat(repeats=number_of_input_channels,axis=0)
            self.gradient_x_kernel_Tensor = torch.Tensor(self.gradient_x_kernel_numpy_multidim)
            self.gradient_y_kernel_Tensor = torch.Tensor(self.gradient_y_kernel_numpy_multidim)
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.view((1,number_of_input_channels,3,3))
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.view((1,number_of_input_channels,3,3))
            self.number_of_kernel_channels = number_of_input_channels
            self.flag_already_set_kernel_size = True

        #Make Sure Devices Are The Same:
        if self.gradient_x_kernel_Tensor.device != x.device:
            self.gradient_x_kernel_Tensor = self.gradient_x_kernel_Tensor.to(x.device)
            self.gradient_y_kernel_Tensor = self.gradient_y_kernel_Tensor.to(x.device)

        G_x = torch.nn.functional.conv2d(x, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        G_y = torch.nn.functional.conv2d(x, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        M = (G-G.min()) / (G.max()-G.min())
        #Get Decomposition of x,y into x=M*x+(1-M)*x, y=M*y+(1-M)*y
        M_x = M * x/number_of_input_channels
        M_y = M * y/number_of_input_channels
        M_x_negative = (1-M) * x/number_of_input_channels
        M_y_negative = (1-M) * y/number_of_input_channels
        #Get Decomposition's Gradients:
        M_x_Gx = torch.nn.functional.conv2d(M_x, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        M_x_Gy = torch.nn.functional.conv2d(M_x, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        M_y_Gx = torch.nn.functional.conv2d(M_y, self.gradient_x_kernel_Tensor, bias=None, stride=1, padding=1)
        M_y_Gy = torch.nn.functional.conv2d(M_y, self.gradient_y_kernel_Tensor, bias=None, stride=1, padding=1)
        #Get Gradient Sensitive Loss:
        L_g = self.L_g_x_Loss(M_x_Gx,M_y_Gx,valid_mask) + self.L_g_y_Loss(M_x_Gy,M_y_Gy,valid_mask)
        L_p = self.L_p_Loss(M_x_negative,M_y_negative,valid_mask)
        return L_g + self.lambda_p * L_p



#(7). Gram Loss:
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.contiguous().view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class Gram_charbonnier_loss(nn.Module):
    def __init__(self):
        super(Gram_charbonnier_loss, self).__init__()

    def forward(self, input1, input2):
        G1 = gram_matrix(input1)
        G2 = gram_matrix(input2)
        char_loss = CharbonnierLoss()
        return char_loss(G1,G2)

class Gram_Loss(nn.Module):
    def __init__(self):
        super(Gram_Loss, self).__init__()

    def forward(self, input1, input2):
        G1 = gram_matrix(input1)
        G2 = gram_matrix(input2)
        return F.mse_loss(G1,G2)



class PSNR_Loss(nn.Module):
    def __init__(self):
        super(PSNR_Loss, self).__init__()
    def forward(self, input1, input2):
        return 10*torch.log10(1/nn.MSELoss()(input1,input2))

class STD_Loss(nn.Module):
    def __init__(self):
        super(STD_Loss, self).__init__()
    def forward(self, input1, input2):
        return torch.sqrt( ((input1-input2)**2).mean() )


class L2_Loss(nn.Module):
    def __init__(self):
        super(L2_Loss, self).__init__()
    def forward(self, input1, input2):
        return torch.sqrt( ((input1-input2)**2).mean() )



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

from RapidBase.Utils.Classical_DSP.FFT_utils import torch_fft2
class MaskedFFTLoss_1(nn.Module):
    def __init__(self, eps=1e-3, sigma=10):
        super(MaskedFFTLoss_1, self).__init__()
        self.eps = eps
        self.sigma = sigma
        self.gaussian_filter = None

    def forward(self, x, y, valid_mask=None):
        ### get gaussian filter to mask DC components in FFT: ###
        if self.gaussian_filter is None:
            H, W = x.shape[-2:]
            self.gaussian_filter = generate_gaussian_kernel(kernel_size=max(H, W), sigma=self.sigma)
            self.gaussian_filter = torch.tensor(self.gaussian_filter).to(x.device)
            self.gaussian_filter = crop_torch_batch(self.gaussian_filter, (H, W))
            self.gaussian_filter = scale_array_to_range(self.gaussian_filter, (0, 1))
            self.gaussian_filter = 1 - self.gaussian_filter

        ### Get valid mask: ###
        if valid_mask is None:
            valid_mask = torch.ones_like(x).bool()

        ### FFTshift to center DC component: ###
        H, W = x.shape[-2:]
        diff = (torch_fftshift(torch_fft2(x).abs(), -2) - torch_fftshift(torch_fft2(y).abs(), -2)) ** 2

        ### Multiple by gaussian filter to disregard lower frequency (~DC) components of the FFT and focus on high frequencies: ###
        diff = diff * self.gaussian_filter

        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff) + (self.eps * self.eps)))

        return loss

from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import crop_torch_batch, scale_array_to_range
from RapidBase.Utils.Classical_DSP.FFT_utils import torch_fftshift, fftshift_torch, fftshift_torch_specific_dim

class MaskedFFTLoss_2(nn.Module):
    def __init__(self, eps=1e-3, sigma=10):
        super(MaskedFFTLoss_2, self).__init__()
        self.eps = eps
        self.gaussian_filter = None
        self.sigma = sigma
    def forward(self, x, y, valid_mask=None):
        ### get gaussian filter to mask DC components in FFT: ###
        if self.gaussian_filter is None:
            H,W = x.shape[-2:]
            self.gaussian_filter = generate_gaussian_kernel(kernel_size=max(H,W), sigma=self.sigma)
            self.gaussian_filter = torch.tensor(self.gaussian_filter).to(x.device)
            self.gaussian_filter = crop_torch_batch(self.gaussian_filter, (H,W))
            self.gaussian_filter = scale_array_to_range(self.gaussian_filter, (0,1))
            self.gaussian_filter = 1 - self.gaussian_filter

        ### Get valid mask: ###
        if valid_mask is None:
            valid_mask = torch.ones_like(x).bool()

        ### Calculate Diff: ###
        diff = (torch_fft2(x) - torch_fft2(y)).abs() ** 2

        ### FFTshift to center DC component: ###
        H,W  = x.shape[-2:]
        diff = crop_torch_batch(fftshift_torch(diff, -2), (H,W))

        ### Multiple by gaussian filter to disregard lower frequency (~DC) components of the FFT and focus on high frequencies: ###
        diff = diff * self.gaussian_filter

        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff) + (self.eps*self.eps)))
        return loss

def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)

class Masked_CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(Masked_CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y, valid_mask):
        # diff = x - y
        # temp_upsampling_layer = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=None)
        # yy = temp_upsampling_layer(y.squeeze(0)).unsqueeze(0)
        # diff = x - yy
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff[valid_mask] ** 2) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class Masked_EdgeLoss(nn.Module):
    def __init__(self):
        super(Masked_EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(1, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = Masked_CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = torch.nn.functional.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return torch.nn.functional.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y, valid_mask):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y), valid_mask)
        return loss





################################################################################################################################################################################################################################################
### Semantic Segmentation (Per Pixel Classification) Losses: ###

def cross_entropy2d(input, target, weight=None, size_average=True):
    input_batch_size, input_number_of_channels, input_height, input_width = input.size()
    target_batch_size, target_height, target_width = target.size()

    # Handle inconsistent size between input and target
    if input_height > target_height and input_width > target_width:  # upsample labels
        target = target.unsqueeze(1)
        target = F.interpolate(target.float(), size=(input_height, input_width), mode="nearest").long()
        target = target.squeeze(1)
    elif input_height < target_height and input_width < target_width:  # upsample images
        input = F.interpolate(input, size=(target_height, target_width), mode="bilinear", align_corners=True)
    elif input_height != target_height and input_width != target_width:
        raise Exception("Only support upsampling")

    #Convert from [N,C,H,W]-->[N,H,W,C] in what i would say is an inefficient way... then collapse input to be only on the channels infex
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, input_number_of_channels) #make sure that this is not a bottle neck in performance
    target = target.view(-1)
    loss = F.cross_entropy(input, target, weight=weight, size_average=size_average, ignore_index=250) #ignore_index=250??? background???
    return loss



def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple): # when evaluation
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to('cuda' if input.is_cuda else 'cpu')

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(input=inp, target=target, weight=weight, size_average=size_average)

    return loss


def bootstrapped_cross_entropy2d(input,
                                  target,
                                  K,
                                  weight=None,
                                  size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input,
                                   target,
                                   K,
                                   weight=None,
                                   size_average=True):

        input_batch_size, input_number_of_channels, input_height, input_width = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, input_number_of_channels)
        target = target.view(-1)
        loss = F.cross_entropy(input,
                               target,
                               weight=weight,
                               reduce=False,
                               size_average=False,
                               ignore_index=250)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
################################################################################################################################################################################################################################################





#
# class Bilateral_Loss(nn.Module):  #logRMSE loss....another metric used which gives more similar weights to far and small depth values.
#     def __init__(self):
#         super(Bilateral_Loss, self).__init__()
#
#         ### Padding Layers: ###
#         number_of_neighbors = 9
#         self.left_top_pad = nn.ZeroPad2d((0, number_of_neighbors//2*2, 0, 2))  # the padding notation is (left,right,top,bottom)  (assuming we diffuse an affinity of 3x3 neighborhood)
#         self.center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
#         self.right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
#         self.left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
#         self.right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
#         self.left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
#         self.center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
#         self.right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))
#         self.center_center_pad = nn.ZeroPad2d((number_of_neighbors//2, number_of_neighbors//2, number_of_neighbors//2, number_of_neighbors//2))
#
#     def forward(self, input_tensor, GT_tensor, input_confidence, input_image):
#         B,C,H,W = input_tensor.shape
#         number_of_elements = H*W*C
#         # loss_direct = input_confidence * (input_tensor - GT_tensor)**2
#         loss_direct = nn.MSELoss()(torch.sqrt(input_confidence) * (input_tensor-GT_tensor).abs())
#
#         number_of_neighbors = 9
#         input_tensor_padded = self.center_center_pad(input_tensor)
#         GT_tensor_padded = self.center_center_pad(GT_tensor)
#         input_image_padded = self.center_center_pad(input_image)
#         pseudo_start_index = number_of_neighbors//2
#         for i in arange(-number_of_neighbors//2+1, number_of_neighbors//2+1):
#             for j in arange(-number_of_neighbors//2+1, number_of_neighbors//2+1):
#                 ### Get Current Weight: ###
#                 grid_term = (i**2+j**2)/(2*self.sigma_grid**2)
#                 bilateral_term = (input_image_padded[:,:,pseudo_start_index-i:pseudo_start_index-i+H, pseudo_start_index-j:pseudo_start_index-j+W]) - input_image_padded[:,:,:,:])**2/(2*self.sigma_bilateral**2)
#                 current_weight = torch.exp(-grid_term - bilateral_term)
#
#                 ### Get bilateral Loss: ###
#                 bilateral_loss = current_weight * (input_tensor_padded[:,:,pseudo_start_index-i:pseudo_start_index-i+H, pseudo_start_index-j:pseudo_start_index-j+W] - GT_tensor_padded[:,:,1:-1,1:-1])**2
#
#                 ### Get Total Loss: ###
#                 total_loss += bilateral_loss
#
#         total_loss += loss_direct
#         return total_loss




class WeightedL1(nn.Module):
    def __init__(self):
        super(WeightedL1, self).__init__()
    def forward(self, pred, target, weight):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        err = (pred-target).abs()*weight
        weighted_L1 = (err).mean()   # *200

        return weighted_L1


class WeightedL2(nn.Module):
    def __init__(self):
        super(WeightedL2, self).__init__()
    def forward(self, pred, target, weight):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        # err = (pred-target).abs()*torch.sqrt(weight)
        err = (pred-target).abs()*weight
        weighted_L1 = (err**2).mean()   # *200
        return weighted_L1






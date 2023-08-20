import torch.nn as nn
from easydict import EasyDict

from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_get_valid_center
from RapidBase.Utils.MISCELENEOUS import update_dict
import torch
from RapidBase.TrainingCore.losses_basic import *
import RapidBase.TrainingCore.losses_basic
import RapidBase.TrainingCore as TrainingCore
from RapidBase.Utils.IO.Imshow_and_Plots import imshow_torch, imshow_torch_multiple
import RapidBase.TrainingCore as TrainingCore

############################################################################################################################################################################################
class Loss_Base(nn.Module):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Base, self).__init__()
        ### Default Parameters: ###
        # self.L1_loss_function = torch.nn.L1Loss()
        self.L1_loss_function = TrainingCore.losses.MaskedL1Loss()
        self.L2_loss_function = TrainingCore.losses_basic.MaskedMSELoss()
        # self.SSIM_loss_function = TrainingCore.losses_basic.SSIM()  #TODO: fix this
        self.contextual_loss_function = TrainingCore.losses_basic.Contextual_Loss()
        self.gradient_sensitive_loss_function = TrainingCore.losses_basic.Gradient_Sensitive_Loss_2(lambda_p=0)  #lambda_p = actual values (regular loss) as opposed to gadient loss (which has no weight)
        self.masked_gradient_sensitive_loss_function = TrainingCore.losses_basic.Masked_Gradient_Sensitive_Loss(lambda_p=0)  #lambda_p = actual values (regular loss) as opposed to gadient loss (which has no weight)
        self.gram_loss_function = TrainingCore.losses_basic.Gram_Loss()
        self.gram_charbon_loss_function = TrainingCore.losses_basic.Gram_charbonnier_loss()
        self.L1_TargetValue_loss_function = TrainingCore.losses.MaskedL1Loss_TargetValue()
        self.L1_TargetMean_loss_function = TrainingCore.losses.MaskedL1Loss_TargetMean()
        self.optical_flow_unsupervised_loss = Unsupervised_Loss_Layer()
        self.optical_flow_unsupervised_OnlyForward_loss = Unsupervised_OnlyForward_Loss_Layer()
        self.FFT_loss_function_1 = MaskedFFTLoss_1()
        self.FFT_loss_function_2 = MaskedFFTLoss_2()
        self.Charbonnier_loss_function = TrainingCore.losses_basic.Masked_CharbonnierLoss()
        self.Charbonnier_loss_function2 = TrainingCore.losses_basic.CharbonnierLoss()
        self.VRT_contextual_loss_function = TrainingCore.losses_basic.VRT_ContextualLoss(use_vgg=True, vgg_layer='relu5_4', device=7) # fix to general device
        # self.L1_loss_function = self.masked_gradient_sensitive_loss_function

    def get_valid_mask(self, inputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Get Current Scale Outputs-Targets: ###
        if len(id.center_frame_noisy.shape)==4:
            B, C, H, W = id.center_frame_noisy.shape
        elif len(id.center_frame_noisy.shape)==5:
            B, T, C, H, W = id.center_frame_noisy.shape
        # center_frame = id.center_frame_noisy.detach().clone()

        ### Crop Results To Account For The Biggest Obvious Occlusions (left and right sides of the image which don't have matches): ###
        valid_mask = torch.ones((B,C,H,W)).to(id.center_frame_noisy.device)

        ### Remove Frame To Not Include Invalid Border Regions: ###
        valid_mask = valid_mask * torch_get_valid_center(valid_mask, Train_dict.non_valid_border_size).byte()

        return valid_mask.byte()

    def get_valid_mask_2(self, inputs_dict, Train_dict, clean_image_estimate):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Get Current Scale Outputs-Targets: ###
        if len(clean_image_estimate.shape) == 4:
            ### Get Shape: ###
            B, C, H, W = clean_image_estimate.shape
            ### Crop Results To Account For The Biggest Obvious Occlusions (Outer Frame): ###
            valid_mask = torch.ones((B, C, H, W)).to(clean_image_estimate.device)
            ### Remove Frame To Not Include Invalid Border Regions: ###
            valid_mask = valid_mask * torch_get_valid_center(valid_mask, Train_dict.non_valid_border_size)
            valid_mask = valid_mask[:, 0:clean_image_estimate.shape[1], :, :]  # in case there is a RGB conversion somewhere in the beginning
            if clean_image_estimate.shape[1] == 3 and valid_mask.shape[1] == 1:
                valid_mask = BW2RGB(valid_mask)
        elif len(clean_image_estimate.shape) == 5:
            B, T, C, H, W = clean_image_estimate.shape
            ### Crop Results To Account For The Biggest Obvious Occlusions (Outer Frame): ###
            valid_mask = torch.ones((B, T, C, H, W)).to(clean_image_estimate.device)
            ### Remove Frame To Not Include Invalid Border Regions: ###
            valid_mask = valid_mask * torch_get_valid_center(valid_mask, Train_dict.non_valid_border_size).byte()
            valid_mask = valid_mask[:, :, 0:clean_image_estimate.shape[2], :, :]  # in case there is a RGB conversion somewhere in the beginning
            if clean_image_estimate.shape[2] == 3 and valid_mask.shape[2] == 1:
                valid_mask = BW2RGB(valid_mask)

        return valid_mask.bool()


class Loss_Recursive_Simple(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Recursive_Simple, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def assign_things_to_keep_track_of_over_time(self, id, od, td, total_loss, losses_dict):
        ### Keep Track Of Things: ###
        if td.number_of_time_steps_from_batch_start == 0:
            od.losses_over_time = []
            od.output_frames_over_time = []
            od.l1_losses_over_time = []

        clean_image_estimate = od.clean_frame_estimate  #TODO: switch to od.model_output instead of clean_frame_estimate!!!!!
        od.losses_over_time.append(total_loss.detach().cpu().numpy())
        od.l1_losses_over_time.append(losses_dict.L1_loss.detach().cpu().numpy())
        od.output_frames_over_time.append(clean_image_estimate[0:1].detach().cpu())
        return id, od, td

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict
        losses_dict = EasyDict()

        ### Adjust recursive loss parameters: ###
        clean_frame_estimate = od.clean_frame_for_loss   #TODO: get rid of this, there is no more clean_frame_for_loss

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask_2(inputs_dict, Train_dict, clean_frame_estimate)

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # # (1). Masked L1:
        losses_dict.L1_loss = self.L1_loss_function(id.gt_frame_for_loss, clean_frame_estimate, valid_mask)
        total_loss += 1 * td.lambda_time * losses_dict.L1_loss * 10
        # (2). Target Running-Mean Value:
        # total_loss += 1 * td.lambda_time * self.L1_TargetValue_loss_function(id.gt_frame_for_loss, od.clean_frame_for_loss, id.RA_loss_mask, valid_mask)
        # total_loss += 1 * td.lambda_time * self.L1_TargetMean_loss_function(id.gt_frame_for_loss, od.clean_frame_for_loss, id.RA_loss_mask, valid_mask)

        # (2). Add Constraint For Reset Gate To Be 1:
        # total_loss += 50 * td.lambda_time * self.L1_loss_function(torch.ones_like(od.Reset_Gates_Combine), od.Reset_Gates_Combine, valid_mask)
        # total_loss += 50 * td.lambda_time * self.L1_loss_function(id.reset_gates_classic, od.Reset_Gates_Combine, valid_mask)

        ### Accumulate Intermediate Results For Later Use: ###id.RA_loss_mask
        od.valid_mask = valid_mask
        od.total_loss = total_loss

        ### Assign Things I Want To Keep Track Of Over Time: ###
        id, od, td = self.assign_things_to_keep_track_of_over_time(id, od, td, total_loss, losses_dict)

        return total_loss, od, id


# import models.EGVSR.codes.utils.net_utils as net_utils
class Loss_Recursive_EGVSR(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Recursive_EGVSR, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def assign_things_to_keep_track_of_over_time(self, id, od, td, total_loss, losses_dict):
        ### Keep Track Of Things: ###
        if td.number_of_time_steps_from_batch_start == 0:
            od.losses_over_time = []
            od.output_frames_over_time = []
            od.l1_losses_over_time = []

        clean_image_estimate = od.clean_frame_estimate
        od.losses_over_time.append(total_loss.detach().cpu().numpy())
        od.l1_losses_over_time.append(losses_dict.L1_loss.detach().cpu().numpy())
        od.output_frames_over_time.append(clean_image_estimate[0:1].detach().cpu())
        return id, od, td

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict
        losses_dict = EasyDict()

        ### Adjust recursive loss parameters: ###
        clean_frame_estimate = od.clean_frame_for_loss

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask_2(inputs_dict, Train_dict, clean_frame_estimate)

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        losses_dict.L1_loss = self.L1_loss_function(id.gt_frame_for_loss, clean_frame_estimate, valid_mask)
        total_loss += 1 * td.lambda_time * losses_dict.L1_loss * 10
        # (2). Warp L1:
        lr_flow = od.flow_output
        lr_prev = id.lr_prev
        lr_curr = id.lr_curr
        lr_warp = net_utils.backward_warp(lr_prev, lr_flow)
        valid_mask_warp = self.get_valid_mask_2(inputs_dict, Train_dict, lr_prev)
        losses_dict.warp_loss = self.L1_loss_function(lr_curr, lr_warp, valid_mask_warp)
        total_loss += 1 * td.lambda_time * losses_dict.warp_loss * 1

        ### for next iteration replace lr_prev with lr_curr: ###
        #TODO: putting this here is an ed hok solution, make this correct
        id.lr_prev = id.lr_curr  #

        # (2). Target Running-Mean Value:
        # total_loss += 1 * td.lambda_time * self.L1_TargetValue_loss_function(id.gt_frame_for_loss, od.clean_frame_for_loss, id.RA_loss_mask, valid_mask)
        # total_loss += 1 * td.lambda_time * self.L1_TargetMean_loss_function(id.gt_frame_for_loss, od.clean_frame_for_loss, id.RA_loss_mask, valid_mask)

        # (2). Add Constraint For Reset Gate To Be 1:
        # total_loss += 50 * td.lambda_time * self.L1_loss_function(torch.ones_like(od.Reset_Gates_Combine), od.Reset_Gates_Combine, valid_mask)
        # total_loss += 50 * td.lambda_time * self.L1_loss_function(id.reset_gates_classic, od.Reset_Gates_Combine, valid_mask)

        ### Accumulate Intermediate Results For Later Use: ###id.RA_loss_mask
        od.valid_mask = valid_mask
        od.total_loss = total_loss

        ### Assign Things I Want To Keep Track Of Over Time: ###
        id, od, td = self.assign_things_to_keep_track_of_over_time(id, od, td, total_loss, losses_dict)

        return total_loss, od, id


class Loss_Simple(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Simple, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###
        clean_image_estimate = od.model_output
        gt_clean_image = id.center_frame_original
        # gt_clean_image = id.output_frames_original

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask_2(inputs_dict, Train_dict, clean_image_estimate)

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        total_loss += self.L1_loss_function(gt_clean_image, clean_image_estimate, valid_mask) * 1
        # total_loss += self.FFT_loss_function_2(gt_clean_image, clean_image_estimate, valid_mask) * 0.01

        # if total_loss > 1e-1:
        #     total_loss = total_loss * 0


        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss

        return total_loss, od, id


class Loss_Simple_VRT(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Simple_VRT, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###
        clean_image_estimate = od.model_output
        gt_clean_image = id.output_frames_original

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask_2(inputs_dict, Train_dict, clean_image_estimate)

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        # total_loss += self.L1_loss_function(gt_clean_image, clean_image_estimate, valid_mask) * 1
        # total_loss += self.FFT_loss_function_2(gt_clean_image, clean_image_estimate, valid_mask) * 0.01
        total_loss = self.Charbonnier_loss_function(clean_image_estimate, gt_clean_image, valid_mask) * 1
        # total_loss = self.Charbonnier_loss_function2(clean_image_estimate, gt_clean_image) * 1
        # if total_loss > 1e-1:
        #     total_loss = total_loss * 0

        # x,y = clean_image_estimate[0], gt_clean_image[0]
        # a = self.contextual_loss_function(clean_image_estimate[0], gt_clean_image[0])
        # b = criterion(x[:1].cpu(), y[:1].cpu())
        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss

        return total_loss, od, id


class contextual_loss_VRT(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(contextual_loss_VRT, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###
        clean_image_estimate = od.model_output
        gt_clean_image = id.output_frames_original


        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        # total_loss += self.L1_loss_function(gt_clean_image, clean_image_estimate, valid_mask) * 1
        # total_loss += self.FFT_loss_function_2(gt_clean_image, clean_image_estimate, valid_mask) * 0.01
        if not td.now_training: # Use normal loss for validation to avoid OOM
            total_loss = self.Charbonnier_loss_function2(clean_image_estimate, gt_clean_image) * 1

        else: # Use contextual loss while training
            total_loss = self.VRT_contextual_loss_function(clean_image_estimate[0], gt_clean_image[0]) * 1
        # if total_loss > 1e-1:
        #     total_loss = total_loss * 0

        # x,y = clean_image_estimate[0], gt_clean_image[0]
        # a = self.contextual_loss_function(clean_image_estimate[0], gt_clean_image[0])
        # b = criterion(x[:1].cpu(), y[:1].cpu())
        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.total_loss = total_loss

        return total_loss, od, id

# import torch
#
# import contextual_loss as cl
# #
# #
# # # input features
#
# x = torch.rand(6, 3, 192, 192)
# y = torch.rand(6, 3, 192, 192)
# N, C, H, W = x.size()
#
# # Tensor -> patches
# size = 16 # patch size
# stride = 16 # patch stride
# x_patches = x.unfold(2, size, stride).unfold(3, size, stride)
# x_patches = x_patches.reshape(6,3,-1,16,16).reshape(6,-1,16,16)
# y_patches = y.unfold(2, size, stride).unfold(3, size, stride)
# y_patches = y_patches.reshape(6,3,-1,16,16).reshape(6,-1,16,16)
#
#
# x_patches_vec = x_patches.view(N,432,-1)
# y_patches_vec = y_patches.view(N,432,-1)
# x_patches_s = torch.sum(x_patches_vec**2,dim=1)
# y_patches_s = torch.sum(y_patches_vec**2,dim=1)
#
# y_vec = y.view(N, C, -1)
# A = y_vec.transpose(1,2).repeat(1,1,144) @ x_patches.flatten(-2,-1)
# dist = y_patches_s - 2 * A + x_patches_s.transpose(0, 1)
# dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
# dist = dist.clamp(min=0.)
#
# N, C, H, W = x.size()
# x_vec = x.view(N, C, -1)
# y_vec = y.view(N, C, -1)
# x_s = torch.sum(x_vec ** 2, dim=1)
# y_s = torch.sum(y_vec ** 2, dim=1)
#
#
#
# img1 = torch.rand(1, 3, 196, 196)
# img2 = torch.rand(1, 3, 96, 96)
# #
# # img1 = torch.rand(1, 3, 51, 196).to(15)
# # img2 = torch.rand(1, 3,196, 196).to(15)
# # # contextual loss
# criterion = cl.ContextualLoss()
# crit2 = TrainingCore.losses_basic.Contextual_Loss()
# loss = criterion(img1, img2)
# loss2 = crit2(img1, img2)
# # with torch.no_grad():
# #
# #
# torch.Size([1, 3, 512, 512])

class Loss_FFT_VRT(Loss_Base):
    def __init__(self, input_dict=None, fft_function_num=1, p=0, **kw):
        super(Loss_FFT_VRT, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__
        self.fft_function_num = fft_function_num
        self.p = p

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###
        clean_image_estimate = od.model_output
        gt_clean_image = id.output_frames_original

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask_2(inputs_dict, Train_dict, clean_image_estimate)

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        total_loss += self.L1_loss_function(gt_clean_image, clean_image_estimate, valid_mask) * 1
        # total_loss = self.gradient_sensitive_loss_function(clean_image_estimate, gt_clean_image, valid_mask) * 1
        if self.fft_function_num == 1:
            loss_function = self.FFT_loss_function_1

        if self.fft_function_num == 2:
            loss_function = self.FFT_loss_function_2

        if self.fft_function_num ==3:
            loss_function = self.gram_charbon_loss_function

        total_loss += loss_function(gt_clean_image[0], clean_image_estimate[0]) * 100 * self.p

        # if total_loss > 1e-1:
        #     total_loss = total_loss * 0


        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss

        return total_loss, od, id

class Loss_grad_VRT(Loss_Base):
    def __init__(self, input_dict=None, grad_function_num=2, p=0, **kw):
        super(Loss_grad_VRT, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__
        self.grad_function_num = grad_function_num
        self.p = p

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###
        clean_image_estimate = od.model_output
        gt_clean_image = id.output_frames_original

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask_2(inputs_dict, Train_dict, clean_image_estimate)

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        # total_loss += self.L1_loss_function(gt_clean_image, clean_image_estimate, valid_mask) * 1
        # total_loss += self.FFT_loss_function_2(gt_clean_image, clean_image_estimate, valid_mask) * 0.01
        # total_loss = self.gradient_sensitive_loss_function(clean_image_estimate, gt_clean_image, valid_mask) * 1
        if self.grad_function_num == 2:
            loss_function = TrainingCore.losses_basic.Gradient_Sensitive_Loss_2(lambda_p=self.p)
        if self.grad_function_num == 3:
            loss_function = TrainingCore.losses_basic.Gradient_Sensitive_Loss_3(lambda_p=self.p)
        if self.grad_function_num == 4:
            loss_function = TrainingCore.losses_basic.Gradient_Sensitive_Loss_4(lambda_p=self.p)
        if self.grad_function_num == 5:
            loss_function = TrainingCore.losses_basic.Gradient_Sensitive_Loss_5(lambda_p=self.p)
        if self.grad_function_num == 6:
            loss_function = TrainingCore.losses_basic.Gradient_Sensitive_Loss_6(lambda_p=self.p)

        total_loss = loss_function(clean_image_estimate, gt_clean_image) * 1
        # if total_loss > 1e-1:
        #     total_loss = total_loss * 0


        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss

        return total_loss, od, id


class Loss_Simple_RAFT_OpticalFlow(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Simple_RAFT_OpticalFlow, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###
        optical_flow_estimate = od.model_output
        reference_frame_index = 0
        gt_optical_flow_delta_x = id.optical_flow_delta_x - id.optical_flow_delta_x[:,reference_frame_index:reference_frame_index+1]
        gt_optical_flow_delta_y = id.optical_flow_delta_y - id.optical_flow_delta_y[:,reference_frame_index:reference_frame_index+1]
        ### Get Rid Of The Reference Image Column: ###
        gt_optical_flow_delta_x = torch.cat((gt_optical_flow_delta_x[:, 0:reference_frame_index], gt_optical_flow_delta_x[:, reference_frame_index + 1:]), 1) #[B,T,H,W]
        gt_optical_flow_delta_y = torch.cat((gt_optical_flow_delta_y[:, 0:reference_frame_index], gt_optical_flow_delta_y[:, reference_frame_index + 1:]), 1) #[B,T,H,W]
        gt_optical_flow = torch.cat((gt_optical_flow_delta_x.unsqueeze(2), gt_optical_flow_delta_y.unsqueeze(2)),2)  #[B,T,2,H,W]
        ### in this case i only output 1 optical flow between 2 images so i only take the relevant optical flow: ###
        gt_optical_flow = gt_optical_flow[:,0,:,:,:]

        ### Get Valid Mask For Loss: ###
        valid_mask = torch.ones_like((gt_optical_flow)).bool()

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        total_loss += self.L1_loss_function(gt_optical_flow, optical_flow_estimate, valid_mask) * 1

        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss
        outputs_dict.gt_optical_flow = gt_optical_flow
        outputs_dict.gt_optical_flow_delta_x = gt_optical_flow
        outputs_dict.gt_optical_flow_delta_y = gt_optical_flow
        outputs_dict.optical_flow_estimate = optical_flow_estimate

        return total_loss, od, id

from RDND_proper.models.FastFlowNet.flow_vis import flow_to_color, flow_uv_to_colors
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import crop_torch_batch
class Loss_Uns_RAFT_OpticalFlow(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Uns_RAFT_OpticalFlow, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###
        reference_frame_index = 0
        gt_optical_flow_delta_x = id.optical_flow_delta_x - id.optical_flow_delta_x[:,reference_frame_index:reference_frame_index + 1]
        gt_optical_flow_delta_y = id.optical_flow_delta_y - id.optical_flow_delta_y[:,reference_frame_index:reference_frame_index + 1]
        ### Get Rid Of The Reference Image Column: ###
        gt_optical_flow_delta_x = torch.cat((gt_optical_flow_delta_x[:, 0:reference_frame_index],
                                             torch.zeros_like(gt_optical_flow_delta_x[:,0:1]).to(gt_optical_flow_delta_x.device),
                                             gt_optical_flow_delta_x[:, reference_frame_index + 1:]), 1)  # [B,T,H,W]
        gt_optical_flow_delta_y = torch.cat((gt_optical_flow_delta_y[:, 0:reference_frame_index],
                                             torch.zeros_like(gt_optical_flow_delta_y[:, 0:1]).to(gt_optical_flow_delta_y.device),
                                             gt_optical_flow_delta_y[:, reference_frame_index + 1:]), 1)  # [B,T,H,W]
        gt_optical_flow = torch.cat((gt_optical_flow_delta_x, gt_optical_flow_delta_y),2)  # [B,T,2,H,W]
        ### in this case i only output 1 optical flow between 2 images so i only take the relevant optical flow: ###
        gt_optical_flow = gt_optical_flow[:, 1, :, :, :]

        ### Unpack Network Outputs For Easy Handling: ###
        optical_flow_estimate = od.model_output * 20  #(*). Ed-Hok div factor used in the NN training
        optical_flow_estimate = F.interpolate(optical_flow_estimate, scale_factor=4)
        optical_flow_estimate_magnitude = torch.sqrt(optical_flow_estimate[:,0:1]**2 + optical_flow_estimate[:,1:2]**2)
        gt_optical_flow_magnitude = torch.sqrt(gt_optical_flow[:,0:1]**2 + gt_optical_flow[:,1:2]**2)
        optical_flow_estimate_translation = crop_torch_batch(optical_flow_estimate, optical_flow_estimate.shape[-1]-Train_dict.non_valid_border_size).mean(-1,True).mean(-2,True) * torch.ones_like(optical_flow_estimate).to(optical_flow_estimate.device)
        # imshow(flow_to_color(optical_flow_estimate[0].detach().permute([1, 2, 0]).cpu().numpy()))

        ### Get Valid Mask For Loss: ###
        # valid_mask = torch.ones_like((optical_flow_estimate[:,0:1,:,:])).bool()
        valid_mask = self.get_valid_mask_2(inputs_dict, Train_dict, optical_flow_estimate[:,0:1,:,:])

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Unsupervised Loss:
        total_loss += self.optical_flow_unsupervised_OnlyForward_loss.forward(id.left_image, id.right_image, optical_flow_estimate,
                                                                              valid_mask, interpolation_mode='bilinear') * 1
        # supervised_loss = torch.abs(optical_flow_estimate - gt_optical_flow)
        # supervised_loss_translation = torch.abs(optical_flow_estimate_translation - gt_optical_flow)

        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss
        outputs_dict.optical_flow_estimate = optical_flow_estimate
        outputs_dict.gt_optical_flow = gt_optical_flow

        return total_loss, od, id

class Loss_Simple_RAFT_Translation(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Simple_RAFT_Translation, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###
        translation_estimate = od.model_output
        reference_frame_index = 0
        gt_translation_x = id.shift_x_vec - id.shift_x_vec[:,reference_frame_index:reference_frame_index+1]
        gt_translation_y = id.shift_y_vec - id.shift_y_vec[:,reference_frame_index:reference_frame_index+1]
        ### Get Rid Of The Reference Image Column: ###
        gt_translation_x = torch.cat((gt_translation_x[:, 0:reference_frame_index], gt_translation_x[:, reference_frame_index + 1:]), -1)
        gt_translation_y = torch.cat((gt_translation_y[:, 0:reference_frame_index], gt_translation_y[:, reference_frame_index + 1:]), -1)
        gt_translation = torch.cat((gt_translation_x, gt_translation_y),-1)

        ### Get Valid Mask For Loss: ###
        valid_mask = torch.ones_like((translation_estimate)).bool()

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        total_loss += self.L1_loss_function(gt_translation, translation_estimate, valid_mask) * 1

        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss
        outputs_dict.gt_translation = gt_translation
        outputs_dict.translation_estimate = translation_estimate

        return total_loss, od, id



class Loss_Simple_RLSP(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Simple_RLSP, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###
        #(*). TODO: understand how to deal with deep supervision in a smart way - whether to create different loss objects for deep supervision models or something else
        clean_image_estimate = od.model_output
        gt_clean_image = id.output_frames_original[:,-1]
        # gt_clean_image = id.center_frame_original

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask_2(inputs_dict, Train_dict, clean_image_estimate)

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        total_loss += self.L1_loss_function(gt_clean_image, clean_image_estimate, valid_mask) * 100

        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss

        return total_loss, od, id


class Loss_Simple_Charbonnier(Loss_Simple):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Simple_Charbonnier, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__
        self.L1_loss_function = Masked_CharbonnierLoss(1e-3)


from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import RGB2BW, BW2RGB
class Loss_Simple_RGB2BW(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Simple_RGB2BW, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###
        # (*). TODO: understand how to deal with deep supervision in a smart way - whether to create different loss objects for deep supervision models or something else
        clean_image_estimate = od.model_output
        gt_clean_image = RGB2BW(id.center_frame_original)

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask(inputs_dict, Train_dict)
        valid_mask = valid_mask[:, 0:1, :, :]  # in case there is a RGB conversion somewhere in the beginning
        valid_mask = valid_mask.bool()

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        total_loss += self.L1_loss_function(gt_clean_image, clean_image_estimate, valid_mask)

        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss

        return total_loss, od, id


class Loss_Simple_MPRNet(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Simple_MPRNet, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

        ### L1 loss function: ###
        self.L1_loss_function = Masked_CharbonnierLoss(1e-3)
        # self.L1_loss_function = MaskedL1Loss()
        self.gradient_loss_function = Masked_EdgeLoss()

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask(inputs_dict, Train_dict)
        valid_mask = valid_mask.bool()

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        total_loss += 1 * self.L1_loss_function(id.gt_frame_for_loss, od.clean_frame_estimate_1, valid_mask) +\
                      0.05 * self.gradient_loss_function(id.gt_frame_for_loss, od.clean_frame_estimate_1, valid_mask) + 1e-8
        total_loss += 1 * self.L1_loss_function(id.gt_frame_for_loss, od.clean_frame_estimate_2, valid_mask) + \
                      0.05 * self.gradient_loss_function(id.gt_frame_for_loss, od.clean_frame_estimate_2, valid_mask) + 1e-8
        total_loss += 1 * self.L1_loss_function(id.gt_frame_for_loss, od.clean_frame_estimate_3, valid_mask) +\
                      0.05 * self.gradient_loss_function(id.gt_frame_for_loss, od.clean_frame_estimate_3, valid_mask) + 1e-8
        if torch.isnan(total_loss):
            total_loss = 0 * (od.clean_frame_estimate_1.mean())
        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss

        return total_loss, od, id


import torch.nn.functional as F
class Loss_Simple_CBDNet(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Simple_CBDNet, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

        ### L1 loss function: ### #TODO: CBDNet uses L2 norms everywhere
        self.L2_loss_function = MaskedL2LossConfidence()
        self.gradient_loss_function = Masked_EdgeLoss()

        ### Specific Parameters: ###
        self.flag_use_assymetric_loss = True

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def simple_gradient_loss(self, x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w
        return tvloss

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask(inputs_dict, Train_dict)
        valid_mask = valid_mask.bool()

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        if self.flag_use_assymetric_loss:
            assymetric_mask = torch.abs(0.3 - F.relu(id.noise_sigma_map - od.noise_estimate))
        else:
            assymetric_mask = torch.ones_like(valid_mask)
        total_loss += 1 * self.L2_loss_function(id.gt_frame_for_loss, od.clean_frame_estimate, valid_mask)
        total_loss += 0.5 * self.L2_loss_function(id.noise_sigma_map, od.noise_estimate, valid_mask, assymetric_mask)
        total_loss += 0.05 * self.simple_gradient_loss(od.noise_estimate)

        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss

        return total_loss, od, id


class Loss_Gradient(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_Gradient, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###
        clean_image_estimate = od.model_output
        gt_clean_image = id.center_frame_original

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask(inputs_dict, Train_dict)
        valid_mask = valid_mask.bool()

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        total_loss += self.masked_gradient_sensitive_loss_function(gt_clean_image, clean_image_estimate, valid_mask)

        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss

        return total_loss, od, id


class Loss_DVDNet(Loss_Base):
    def __init__(self, input_dict=None, **kw):
        super(Loss_DVDNet, self).__init__()
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)

        ### Lambdas: ###
        self.temporal_lambda = 1
        self.spatial_lambda = 0

        ### Final Dict: ###
        self.final_dict = self.__dict__

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Unpack Network Outputs For Easy Handling: ###80
        clean_image_estimate = od.temporal_part_output_frame
        clean_frames_estimate_spatial_part = od.spatial_part_output_frames
        gt_clean_image = id.center_frame_original
        gt_clean_frames = id.original_frames

        ### Get Valid Mask For Loss: ###
        valid_mask = self.get_valid_mask(inputs_dict, Train_dict)
        valid_mask = valid_mask.bool()

        ### Initialize Total Loss: ###
        total_loss = 0

        ### Accumulate Losses: ###
        # (1). Masked L1:
        total_loss += self.temporal_lambda * self.L1_loss_function(gt_clean_image, clean_image_estimate, valid_mask)
        for frame_counter in np.arange(gt_clean_frames.shape[1]):
            total_loss += self.spatial_lambda * self.L1_loss_function(clean_frames_estimate_spatial_part[frame_counter], gt_clean_frames[:,frame_counter,:,:,:], valid_mask)

        ### Accumulate Intermediate Results For Later Use: ###
        outputs_dict.valid_mask = valid_mask
        outputs_dict.total_loss = total_loss

        return total_loss, od, id



#######
#TODO: this is an example or when learning a model with learnable leaky relu, use this as a template if i get to that situation!!!
class Loss_ConfidenceRefactoring_full_model_LearnableLeaky(Loss_Base):
    def __init__(self, input_dict=None, model=None, lambda_leak=0.01, **kw):
        #TODO to move params to the constructor
        super(Loss_ConfidenceRefactoring_full_model_LearnableLeaky, self).__init__()
        ### Default Parameters: ###
        self.sigmoid_act = nn.Sigmoid()
        self.criterion = nn.BCEWithLogitsLoss()
        self.offset = 4
        # (1). Direct Pixels Loss Lambda:
        self.lambda_BCE = 1
        self.lambda_grad = 0
        self.lambda_leak = lambda_leak
        self.disp_thr = 1
        self.counter = 0
        ### Get Final Internal Dictionary: ###  #TODO: think how to make cleaner
        self.__dict__ = update_dict(self.__dict__, input_dict)
        ### Final Dict: ###
        self.final_dict = self.__dict__

        ### Get Wanted Network: ###
        self.MaskedL1LossConfidence = MaskedL1LossConfidence()
        self.MaskedConfidenceBCELoss = MaskedConfidenceBCELoss()
        self.MaskedGradLossConfidence = MaskedGradLossConfidence()
        self.model = model

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        total_loss = loss_l1 + self.lambda_grad * loss_grad + self.lambda_BCE * loss_bce +\
                     self.lambda_leak * self.model.leak_param.abs()

        return total_loss, outputs_dict


#################################################################################################################################################################################################################################################
### SSIM: ####
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



def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class SSIM_Scalar_Layer1(torch.nn.Module):
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



def SSIM_PerPixel(x, y):
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
################################################################################################################################################################################################################################################







################################################################################################################################################################################################################################################
### GAN Discriminator Losses: ###
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
        self.current_discriminator_real_images_validity_loss = 0;
        self.current_discriminator_fake_images_validity_loss = 0;
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
################################################################################################################################################################################################################################################







################################################################################################################################################################################################################################################
### Contextual Loss: ###


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
        score = torch.mean(CX_loss);
    return score


class Contextual_Loss(nn.Module):
    def __init__(self, device=torch.device('cpu'), flag_L2_or_dotP=1):
        super(Contextual_Loss, self).__init__()
        self.flag_L2_or_dotP = flag_L2_or_dotP;
    def forward(self, x, y):
        return CX_loss(x, y, deformation=False, dis=False, flag_L2_or_dotP=self.flag_L2_or_dotP)





def extract_patches_2D(input_tensor,patch_size):
    #A function to extract patches of certain size.
    patch_H, patch_W = min(input_tensor.size(2),patch_size[0]),min(input_tensor.size(3),patch_size[1])
    patches_fold_H = input_tensor.unfold(2, patch_H, patch_H)
    if(input_tensor.size(2) % patch_H != 0):
        patches_fold_H = torch.cat((patches_fold_H,input_tensor[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, patch_W)
    if(input_tensor.size(3) % patch_W != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    # patches = patches_fold_HW.permute(0,2,3,1,4,5).reshape(-1,input_tensor.size(1),patch_H,patch_W)
    patches_permuted = patches_fold_HW.permute(0,2,3,1,4,5)
    patches_permuted_reshaped = patches_permuted.reshape(input_tensor.size(0),-1,patch_H,patch_W)
    return patches_permuted_reshaped


# ### Use Example: ###
# ### Direct Pixels Contextual Loss: ###
# ContextualLoss_flag_L2_or_dotP = 1;  #1=L2, 2=dot product
# Contextual_Loss_Function = Contextual_Loss(ContextualLoss_flag_L2_or_dotP)
# direct_pixels_Contextual_patch_factor = 5;
# real_images = torch.cuda.FloatTensor(np.random.randn(1,3,100,100))
# fake_images = torch.cuda.FloatTensor(np.random.randn(1,3,100,100))
# real_patches = extract_patches_2D(real_images, [int(real_images.shape[2]/direct_pixels_Contextual_patch_factor),int(real_images.shape[3]/direct_pixels_Contextual_patch_factor)])
# fake_patches = extract_patches_2D(fake_images, [int(fake_images.shape[2]/direct_pixels_Contextual_patch_factor),int(fake_images.shape[3]/direct_pixels_Contextual_patch_factor)])
# current_generator_direct_pixels_contextual_loss = Contextual_Loss_Function(real_patches, fake_patches);
################################################################################################################################################################################################################################################






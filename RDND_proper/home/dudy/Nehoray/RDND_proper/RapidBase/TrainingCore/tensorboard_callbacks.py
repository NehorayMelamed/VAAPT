from easydict import EasyDict
from RapidBase.TrainingCore.losses import MaskedL1Loss

import torch

from RapidBase.import_all import *
from RapidBase.Utils.Metrics import *

def update_dict(main_dict, input_dict):
    default_dict = EasyDict(main_dict)
    if input_dict is None:
        input_dict = EasyDict()
    default_dict.update(input_dict)
    return default_dict


############################################################################################################################################################################################
### TensorBoard: ###
class TB_Base(object):
    #TODO: add possibility of internal tracking of AVERAGES (mostly for validation)
    def __init__(self, TB_writer):
        self.Masked_L1_Function = MaskedL1Loss()
        self.TensorBoard_Train_dict = EasyDict()
        self.TensorBoard_Val_dict = EasyDict()
        self.TB_writer = TB_writer

    def get_TensorBoard_Metrics_Train_Base(self, inputs_dict, outputs_dict, Metrics_dict=None):  # The TensorBoard_dict will be Network.TensorBoard_Train_dict when training and Network.TensorBoard_Val_dict when validation. the reason for this is lack of code duplication
        ### Unpack data from dataloader: ###
        left_disparity_GT = inputs_dict['left_disparity_GT']
        valid_mask = outputs_dict['valid_mask']

        ### Unpack Network Outputs For Easy Handling: ###
        #left_disparity_Est1 = outputs_dict.left_disparity_Est1
        #left_disparity_Est2 = outputs_dict.left_disparity_Est1
        #confidence_output = outputs_dict.confidence_output

        ### L1: ###
        #left_disparity_Est_tensorboard = left_disparity_Est1.detach().clone().clamp(0)
        #left_disparity_GT_tensorboard = left_disparity_GT.detach().clone().clamp(0)
        Metrics_dict['L1_Refactoring'] = outputs_dict.total_loss.item()
        return Metrics_dict


    def get_TensorBoard_Metrics_Train(self, inputs_dict, outputs_dict):
        self.TensorBoard_Train_dict = self.get_TensorBoard_Metrics_Train_Base(inputs_dict, outputs_dict, self.TensorBoard_Train_dict)
        return self.TensorBoard_Train_dict

    def get_TensorBoard_Metrics_Val(self, inputs_dict, outputs_dict):
        self.TensorBoard_Val_dict = self.get_TensorBoard_Metrics_Train_Base(inputs_dict, outputs_dict, self.TensorBoard_Val_dict)
        return self.TensorBoard_Val_dict


    ### Log To TensorBoard During Traing: ###
    def Log_TensorBoard_Train(self, Train_dict, **kwargs):
        ### Gradient Clipping: ###
        if Train_dict.optimization_dict.flag_use_gradient_clipping:
            self.TB_writer.add_scalar('Generator/Total_Grad_Norm', Train_dict.total_norm, Train_dict.global_step + Train_dict.Network_checkpoint_step)
            self.TB_writer.add_scalar('Generator/Gradient_Clip_Norm', Train_dict.gradient_clip_value, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        ### Learning Rate: ###
        self.TB_writer.add_scalar('Generator/Learning_Rate', Train_dict.Network_lr, Train_dict.global_step + Train_dict.Network_checkpoint_step)

        #### Losses: ####
        for key, value in self.TensorBoard_Train_dict.items():
            self.TB_writer.add_scalar('Generator/' + key, value, Train_dict.global_step + Train_dict.Network_checkpoint_step)

        return self.TB_writer

    ### Log To TensorBoard During Traing: ###
    def Log_TensorBoard_Val(self, Train_dict, **kwargs):
        #### Losses: ####
        for key, value in self.TensorBoard_Val_dict.items():
            self.TB_writer.add_scalar('Generator/Val_' + key, value, Train_dict.global_step + Train_dict.Network_checkpoint_step)

        return self.TB_writer

############################################################################################################################################################################################


class TB_Denoising(TB_Base):
    def __init__(self, TB_writer):
        self.Masked_L1_Function = MaskedL1Loss()
        self.TensorBoard_Train_dict = EasyDict()
        self.TensorBoard_Val_dict = EasyDict()
        self.TensorBoard_Val_Average_Original_Noisy_dict = AverageMeter_Dict()
        self.TensorBoard_Val_History_Original_Noisy_dict = KeepValuesHistory_Dict()
        self.TensorBoard_Val_Average_Original_Estimate_dict = AverageMeter_Dict()
        self.TensorBoard_Val_History_Original_Estimate_dict = KeepValuesHistory_Dict()
        self.TB_writer = TB_writer

    def get_TensorBoard_Metrics_Train_Base(self, inputs_dict, outputs_dict, Metrics_dict=None):  # The TensorBoard_dict will be Network.TensorBoard_Train_dict when training and Network.TensorBoard_Val_dict when validation. the reason for this is lack of code duplication
        ### Unpack data from dataloader: ###
        center_frame_original = torch_get_4D(inputs_dict['center_frame_original'].data)
        # center_frame_noisy = torch_get_4D(inputs_dict['center_frame_noisy_HR'].data)
        center_frame_noisy = torch_get_4D(inputs_dict['center_frame_original'].data)  # Change to this on SR-inference
        # valid_mask = outputs_dict['valid_mask']

        ### Unpack Network Outputs For Easy Handling: ###
        clean_frame_estimate = torch_get_4D(outputs_dict['model_output_for_tensorboard'].data)

        ### Get results as numpy array: ###
        center_frame_original_tensorboard = center_frame_original.detach().clone().clamp(0,1)
        center_frame_noisy_tensorboard = center_frame_noisy.detach().clone().clamp(0,1)
        clean_frame_estimate_tensorboard = clean_frame_estimate.detach().clone().clamp(0,1)

        ### Metrics: ###
        original_to_noisy_metrics_dict = get_metrics_image_pair_torch(center_frame_noisy_tensorboard, center_frame_original_tensorboard)
        original_to_estimate_metrics_dict = get_metrics_image_pair_torch(clean_frame_estimate_tensorboard, center_frame_original_tensorboard)

        ### Assign Results To TensorBoard Metrics Dict: ###
        Metrics_dict['original_to_noisy_metrics_dict'] = original_to_noisy_metrics_dict
        Metrics_dict['original_to_estimate_metrics_dict'] = original_to_estimate_metrics_dict
        Metrics_dict['total_loss'] = outputs_dict.total_loss.item()

        return Metrics_dict


    def get_TensorBoard_Metrics_Train(self, inputs_dict, outputs_dict):
        self.TensorBoard_Train_dict = self.get_TensorBoard_Metrics_Train_Base(inputs_dict, outputs_dict, self.TensorBoard_Train_dict)
        return self.TensorBoard_Train_dict

    def get_TensorBoard_Metrics_Val(self, inputs_dict, outputs_dict):
        ### Get Current Example Metrics: ###
        self.TensorBoard_Val_dict = self.get_TensorBoard_Metrics_Train_Base(inputs_dict, outputs_dict, self.TensorBoard_Val_dict)

        ### Get Original-Noisy Dicts: ###
        self.TensorBoard_Val_Average_Original_Noisy_dict.update_dict(self.TensorBoard_Val_dict['original_to_noisy_metrics_dict'])
        self.TensorBoard_Val_History_Original_Noisy_dict.update_dict(self.TensorBoard_Val_dict['original_to_noisy_metrics_dict'])

        ### Get Original-Estimate Dicts: ###
        self.TensorBoard_Val_Average_Original_Estimate_dict.update_dict(self.TensorBoard_Val_dict['original_to_estimate_metrics_dict'])
        self.TensorBoard_Val_History_Original_Estimate_dict.update_dict(self.TensorBoard_Val_dict['original_to_estimate_metrics_dict'])
        return self.TensorBoard_Val_dict


    ### Log To TensorBoard During Traing: ###
    def Log_TensorBoard_Train(self, Train_dict, **kwargs):
        ### Gradient Clipping: ###
        if Train_dict.optimization_dict.flag_use_gradient_clipping:
            self.TB_writer.add_scalar('Generator/Total_Grad_Norm', Train_dict.total_norm, Train_dict.global_step + Train_dict.Network_checkpoint_step)
            self.TB_writer.add_scalar('Generator/Gradient_Clip_Norm', Train_dict.gradient_clip_value, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        ### Learning Rate: ###
        self.TB_writer.add_scalar('Generator/Learning_Rate', Train_dict.Network_lr, Train_dict.global_step + Train_dict.Network_checkpoint_step)

        #### Losses and Other Dictionary Keys: ####
        total_steps_so_far = Train_dict.global_step + Train_dict.Network_checkpoint_step
        for key, value in self.TensorBoard_Train_dict.items():
            if 'dict' in key: #if it's an inner dictionary loop over it own (key,value) pairs
                for key_in, value_in in self.TensorBoard_Train_dict[key].items():
                    if self.TensorBoard_Train_dict[key][key_in] is not np.inf and self.TensorBoard_Train_dict[key][key_in] is not None:
                        # self.TB_writer.add_scalar('Generator/' + key_in, self.TensorBoard_Train_dict[key][key_in], total_steps_so_far)
                        try:
                            self.TB_writer.add_scalar('Generator/' + key_in,
                                                      self.TensorBoard_Train_dict[key][key_in],
                                                      total_steps_so_far)
                        except:
                            1
            else: #if it's a simple value just write it down
                try:
                    self.TB_writer.add_scalar('Generator/' + key,
                                              self.TensorBoard_Train_dict[key],
                                              total_steps_so_far)
                except:
                    1
        # Keys: SSIM, NMSE, MSE, SNR_linear, SNR_dB, PSNR, VOI, contrast_measure_delta, blur_measurement


        # #### Histograms: ####
        # if ('model' in kwargs):
        #     model = kwargs['model']
        #     bias = torch.zeros(0)
        #     # import torch.nn as nn
        #     # for sub_module in model.modules():
        #     #     if isinstance(sub_module, nn.Conv2d):
        #     #             sub_module.bias.data.zero_()
        #     if((Train_dict.global_step + Train_dict.Network_checkpoint_step) % 200 == 0):
        #         for tag, value in model.named_parameters():
        #             # if ("bias" in tag) and (value.requires_grad) and (hasattr(value, 'data')):
        #             if ("bias" in tag) and (hasattr(value, 'data')):
        #                 bias = torch.cat([bias.to(value.device), value])
        #         self.TB_writer.add_histogram('Generator/bias_hist', value.data.cpu().numpy(), Train_dict.global_step + Train_dict.Network_checkpoint_step)

        return self.TB_writer

    ### Log To TensorBoard During Traing: ###
    def Log_TensorBoard_Val(self, Train_dict, **kwargs):
        # #### Losses: ####
        # self.TB_writer.add_scalars('Generator/train_valid_total_loss',
        #                             {'train': self.TensorBoard_Train_dict.total_loss,
        #                              'valid': self.TensorBoard_Val_dict.total_loss},
        #                             Train_dict.global_step + Train_dict.Network_checkpoint_step)

        #### Losses and Other Dictionary Keys: ####
        total_steps_so_far = Train_dict.global_step + Train_dict.Network_checkpoint_step
        for key, value in self.TensorBoard_Val_Average_Original_Estimate_dict.items():
            if 'dict' in key:  # if it's an inner dictionary loop over it own (key,value) pairs
                for key_in, value_in in self.TensorBoard_Val_Average_Original_Estimate_dict[key].items():
                    if self.TensorBoard_Val_Average_Original_Estimate_dict[key][key_in] is not np.inf and self.TensorBoard_Val_Average_Original_Estimate_dict[key][key_in] is not None:
                        # self.TB_writer.add_scalar('Generator/' + key_in, self.TensorBoard_Train_dict[key][key_in], total_steps_so_far)
                        self.TB_writer.add_scalar('Generator/' + key_in,
                                                  self.TensorBoard_Val_Average_Original_Estimate_dict[key][key_in],
                                                  total_steps_so_far)
            else:  # if it's a simple value just write it down
                self.TB_writer.add_scalar('Generator/' + key,
                                          self.TensorBoard_Val_Average_Original_Estimate_dict[key],
                                          total_steps_so_far)

        return self.TB_writer


    def Log_TensorBoard_Val_Averages(self, Train_dict, **kwargs):
        # #### Losses: ####
        # self.TB_writer.add_scalars('Generator/train_valid_total_loss',
        #                             {'train': self.TensorBoard_Train_dict.total_loss,
        #                              'valid': self.TensorBoard_Val_dict.total_loss},
        #                             Train_dict.global_step + Train_dict.Network_checkpoint_step)

        #### Losses and Other Dictionary Keys: ####
        total_steps_so_far = Train_dict.global_step + Train_dict.Network_checkpoint_step
        #total_steps_so_far = Train_dict.global_step
        for key, value in self.TensorBoard_Val_Average_Original_Estimate_dict.items():
            if 'dict' in key:  # if it's an inner dictionary loop over it own (key,value) pairs
                for key_in, value_in in self.TensorBoard_Val_Average_Original_Estimate_dict[key].items():
                    if self.TensorBoard_Val_Average_Original_Estimate_dict[key][key_in] is not np.inf and \
                            self.TensorBoard_Val_Average_Original_Estimate_dict[key][key_in] is not None:
                        # self.TB_writer.add_scalar('Generator/' + key_in, self.TensorBoard_Train_dict[key][key_in], total_steps_so_far)
                        self.TB_writer.add_scalar('Generator/' + key_in,
                                                  self.TensorBoard_Val_Average_Original_Estimate_dict[key][key_in],
                                                  total_steps_so_far)
            else:  # if it's a simple value just write it down
                try:
                    self.TB_writer.add_scalar('Generator/' + key,
                                              self.TensorBoard_Val_Average_Original_Estimate_dict[key],
                                              total_steps_so_far)
                except:
                    1
        return self.TB_writer


    ### TODO: i think this is un-necessary, understand if that's the case and delete: ###
    def Log_TensorBoard_Val_mean(self, Train_dict, outputs_dict, **kwargs):
        #### Losses: ####
        # for key, value in self.TensorBoard_Val_dict.items():
        #     self.TB_writer.add_scalar('Generator/Val_' + key, value, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        total_mean_val_loss = (torch.FloatTensor(outputs_dict.mean_total_val_loss)).mean()
        self.TB_writer.add_scalars('Generator/train_valid_total_loss',
                                    {'train': self.TensorBoard_Train_dict.total_loss,
                                     'valid': total_mean_val_loss.item()},
                                    Train_dict.global_step + Train_dict.Network_checkpoint_step)

        return total_mean_val_loss, self.TB_writer


class TB_SuperResolution(TB_Denoising):
    #TODO: unify this TB with the denoise TB!!!!
    def __init__(self, TB_writer):
        super().__init__(TB_writer)

    def get_TensorBoard_Metrics_Train_Base(self, inputs_dict, outputs_dict, Metrics_dict=None):  # The TensorBoard_dict will be Network.TensorBoard_Train_dict when training and Network.TensorBoard_Val_dict when validation. the reason for this is lack of code duplication
        ### Unpack data from dataloader: ###
        center_frame_original = inputs_dict['center_frame_original']
        center_frame_noisy = inputs_dict['center_frame_noisy_HR']
        # valid_mask = outputs_dict['valid_mask']

        ### Unpack Network Outputs For Easy Handling: ###
        clean_frame_estimate = outputs_dict['model_output_for_tensorboard']

        ### Get results as numpy array: ###
        center_frame_original_tensorboard = torch_get_4D(center_frame_original.detach().clone().clamp(0,1))
        center_frame_noisy_tensorboard = torch_get_4D(center_frame_noisy.detach().clone().clamp(0,1))
        clean_frame_estimate_tensorboard = torch_get_4D(clean_frame_estimate.detach().clone().clamp(0,1))

        ### Metrics: ###
        original_to_noisy_metrics_dict = get_metrics_image_pair_torch(center_frame_noisy_tensorboard, center_frame_original_tensorboard)
        original_to_estimate_metrics_dict = get_metrics_image_pair_torch(clean_frame_estimate_tensorboard, center_frame_original_tensorboard)

        ### Assign Results To TensorBoard Metrics Dict: ###
        Metrics_dict['original_to_noisy_metrics_dict'] = original_to_noisy_metrics_dict
        Metrics_dict['original_to_estimate_metrics_dict'] = original_to_estimate_metrics_dict
        Metrics_dict['total_loss'] = outputs_dict.total_loss.item()

        return Metrics_dict

class TB_OpticalFlow(TB_Denoising):
    def __init__(self, TB_writer):
        super().__init__(TB_writer)

    def get_TensorBoard_Metrics_Train_Base(self, inputs_dict, outputs_dict, Metrics_dict=None):  # The TensorBoard_dict will be Network.TensorBoard_Train_dict when training and Network.TensorBoard_Val_dict when validation. the reason for this is lack of code duplication
        ### Unpack data from dataloader: ###
        gt_optical_flow = outputs_dict['gt_optical_flow']
        optical_flow_estimate = outputs_dict['optical_flow_estimate']

        ### Assign Results To TensorBoard Metrics Dict: ###
        Metrics_dict['total_loss'] = outputs_dict.total_loss.item()

        return Metrics_dict

class TB_OpticalFlow_Translation(TB_Denoising):
    def __init__(self, TB_writer):
        super().__init__(TB_writer)

    def get_TensorBoard_Metrics_Train_Base(self, inputs_dict, outputs_dict, Metrics_dict=None):  # The TensorBoard_dict will be Network.TensorBoard_Train_dict when training and Network.TensorBoard_Val_dict when validation. the reason for this is lack of code duplication
        ### Unpack data from dataloader: ###
        gt_translation = outputs_dict['gt_translation']
        translation_estimate = outputs_dict['translation_estimate']

        ### Assign Results To TensorBoard Metrics Dict: ###
        Metrics_dict['total_loss'] = outputs_dict.total_loss.item()

        return Metrics_dict

class TB_Denoising_Recursive(TB_Base):
    def __init__(self, TB_writer):
        self.Masked_L1_Function = MaskedL1Loss()
        self.TensorBoard_Train_dict = EasyDict()
        self.TensorBoard_Val_dict = EasyDict()
        self.TensorBoard_Val_Average_Original_Noisy_dict = AverageMeter_Dict()
        self.TensorBoard_Val_History_Original_Noisy_dict = KeepValuesHistory_Dict()
        self.TensorBoard_Val_Average_Original_Estimate_dict = AverageMeter_Dict()
        self.TensorBoard_Val_History_Original_Estimate_dict = KeepValuesHistory_Dict()
        self.TB_writer = TB_writer

    def get_TensorBoard_Metrics_Train_Base(self, inputs_dict, outputs_dict,
                                           Metrics_dict=None):  # The TensorBoard_dict will be Network.TensorBoard_Train_dict when training and Network.TensorBoard_Val_dict when validation. the reason for this is lack of code duplication
        ### Unpack data from dataloader: ###
        center_frame_original = inputs_dict['current_gt_clean_frame']
        center_frame_noisy = inputs_dict['current_noisy_frame']  #TODO: would probably need to change for recursive super-resolution
        valid_mask = outputs_dict['valid_mask']

        ### Unpack Network Outputs For Easy Handling: ###
        clean_frame_estimate = outputs_dict['model_output_for_tensorboard']

        ### Get results as numpy array: ###
        center_frame_original_tensorboard = center_frame_original.detach().clone().clamp(0, 1)
        center_frame_noisy_tensorboard = center_frame_noisy.detach().clone().clamp(0, 1)
        clean_frame_estimate_tensorboard = clean_frame_estimate.detach().clone().clamp(0, 1)

        ### Get Parameters Specific For Recursion: ###
        losses_over_time = outputs_dict.losses_over_time
        l1_losses_over_time = outputs_dict.l1_losses_over_time

        ### Metrics: ###
        original_to_noisy_metrics_dict = get_metrics_image_pair_torch(center_frame_noisy_tensorboard,
                                                                      center_frame_original_tensorboard)
        original_to_estimate_metrics_dict = get_metrics_image_pair_torch(clean_frame_estimate_tensorboard,
                                                                         center_frame_original_tensorboard)

        ### Assign Results To TensorBoard Metrics Dict: ###
        Metrics_dict['original_to_noisy_metrics_dict'] = original_to_noisy_metrics_dict
        Metrics_dict['original_to_estimate_metrics_dict'] = original_to_estimate_metrics_dict
        Metrics_dict['total_loss'] = outputs_dict.total_loss.item()
        Metrics_dict['L1_delta'] = l1_losses_over_time[0] - l1_losses_over_time[-1]
        print(Metrics_dict['L1_delta'])

        return Metrics_dict

    def get_TensorBoard_Metrics_Train(self, inputs_dict, outputs_dict):
        self.TensorBoard_Train_dict = self.get_TensorBoard_Metrics_Train_Base(inputs_dict, outputs_dict,
                                                                              self.TensorBoard_Train_dict)
        return self.TensorBoard_Train_dict

    def get_TensorBoard_Metrics_Val(self, inputs_dict, outputs_dict):
        ### Get Current Example Metrics: ###
        self.TensorBoard_Val_dict = self.get_TensorBoard_Metrics_Train_Base(inputs_dict, outputs_dict,
                                                                            self.TensorBoard_Val_dict)

        ### Get Original-Noisy Dicts: ###
        self.TensorBoard_Val_Average_Original_Noisy_dict.update_dict(self.TensorBoard_Val_dict['original_to_noisy_metrics_dict'])
        self.TensorBoard_Val_History_Original_Noisy_dict.update_dict(self.TensorBoard_Val_dict['original_to_noisy_metrics_dict'])

        ### Get Original-Estimate Dicts: ###
        self.TensorBoard_Val_Average_Original_Estimate_dict.update_dict(self.TensorBoard_Val_dict['original_to_estimate_metrics_dict'])
        self.TensorBoard_Val_History_Original_Estimate_dict.update_dict(self.TensorBoard_Val_dict['original_to_estimate_metrics_dict'])
        return self.TensorBoard_Val_dict

    ### Log To TensorBoard During Traing: ###
    def Log_TensorBoard_Train(self, Train_dict, **kwargs):
        ### Gradient Clipping: ###
        if Train_dict.optimization_dict.flag_use_gradient_clipping:
            self.TB_writer.add_scalar('Generator/Total_Grad_Norm', Train_dict.total_norm,
                                      Train_dict.global_step + Train_dict.Network_checkpoint_step)
            self.TB_writer.add_scalar('Generator/Gradient_Clip_Norm', Train_dict.gradient_clip_value,
                                      Train_dict.global_step + Train_dict.Network_checkpoint_step)
        ### Learning Rate: ###
        self.TB_writer.add_scalar('Generator/Learning_Rate', Train_dict.Network_lr,
                                  Train_dict.global_step + Train_dict.Network_checkpoint_step)

        #### Losses and Other Dictionary Keys: ####
        total_steps_so_far = Train_dict.global_step + Train_dict.Network_checkpoint_step
        for key, value in self.TensorBoard_Train_dict.items():
            if 'dict' in key:  # if it's an inner dictionary loop over it own (key,value) pairs
                for key_in, value_in in self.TensorBoard_Train_dict[key].items():
                    if self.TensorBoard_Train_dict[key][key_in] is not np.inf and self.TensorBoard_Train_dict[key][
                        key_in] is not None:
                        # self.TB_writer.add_scalar('Generator/' + key_in, self.TensorBoard_Train_dict[key][key_in], total_steps_so_far)
                        try:
                            self.TB_writer.add_scalar('Generator/' + key_in, self.TensorBoard_Train_dict[key][key_in],
                                                      total_steps_so_far)
                        except:
                            1
            else:  # if it's a simple value just write it down
                self.TB_writer.add_scalar('Generator/' + key, self.TensorBoard_Train_dict[key], total_steps_so_far)
        # Keys: SSIM, NMSE, MSE, SNR_linear, SNR_dB, PSNR, VOI, contrast_measure_delta, blur_measurement

        # #### Histograms: #### #TODO: create a seperate function
        # if ('model' in kwargs):
        #     model = kwargs['model']
        #     bias = torch.zeros(0)
        #     # import torch.nn as nn
        #     # for sub_module in model.modules():
        #     #     if isinstance(sub_module, nn.Conv2d):
        #     #             sub_module.bias.data.zero_()
        #     if((Train_dict.global_step + Train_dict.Network_checkpoint_step) % 200 == 0):
        #         for tag, value in model.named_parameters():
        #             # if ("bias" in tag) and (value.requires_grad) and (hasattr(value, 'data')):
        #             if ("bias" in tag) and (hasattr(value, 'data')):
        #                 bias = torch.cat([bias.to(value.device), value])
        #         self.TB_writer.add_histogram('Generator/bias_hist', value.data.cpu().numpy(), Train_dict.global_step + Train_dict.Network_checkpoint_step)

        return self.TB_writer

    ### Log To TensorBoard During Traing: ###
    def Log_TensorBoard_Val(self, Train_dict, **kwargs):
        #### Losses and Other Dictionary Keys: ####
        total_steps_so_far = Train_dict.global_step + Train_dict.Network_checkpoint_step
        for key, value in self.TensorBoard_Val_Average_Original_Estimate_dict.items():
            if 'dict' in key:  # if it's an inner dictionary loop over it own (key,value) pairs
                for key_in, value_in in self.TensorBoard_Val_Average_Original_Estimate_dict[key].items():
                    if self.TensorBoard_Val_Average_Original_Estimate_dict[key][key_in] is not np.inf and \
                            self.TensorBoard_Val_Average_Original_Estimate_dict[key][key_in] is not None:
                        # self.TB_writer.add_scalar('Generator/' + key_in, self.TensorBoard_Train_dict[key][key_in], total_steps_so_far)
                        self.TB_writer.add_scalar('Generator/' + key_in,
                                                  {'validation':
                                                       self.TensorBoard_Val_Average_Original_Estimate_dict[key][
                                                           key_in]},
                                                  total_steps_so_far)
            else:  # if it's a simple value just write it down
                self.TB_writer.add_scalar('Generator/' + key, self.TensorBoard_Val_Average_Original_Estimate_dict[key],
                                          total_steps_so_far)

        return self.TB_writer

    def Log_TensorBoard_Val_Averages(self, Train_dict, **kwargs):
        #### Losses and Other Dictionary Keys: ####
        total_steps_so_far = Train_dict.global_step + Train_dict.Network_checkpoint_step
        for key, value in self.TensorBoard_Val_Average_Original_Estimate_dict.items():
            if 'dict' in key:  # if it's an inner dictionary loop over it own (key,value) pairs
                for key_in, value_in in self.TensorBoard_Val_Average_Original_Estimate_dict[key].items():
                    if self.TensorBoard_Val_Average_Original_Estimate_dict[key][key_in] is not np.inf and \
                            self.TensorBoard_Val_Average_Original_Estimate_dict[key][key_in] is not None:
                        # self.TB_writer.add_scalar('Generator/' + key_in, self.TensorBoard_Train_dict[key][key_in], total_steps_so_far)
                        self.TB_writer.add_scalar('Generator/' + key_in,
                                                  {'validation':
                                                       self.TensorBoard_Val_Average_Original_Estimate_dict[key][
                                                           key_in]},
                                                  total_steps_so_far)
            else:  # if it's a simple value just write it down
                try:
                    self.TB_writer.add_scalar('Generator/' + key,
                                              self.TensorBoard_Val_Average_Original_Estimate_dict[key],
                                              total_steps_so_far)
                except:
                    1
        return self.TB_writer

    ### TODO: i think this is un-necessary, understand if that's the case and delete: ###
    def Log_TensorBoard_Val_mean(self, Train_dict, outputs_dict, **kwargs):
        #### Losses: ####
        # for key, value in self.TensorBoard_Val_dict.items():
        #     self.TB_writer.add_scalar('Generator/Val_' + key, value, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        total_mean_val_loss = (torch.FloatTensor(outputs_dict.mean_total_val_loss)).mean()
        self.TB_writer.add_scalars('Generator/train_valid_total_loss',
                                   {'train': self.TensorBoard_Train_dict.total_loss,
                                    'valid': total_mean_val_loss.item()},
                                   Train_dict.global_step + Train_dict.Network_checkpoint_step)

        return total_mean_val_loss, self.TB_writer


class TB_Confidence_Refactoring_LearnableLeaky(TB_Base):
    #TODO: add possibility of internal tracking of AVERAGES (mostly for validation)
    def __init__(self, TB_writer):
        self.Masked_L1_Function = MaskedL1Loss()
        self.TensorBoard_Train_dict = EasyDict()
        self.TensorBoard_Val_dict = EasyDict()
        self.TB_writer = TB_writer

    def get_TensorBoard_Metrics_Train_Base(self, inputs_dict, outputs_dict, Metrics_dict=None):  # The TensorBoard_dict will be Network.TensorBoard_Train_dict when training and Network.TensorBoard_Val_dict when validation. the reason for this is lack of code duplication
        ### Unpack data from dataloader: ###
        left_disparity_GT = inputs_dict['left_disparity_GT']
        valid_mask = outputs_dict['valid_mask']

        ### Unpack Network Outputs For Easy Handling: ###
        left_disparity_Est1 = outputs_dict.left_disparity_Est1
        left_disparity_Est2 = outputs_dict.left_disparity_Est1
        confidence_output = outputs_dict.confidence_output

        ### L1: ###
        left_disparity_Est_tensorboard = left_disparity_Est1.detach().clone().clamp(0)
        left_disparity_GT_tensorboard = left_disparity_GT.detach().clone().clamp(0)
        Metrics_dict['total_loss'] = outputs_dict.total_loss.item()
        Metrics_dict['l1_loss_confidence_out'] = outputs_dict.loss_l1_confidence_out.item()
        Metrics_dict['l1_loss_confidence_in'] = outputs_dict.loss_l1_confidence_in.item()
        Metrics_dict['bce_loss'] = outputs_dict.loss_bce.item()
        Metrics_dict['alpha_leak'] = outputs_dict.alpha_leak.item()
        Metrics_dict['leak_reg_term'] = outputs_dict.leak_reg_term.item()
        Metrics_dict['loss_l1_confidence_out_th_0p35'] = outputs_dict.loss_l1_confidence_out_th_0p35.item()
        Metrics_dict['loss_l1_confidence_out_th_0p55'] = outputs_dict.loss_l1_confidence_out_th_0p55.item()
        Metrics_dict['loss_l1_confidence_out_th_0p85'] = outputs_dict.loss_l1_confidence_out_th_0p85.item()
        Metrics_dict['fill_factor_conf_out_th_0p35'] = outputs_dict.fill_factor_conf_out_th_0p35.item()
        Metrics_dict['fill_factor_conf_out_th_0p55'] = outputs_dict.fill_factor_conf_out_th_0p55.item()
        Metrics_dict['fill_factor_conf_out_th_0p85'] = outputs_dict.fill_factor_conf_out_th_0p85.item()
        Metrics_dict['fill_factor_conf_in'] = outputs_dict.fill_factor_conf_in.item()

        return Metrics_dict


    def get_TensorBoard_Metrics_Train(self, inputs_dict, outputs_dict):
        self.TensorBoard_Train_dict = self.get_TensorBoard_Metrics_Train_Base(inputs_dict, outputs_dict, self.TensorBoard_Train_dict)
        return self.TensorBoard_Train_dict

    def get_TensorBoard_Metrics_Val(self, inputs_dict, outputs_dict):
        self.TensorBoard_Val_dict = self.get_TensorBoard_Metrics_Train_Base(inputs_dict, outputs_dict, self.TensorBoard_Val_dict)
        return self.TensorBoard_Val_dict


    ### Log To TensorBoard During Traing: ###
    def Log_TensorBoard_Train(self, Train_dict, **kwargs):
        ### Gradient Clipping: ###
        if Train_dict.optimization_dict.flag_use_gradient_clipping:
            self.TB_writer.add_scalar('Generator/Total_Grad_Norm', Train_dict.total_norm, Train_dict.global_step + Train_dict.Network_checkpoint_step)
            self.TB_writer.add_scalar('Generator/Gradient_Clip_Norm', Train_dict.gradient_clip_value, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        ### Learning Rate: ###
        self.TB_writer.add_scalar('Generator/Learning_Rate', Train_dict.Network_lr, Train_dict.global_step + Train_dict.Network_checkpoint_step)

        #### Losses: ####
        # for key, value in self.TensorBoard_Train_dict.items():
        self.TB_writer.add_scalar('Generator/total_loss', self.TensorBoard_Train_dict.total_loss, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        self.TB_writer.add_scalar('Generator/masked_L1_loss_confidence_out', self.TensorBoard_Train_dict.l1_loss_confidence_out,Train_dict.global_step + Train_dict.Network_checkpoint_step)
        self.TB_writer.add_scalar('Generator/bce_loss', self.TensorBoard_Train_dict.bce_loss, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        self.TB_writer.add_scalars('Generator/masked_L1_loss', {'confidence_in': self.TensorBoard_Train_dict.l1_loss_confidence_in, 'confidence_out': self.TensorBoard_Train_dict.l1_loss_confidence_out}, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        self.TB_writer.add_scalar('Generator/alpha_leak', self.TensorBoard_Train_dict.alpha_leak, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        self.TB_writer.add_scalar('Generator/leak_reg_term', self.TensorBoard_Train_dict.leak_reg_term, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        self.TB_writer.add_scalars('Generator/masked_L1_loss_th',
                                   {'confidence_in': self.TensorBoard_Train_dict.l1_loss_confidence_in,
                                    'confidence_out_th_0p35': self.TensorBoard_Train_dict.loss_l1_confidence_out_th_0p35,
                                    'confidence_out_th_0p55': self.TensorBoard_Train_dict.loss_l1_confidence_out_th_0p55,
                                    'confidence_out_th_0p85': self.TensorBoard_Train_dict.loss_l1_confidence_out_th_0p85},
                                   Train_dict.global_step + Train_dict.Network_checkpoint_step)
        self.TB_writer.add_scalars('Generator/fill_factor',
                                   {'confidence_in': self.TensorBoard_Train_dict.fill_factor_conf_in,
                                    'confidence_out_th_0p35': self.TensorBoard_Train_dict.fill_factor_conf_out_th_0p35,
                                    'confidence_out_th_0p55': self.TensorBoard_Train_dict.fill_factor_conf_out_th_0p55,
                                    'confidence_out_th_0p85': self.TensorBoard_Train_dict.fill_factor_conf_out_th_0p85},
                                   Train_dict.global_step + Train_dict.Network_checkpoint_step)

        # #### Histograms: ####
        # if ('model' in kwargs):
        #     model = kwargs['model']
        #     bias = torch.zeros(0)
        #     # import torch.nn as nn
        #     # for sub_module in model.modules():
        #     #     if isinstance(sub_module, nn.Conv2d):
        #     #             sub_module.bias.data.zero_()
        #     if((Train_dict.global_step + Train_dict.Network_checkpoint_step) % 200 == 0):
        #         for tag, value in model.named_parameters():
        #             # if ("bias" in tag) and (value.requires_grad) and (hasattr(value, 'data')):
        #             if ("bias" in tag) and (hasattr(value, 'data')):
        #                 bias = torch.cat([bias.to(value.device), value])
        #         self.TB_writer.add_histogram('Generator/bias_hist', value.data.cpu().numpy(), Train_dict.global_step + Train_dict.Network_checkpoint_step)

        return self.TB_writer

    ### Log To TensorBoard During Traing: ###
    def Log_TensorBoard_Val(self, Train_dict, **kwargs):
        #### Losses: ####
        # for key, value in self.TensorBoard_Val_dict.items():
        #     self.TB_writer.add_scalar('Generator/Val_' + key, value, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        self.TB_writer.add_scalars('Generator/train_valid_total_loss',
                                    {'train': self.TensorBoard_Train_dict.total_loss,
                                     'valid': self.TensorBoard_Val_dict.total_loss},
                                    Train_dict.global_step + Train_dict.Network_checkpoint_step)

        return self.TB_writer


    def Log_TensorBoard_Val_mean(self, Train_dict, outputs_dict, **kwargs):
        #### Losses: ####
        # for key, value in self.TensorBoard_Val_dict.items():
        #     self.TB_writer.add_scalar('Generator/Val_' + key, value, Train_dict.global_step + Train_dict.Network_checkpoint_step)
        total_mean_val_loss = (torch.FloatTensor(outputs_dict.mean_total_val_loss)).mean()
        self.TB_writer.add_scalars('Generator/train_valid_total_loss',
                                    {'train': self.TensorBoard_Train_dict.total_loss,
                                     'valid': total_mean_val_loss.item()},
                                    Train_dict.global_step + Train_dict.Network_checkpoint_step)

        return total_mean_val_loss, self.TB_writer




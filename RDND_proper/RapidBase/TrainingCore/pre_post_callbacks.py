import torch.nn as nn
import torch
from RapidBase.import_all import *

############################################################################################################################################################################################
class PreProcessing_Base(nn.Module):
    def __init__(self, **kw):
        super(PreProcessing_Base, self).__init__()

    def RGB2Y(self,in_rgb):
        out_y = 0.299 * in_rgb[:, 0, :, :, ] + 0.587 * in_rgb[:, 1, :, :, ] + 0.114 * in_rgb[:, 2, :, :, ]

        return out_y

    def forward(self, inputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Prepare Input: ###
        network_input = torch.cat([id.output_frames])

        return network_input, id


class PostProcessing_Base(nn.Module):
    def __init__(self, **kw):
        super(PostProcessing_Base, self).__init__()

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        od.model_output = od.model_output

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##
################################################################################################################################################



############################################################################################################################################################################################
class PreProcessing_Denoise_NonRecursive_Base(PreProcessing_Base):
    def __init__(self, **kw):
        super(PreProcessing_Denoise_NonRecursive_Base, self).__init__()
        self.flag_BW2RGB = False
        self.flag_RGB2BW = False
        self.flag_movie_inference = False

    def adjust_inputs_dict_for_loss(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        return id, td, model

    def get_normalization_constant(self, network_input, id):
        # normalization_constant = get_max_correct_form(network_input, 0) # TODO: this normalization constant changes from working point to working point. i should do this, or get this from dataset object?
        self.normalization_constant = (network_input.max().item())
        return self.normalization_constant

    def normalize_noise_input(self, network_input, id, td):
        #TODO: get rid of this!
        # td.sigma_to_model = np.float32(td.sigma_to_model / self.normalization_constant)
        return network_input, id, td

    def prepare_noise_to_model(self, network_input, id, td):
        #TODO: get rid of this!
        # id.sigma_to_model = td.sigma_to_model
        # id.sigma_to_model = id.sigma_to_model * torch.ones_like(network_input).to(network_input.device)
        # id.sigma_to_model = id.sigma_to_model[:, 0:1, :, :]
        return network_input, id, td

    def normalize_inputs(self, network_input, id, td):
        ### Normalize: ###
        normalization_constant = self.get_normalization_constant(network_input, id)
        network_input = network_input / normalization_constant
        id.output_frames_original = id.output_frames_original / normalization_constant
        id.output_frames_noisy = id.output_frames_noisy / normalization_constant
        id.center_frame_noisy = id.center_frame_noisy / normalization_constant
        # id.center_frame_pseudo_running_mean = id.center_frame_pseudo_running_mean / normalization_constant
        id.center_frame_original = id.center_frame_original / normalization_constant
        if 'center_frame_noisy_HR' in id.keys():
            id.output_frames_noisy_HR = id.output_frames_noisy_HR / self.normalization_constant
            id.center_frame_noisy_HR = id.center_frame_noisy_HR / self.normalization_constant
        if 'center_frame_actual_mean' in id.keys():
            id.center_frame_actual_mean = id.center_frame_actual_mean / self.normalization_constant
        if 'current_moving_average' in id.keys():
            id.current_moving_average = id.current_moving_average / self.normalization_constant
        ### Normalize Noise Map: ###
        network_input, id, td = self.normalize_noise_input(network_input, id, td)

        return network_input, id, td

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        if self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        return network_input, id, td, model

    def forward(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Adjust Preprocessing Parameters: ###
        network_input, id, td, model = self.prepare_inputs_for_net(id, td, model)
        id, td, model = self.adjust_inputs_dict_for_loss(id, td, model)

        return network_input, id, td, model


class PostProcessing_Denoise_NonRecursive_Base(PostProcessing_Base):
    def __init__(self, **kw):
        super(PostProcessing_Denoise_NonRecursive_Base, self).__init__()
        self.flag_BW2RGB = False
        self.flag_RGB2BW = False

    def adjust_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Flags: ###
        flag_clip_noisy_tensor_after_noising = Train_dict.flag_clip_noisy_tensor_after_noising
        
        ### Use Residual Learning If Wanted: ###
        id.gt_frame_for_loss = id.center_frame_original
        od.model_output = od.model_output

        if self.flag_RGB2BW:
            id.gt_frame_for_loss = RGB2BW_interleave_T(id.gt_frame_for_loss)

        ### Clip Network Output If Wanted: ###
        if flag_clip_noisy_tensor_after_noising:
            id.gt_frame_for_loss = id.gt_frame_for_loss.clamp(0, 1)
            od.model_output = od.model_output.clamp(0, 1)

        return id, od, td

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        # (*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)
        
        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model
################################################################################################################################################


############################################################################################################################################################################################
class PreProcessing_Denoise_Recursive_Base(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_Denoise_Recursive_Base, self).__init__()
        self.number_of_model_input_frames = 1

    def adjust_inner_states_for_new_iteration(self, id, td, model):
        ### Start / Initiailizations: ###
        if td.reset_hidden_states_flag == 0:
            # Zero/Initialize at start of batch: #
            model.previous_states = id.output_frames_noisy[:, 0:1, :,:]  # (*). first frame (TODO: should be center frame)

        ### In Between Iterations Keep Accumulating Graph: ###
        if td.reset_hidden_states_flag == 1:
            # DO NOTHING!!!! - Keep Accumulating Graph: ###
            1

        ### Between Backward Steps -> Detach Previous States: ###
        if td.reset_hidden_states_flag == 2:
            # Use the previous hidden states but detach for memory usage limitation purposes: #
            model.previous_states = model.previous_states.detach()
        return id, td, model

    def get_normalization_constant(self, network_input, id):
        # normalization_constant = get_max_correct_form(network_input, 0) # TODO: this normalization constant changes from working point to working point. i should do this, or get this from dataset object?
        normalization_constant = 255
        self.normalization_constant = normalization_constant
        return normalization_constant

    def normalize_noise_input(self, network_input, id, td):
        td.sigma_to_model = np.float32(td.sigma_to_model / self.normalization_constant)
        id.sigma_to_model = td.sigma_to_model
        id.sigma_to_model = id.sigma_to_model * torch.ones_like(network_input).to(network_input.device)
        id.sigma_to_model = id.sigma_to_model[:, 0:1, :, :]
        return network_input, id, td

    def normalize_inputs(self, network_input, id, td):
        ### Normalize: ###
        normalization_constant = self.get_normalization_constant(network_input)
        network_input = network_input / normalization_constant
        id.output_frames_original = id.output_frames_original / normalization_constant
        id.output_frames_noisy = id.output_frames_noisy / normalization_constant
        id.center_frame_noisy = id.center_frame_noisy / normalization_constant
        id.center_frame_pseudo_running_mean = id.center_frame_pseudo_running_mean / normalization_constant
        id.center_frame_actual_mean = id.center_frame_actual_mean / normalization_constant
        id.center_frame_original = id.center_frame_original / normalization_constant
        ### Normalize Noise Map: ###
        network_input, id, td = self.normalize_noise_input(network_input, id, td)
        return network_input, id, td

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        number_of_time_steps_from_batch_start = td.number_of_time_steps_from_batch_start
        sigma = td.sigma_to_model

        ### Get Input
        input_frames = id.output_frames_noisy[:, number_of_time_steps_from_batch_start:number_of_time_steps_from_batch_start + self.number_of_model_input_frames, :, :]  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(input_frames)
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(input_frames)
        else:
            network_input = input_frames

        ### Normalize Inputs: ###
        if td.number_of_time_steps_from_batch_start == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### Concat Previous State To Input Frames: ###
        final_input_frames = torch.cat([input_frames, model.previous_states], 1)  # concat last output frame as input
        network_input = final_input_frames
        return network_input, id, td, model

    def get_wanted_inputs_to_device(self, id, td):
        for k, v in id.items():
            id[k] = id[k].to(td.device)
        return id, td

    def forward(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Adjust Preprocessing Parameters: ###
        id, td = self.get_wanted_inputs_to_device(id, td)
        id, td, model = self.adjust_inner_states_for_new_iteration(id, td, model)
        network_input, id, td, model = self.prepare_inputs_for_net(id, td, model)

        return network_input, id, td, model


class PostProcessing_Denoise_Recursive_Base(nn.Module):
    def __init__(self, **kw):
        super(PostProcessing_Denoise_Recursive_Base, self).__init__()
        self.number_of_model_input_frames = 5

    def adjust_recursive_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        #(***) MOST IMPORTANTLY - RETURNS id.gt_frame_for_loss, and od.model_output!!!

        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Get Lambda_Time: ###
        if len(td.time_steps_weights) == Train_dict.number_of_time_steps_in_one_backward:
            lambda_time = td.time_steps_weights[td.current_time_step]
        else:
            lambda_time = td.time_steps_weights[td.number_of_time_steps_from_batch_start]
        td.lambda_time = lambda_time

        ### Modify Dict To Follow Wanted Stuff: ###
        #(*). Frame Index To Use/Predict:
        frame_index_to_use_in_loss = td.number_of_time_steps_from_batch_start + self.number_of_model_input_frames//2  # middle frame
        # frame_index_to_use_in_loss = self.number_of_model_input_frames - 1 # last frame
        #(*). Assign Wanted Frames For Loss:
        id.current_gt_clean_frame = id.output_frames_original[:, frame_index_to_use_in_loss:frame_index_to_use_in_loss + 1, :, :]
        id.current_noisy_frame = id.output_frames_noisy[:, frame_index_to_use_in_loss:frame_index_to_use_in_loss + 1, :, :]
        id.gt_noise_residual = id.current_noisy_frame - id.current_gt_clean_frame
        T = td.number_of_time_steps_from_batch_start + 1
        if hasattr(id, 'current_moving_average'):
            id.current_moving_average = (1 - 1 / T) * id.current_moving_average + 1 / T * id.current_noisy_frame
        else:
            id.current_moving_average = id.current_noisy_frame

        ### Take Care Of Running-Average Related Stuff: ###
        id.RA_loss_mask = id.gt_noise_residual.abs() / np.sqrt(td.number_of_time_steps_from_batch_start+1)

        ### Get GT For Loss According To What I'm Targeting (clean frame or RA frame): ###
        if Train_dict.flag_learn_clean_or_running_average == 'clean':
            id.gt_frame_for_loss = id.output_frames_original[:, frame_index_to_use_in_loss:frame_index_to_use_in_loss + 1, :, :]
            od.model_output = od.model_output[:, 0:1, :, :]
        else:
            id.gt_frame_for_loss = id.current_moving_average
            od.model_output = od.model_output[:, 0:1, :, :]

        ### Clip Network Output If Wanted: ###
        if Train_dict.flag_clip_noisy_tensor_after_noising:
            id.gt_frame_for_loss = id.gt_frame_for_loss.clamp(0,1)
            od.model_output = od.model_output.clamp(0,1)

        return id, od, td

    def adjust_internal_states_for_next_iteration(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        #(1). Input Previous Output Frame As Inputs For Next Iteration:
        model.previous_states[:, 0:1, :, :] = od.model_output[:, 0:1, :, :]

        #(2). Inputs Other Previous Outputs (besides predicted clean frame) As Inputs For Next Iteration:
        if model.previous_states.shape[1] > 1:
            model.previous_states[:, 1:, :, :] = od.model_output[:, 1:, :, :]

        return id, od, td, model

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        od.model_output = od.model_output

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_recursive_loss_parameters(id, od, td)

        ### Assign Internal States: ###
        id, od, td, model = self.adjust_internal_states_for_next_iteration(id, od, td, model)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##
################################################################################################################################################


################################################################################################################################################
class PreProcessing_MPRNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_MPRNet, self).__init__()
        self.flag_BW2RGB = True

class PostProcessing_MPRNet(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_MPRNet, self).__init__()

    def adjust_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Flags: ###
        flag_clip_noisy_tensor_after_noising = Train_dict.flag_clip_noisy_tensor_after_noising

        ### Assign Stuff: ###
        id.gt_frame_for_loss = id.center_frame_original
        od.model_output = od.model_output_1

        ### Clip Network Output If Wanted: ###
        if flag_clip_noisy_tensor_after_noising:
            id.gt_frame_for_loss = id.gt_frame_for_loss.clamp(0,1)
            od.model_output = od.model_output.clamp(0,1)

        return id, od, td

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        od.model_output_1 = od.model_output[0].mean(1, True) #original model outputs 3 channels, i need 1 output channel(!!!!)
        od.model_output_2 = od.model_output[1].mean(1, True)
        od.model_output_3 = od.model_output[2].mean(1, True)

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output_1

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##
################################################################################################################################################


################################################################################################################################################
class PreProcessing_MPRNet_CameraNoise(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_MPRNet_CameraNoise, self).__init__()
        self.flag_BW2RGB = True

    def get_normalization_constant(self, network_input, id):
        normalization_constant = get_max_correct_form(network_input, 0)
        return normalization_constant


class PostProcessing_MPRNet_CameraNoise(PostProcessing_MPRNet):
    def __init__(self, **kw):
        super(PostProcessing_MPRNet_CameraNoise, self).__init__()
################################################################################################################################################


################################################################################################################################################
class PreProcessing_FFDNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_FFDNet, self).__init__()
        self.flag_BW2RGB = False

    def get_normalization_constant(self, network_input, id):
        # normalization_constant = get_max_correct_form(network_input, 0)
        normalization_constant = 50
        self.normalization_constant = normalization_constant
        return normalization_constant

    def normalize_noise_input(self, network_input, id, td):
        td.sigma_to_model = td.sigma_to_model / 1
        id.sigma_to_model = td.sigma_to_model / 1
        return network_input, id, td

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(network_input)


        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        #### Model Accepts Frames+Sigma: ###1
        if len(network_input.shape) == 5:
            B,T,C,H,W = network_input.shape
            network_input = torch.reshape(network_input,(B,C*T,H,W))
        network_input = (network_input, td.sigma_to_model)
        return network_input, id, td, model

class PostProcessing_FFDNet(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_FFDNet, self).__init__()
################################################################################################################################################


################################################################################################################################################
class PreProcessing_MIRNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_MIRNet, self).__init__()
        self.flag_BW2RGB = True


class PostProcessing_MIRNet(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_MIRNet, self).__init__()
        self.flag_RGB2BW = True
################################################################################################################################################


################################################################################################################################################
class PreProcessing_FastDVDNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_FastDVDNet, self).__init__()
        self.flag_BW2RGB = True

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### Take Care Of Noise: ###
        network_input, id, td = self.prepare_noise_to_model(network_input, id, td)

        ### Concat Frames & Sigma: ###
        # network_input = torch.nn.UpsamplingBilinear2d(scale_factor=4)(network_input[:,:,0,:,:]).unsqueeze(2)
        # network_input = network_input.unsqueeze(2)  # TODO: delete, simply for testing flag_how_to_concat=C
        # network_input = (network_input, id.sigma_to_model)
        return network_input, id, td, model

class PreProcessing_FastDVDNet_KayaCamera(PreProcessing_FastDVDNet):
    def __init__(self, **kw):
        super(PreProcessing_FastDVDNet_KayaCamera, self).__init__()
        self.flag_BW2RGB = True

class PreProcessing_FastDVDNet_Deblur(PreProcessing_FastDVDNet):
    def __init__(self, **kw):
        super(PreProcessing_FastDVDNet_Deblur, self).__init__()
        self.flag_BW2RGB = False

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### Take Care Of Noise: ###
        network_input, id, td = self.prepare_noise_to_model(network_input, id, td)

        ### Concat Frames & Sigma: ###
        # network_input = network_input.unsqueeze(2)  #for when flag_how_to_concat='C'
        network_input = (network_input, id.sigma_to_model)
        return network_input, id, td, model

class PostProcessing_FastDVDNet(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_FastDVDNet, self).__init__()
        self.flag_RGB2BW = True

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model
################################################################################################################################################


################################################################################################################################################
class PreProcessing_BasicVSR_Simple(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_BasicVSR_Simple, self).__init__()
        self.flag_BW2RGB = True

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0 or self.flag_movie_inference:  #TODO: add possibility of an internal variable like self.normalization_constant and check if it's None
            network_input, id, td = self.normalize_inputs(network_input, id, td)
        
        ### Take Care Of Noise: ###
        network_input, id, td = self.prepare_noise_to_model(network_input, id, td)
        
        ### Concat Frames & Sigma: ###
        # network_input = torch.nn.UpsamplingBilinear2d(scale_factor=4)(network_input[:,:,0,:,:]).unsqueeze(2)
        # network_input = network_input.unsqueeze(2)  # TODO: delete, simply for testing flag_how_to_concat=C
        # network_input = (network_input, id.sigma_to_model)
        return network_input, id, td, model

class PostProcessing_BasicVSR_Simple(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_BasicVSR_Simple, self).__init__()
        self.flag_RGB2BW = True

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #TODO: clean this out!!! unify od.model_output and od.model_output, make more meaningful names like od.model_output -> od.clean_frame_for_callback1
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)
        
        ### Assign Network Output Tensor: ###
        B,T,H,W = od.model_output.shape
        od.model_output_for_callback = torch_get_5D(od.model_output, 'BTHW')
        od.center_clean_frame_estimate_for_callback = od.model_output[:,T//2:T//2+1,:,:]
        od.model_output_for_tensorboard = od.model_output[:,T//2:T//2+1,:,:]

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model
################################################################################################################################################


################################################################################################################################################
class PreProcessing_VRT_Denoise(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_VRT_Denoise, self).__init__()
        self.flag_BW2RGB = True
        self.flag_denoise = True

    def get_normalization_constant(self, network_input, id):
        # normalization_constant = get_max_correct_form(network_input, 0) # TODO: this normalization constant changes from working point to working point. i should do this, or get this from dataset object?

        self.normalization_constant = int(network_input.max().item())
        return self.normalization_constant

    # def normalize_inputs(self, network_input, id, td):
    #     ### Normalize: ###
    #     normalization_constant = self.get_normalization_constant(network_input)
    #     network_input = network_input / normalization_constant
    #     id.output_frames_original = id.output_frames_original / normalization_constant
    #     id.output_frames_noisy = id.output_frames_noisy / normalization_constant
    #     id.center_frame_noisy = id.center_frame_noisy / normalization_constant
    #     # id.center_frame_pseudo_running_mean = id.center_frame_pseudo_running_mean / normalization_constant
    #     id.center_frame_original = id.center_frame_original / normalization_constant
    #     if 'center_frame_noisy_HR' in id.keys():
    #         id.output_frames_noisy_HR = id.output_frames_noisy_HR / self.normalization_constant
    #         id.center_frame_noisy_HR = id.center_frame_noisy_HR / self.normalization_constant
    #     if 'center_frame_actual_mean' in id.keys():
    #         id.center_frame_actual_mean = id.center_frame_actual_mean / self.normalization_constant
    #     if 'current_moving_average' in id.keys():
    #         id.current_moving_average = id.current_moving_average / self.normalization_constant
    #     ### Normalize Noise Map: ###
    #     network_input, id, td = self.normalize_noise_input(network_input, id, td)
    #
    #     return network_input, id, td

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### Take Care Of Noise: ###
        network_input, id, td = self.prepare_noise_to_model(network_input, id, td)

        ### Concat Frames & Sigma: ###
        if self.flag_denoise:
            B, T, C, H, W = network_input.shape
            network_input = torch.cat((network_input, torch.zeros((B,T,1,H,W)).to(network_input.device)), 2)

        network_input = network_input, td  #  For special vrt forward
        return network_input, id, td, model

    def forward(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Adjust Preprocessing Parameters: ###
        network_input, id, td, model = self.prepare_inputs_for_net(id, td, model)
        id, td, model = self.adjust_inputs_dict_for_loss(id, td, model)

        return network_input, id, td, model

################################################################################################################################################
class PreProcessing_VRT_Denoise_X3_frames(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_VRT_Denoise_X3_frames, self).__init__()
        self.flag_BW2RGB = True
        self.flag_denoise = True

    def get_normalization_constant(self, network_input, id):
        # normalization_constant = get_max_correct_form(network_input, 0) # TODO: this normalization constant changes from working point to working point. i should do this, or get this from dataset object?

        # self.normalization_constant = int(network_input.max().item())
        self.normalization_constant = (network_input.max().item())
        return self.normalization_constant

    # def normalize_inputs(self, network_input, id, td):
    #     ### Normalize: ###
    #     normalization_constant = self.get_normalization_constant(network_input)
    #     network_input = network_input / normalization_constant
    #     id.output_frames_original = id.output_frames_original / normalization_constant
    #     id.output_frames_noisy = id.output_frames_noisy / normalization_constant
    #     id.center_frame_noisy = id.center_frame_noisy / normalization_constant
    #     # id.center_frame_pseudo_running_mean = id.center_frame_pseudo_running_mean / normalization_constant
    #     id.center_frame_original = id.center_frame_original / normalization_constant
    #     if 'center_frame_noisy_HR' in id.keys():
    #         id.output_frames_noisy_HR = id.output_frames_noisy_HR / self.normalization_constant
    #         id.center_frame_noisy_HR = id.center_frame_noisy_HR / self.normalization_constant
    #     if 'center_frame_actual_mean' in id.keys():
    #         id.center_frame_actual_mean = id.center_frame_actual_mean / self.normalization_constant
    #     if 'current_moving_average' in id.keys():
    #         id.current_moving_average = id.current_moving_average / self.normalization_constant
    #     ### Normalize Noise Map: ###
    #     network_input, id, td = self.normalize_noise_input(network_input, id, td)
    #
    #     return network_input, id, td

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### Take Care Of Noise: ###
        network_input, id, td = self.prepare_noise_to_model(network_input, id, td)

        B, T, C, H, W = network_input.shape
        network_input = network_input.reshape(B, T // 3, C * 3, H, W)
        ### Concat Frames & Sigma: ###
        if self.flag_denoise:
            B, T, C, H, W = network_input.shape
            network_input = torch.cat((network_input, torch.zeros((B,T,1,H,W)).to(network_input.device)), 2)

        network_input = network_input, td  #  For special vrt forward
        return network_input, id, td, model

    def forward(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Adjust Preprocessing Parameters: ###
        network_input, id, td, model = self.prepare_inputs_for_net(id, td, model)
        id, td, model = self.adjust_inputs_dict_for_loss(id, td, model)

        return network_input, id, td, model

class PostProcessing_VRT_Denoise_X3_frames(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_VRT_Denoise_X3_frames, self).__init__()
        self.flag_RGB2BW = True

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)

        ### Assign Network Output Tensor: ###
        (B,T,C,H,W), shape_len = get_full_shape_torch(od.model_output)
        od.model_output = od.model_output.reshape(B, T*3, C//3, H, W)
        (B,T,C,H,W), shape_len = get_full_shape_torch(od.model_output)
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output[:,T//2]
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model


class PostProcessing_VRT_Denoise(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_VRT_Denoise, self).__init__()
        self.flag_RGB2BW = True

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)

        ### Assign Network Output Tensor: ###
        (B,T,C,H,W), shape_len = get_full_shape_torch(od.model_output)
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output[:,T//2]
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model
################################################################################################################################################


################################################################################################################################################
class PreProcessing_NLEDNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_NLEDNet, self).__init__()
        self.flag_BW2RGB = True


class PostProcessing_NLEDNet(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_NLEDNet, self).__init__()
        self.flag_RGB2BW = True

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model
################################################################################################################################################

################################################################################################################################################
class PreProcessing_ADNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_ADNet, self).__init__()
        self.flag_BW2RGB = False


class PostProcessing_ADNet(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_ADNet, self).__init__()
        self.flag_RGB2BW = False
################################################################################################################################################

################################################################################################################################################
class PreProcessing_ADNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_ADNet, self).__init__()
        self.flag_BW2RGB = False


class PostProcessing_ADNet(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_ADNet, self).__init__()
        self.flag_RGB2BW = False
################################################################################################################################################


################################################################################################################################################
class PreProcessing_RIDNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_RIDNet, self).__init__()
        self.flag_BW2RGB = True

    def get_normalization_constant(self, network_input, id):
        normalization_constant = 1
        self.normalization_constant = 1
        return normalization_constant

class PostProcessing_RIDNet(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_RIDNet, self).__init__()
        self.flag_RGB2BW = True
################################################################################################################################################


################################################################################################################################################
class PreProcessing_CBDNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_CBDNet, self).__init__()
        self.flag_BW2RGB = False

class PostProcessing_CBDNet(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_CBDNet, self).__init__()

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        od.model_output = od.model_output[1].mean(1, True) #original model outputs 3 channels, i need 1 output channel(!!!!)
        od.noise_estimate = od.model_output[0].mean(1, True) * 4.744 #ed-hok scaling, from what i saw they also use some rescaling...need to understand this

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model
################################################################################################################################################



################################################################################################################################################
class PreProcessing_SADNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_SADNet, self).__init__()
        self.flag_BW2RGB = False

class PostProcessing_SADNet(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_SADNet, self).__init__()
        self.flag_RGB2BW = True
################################################################################################################################################


################################################################################################################################################
class PreProcessing_FFDNet_Recursive(PreProcessing_Denoise_Recursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_FFDNet_Recursive, self).__init__()

    def get_wanted_inputs_to_device(self, id, td):
        for k, v in id.items():
            id[k] = id[k].to(td.device)
        return id, td

    def get_normalization_constant(self, network_input, id):
        # normalization_constant = get_max_correct_form(network_input, 0) # TODO: this normalization constant changes from working point to working point. i should do this, or get this from dataset object?
        normalization_constant = 255
        self.normalization_constant = normalization_constant
        # normalization_constant = 255 / 10
        return normalization_constant

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        number_of_time_steps_from_batch_start = td.number_of_time_steps_from_batch_start
        sigma = td.sigma_to_model

        ### Get Input
        input_frames = id.output_frames_noisy[:, number_of_time_steps_from_batch_start:number_of_time_steps_from_batch_start + self.number_of_model_input_frames, :, :]  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(input_frames)
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(input_frames)
        else:
            network_input = input_frames

        ### Normalize Inputs: ###
        if td.number_of_time_steps_from_batch_start == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### Concat Previous State To Input Frames: ###
        final_input_frames = torch.cat([input_frames, model.previous_states], 1)  # concat last output frame as input
        network_input = final_input_frames
        return network_input, id, td, model

    def forward(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Adjust Preprocessing Parameters: ###
        id, td = self.get_wanted_inputs_to_device(id, td)
        id, td, model = self.adjust_inner_states_for_new_iteration(id, td, model)
        network_input, id, td, model = self.prepare_inputs_for_net(id, td, model)

        return network_input, id, td, model

class PostProcessing_FFDNet_Recursive(PostProcessing_Denoise_Recursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_FFDNet_Recursive, self).__init__()


class PreProcessing_FastDVDNet_Recursive(PreProcessing_FFDNet_Recursive, PreProcessing_FastDVDNet_KayaCamera):
    def __init__(self, **kw):
        super(PreProcessing_FastDVDNet_Recursive, self).__init__()

    def get_wanted_inputs_to_device(self, id, td):
        for k, v in id.items():
            id[k] = id[k].to(td.device)
        return id, td

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        number_of_time_steps_from_batch_start = td.number_of_time_steps_from_batch_start
        sigma = td.sigma_to_model
        input_frames = id.output_frames_noisy[:, number_of_time_steps_from_batch_start:number_of_time_steps_from_batch_start + self.number_of_model_input_frames, :, :]  # Get relevant frame
        input_frames = torch.cat([input_frames, model.previous_states.mean(1,True)], 1)  # concat last output frame as input

        ### Prepare Input: ###
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(input_frames)
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(input_frames)
        else:
            network_input = input_frames

        ### Normalize Inputs: ###
        if td.current_time_step == 0 and td.number_of_time_steps_from_batch_start == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### Finalize Input To Network: ###
        network_input = network_input

        return network_input, id, td, model

class PreProcessing_FFDNet_Recursive_2(PreProcessing_FFDNet_Recursive):
    #(*). this class is for situations where you transfer more then the previous clean frame to next iteration but also other outputs
    def __init__(self, **kw):
        super(PreProcessing_FFDNet_Recursive_2, self).__init__()

    def adjust_inner_states_for_new_iteration(self, id, td, model):
        if td.reset_hidden_states_flag == 0:
            # Initialize With Certain A Non-Trivial Tensor: #
            B, C, H, W = id.output_frames_noisy.shape
            model.previous_states = torch.zeros((B,5,H,W)).to(id.output_frames_noisy.device)
            model.previous_states[:, 0:1, :, :] = id.output_frames_noisy[:, 0:1, :, :]
            model.previous_states[:, 1:2, :, :] = id.output_frames_noisy[:, 0:1, :, :]
            model.previous_states[:, 2:3, :, :] = id.output_frames_noisy[:, 0:1, :, :]
            model.previous_states[:, 3:4, :, :] = id.output_frames_noisy[:, 0:1, :, :]
            model.previous_states[:, 4:5, :, :] = id.output_frames_noisy[:, 0:1, :, :]
            model.previous_states[:, 5:6, :, :] = id.output_frames_noisy[:, 0:1, :, :]

        ### Take care of internal states when something happens (backward step, start of batch etc'): ###
        if td.reset_hidden_states_flag == 1:
            # DO NOTHING!!!! - Keep Accumulating Graph: ###
            1

        if td.reset_hidden_states_flag == 2:
            # Use the previous hidden states but detach for memory usage limitation purposes: #
            model.previous_states = model.previous_states.detach()
        return id, td, model
################################################################################################################################################


################################################################################################################################################
class PreProcessing_ClassicIIR_AlphaBlend(PreProcessing_Base):
    def __init__(self, **kw):
        super(PreProcessing_ClassicIIR_AlphaBlend, self).__init__()

    def adjust_inner_states_for_new_iteration(self, id, td, model):
        #TODO: understand what ARE the internal memory states of the model and act accordingly
        ### Take care of internal states when something happens (backward step, start of batch etc'): ###
        if td.reset_hidden_states_flag == 0:
            # DO NOTHING!!!! - Keep Accumulating Graph: ###
            1
        if td.reset_hidden_states_flag == 1:
            # Zero/Initialize at start of batch: #
            1

        if td.reset_hidden_states_flag == 2:
            # Use the previous hidden states but detach for memory usage limitation purposes: #
            model.clean_frame_current = model.clean_frame_current.detach()
            model.clean_frame_previous = model.clean_frame_previous.detach()
            model.clean_frame_previous_warped = model.clean_frame_previous_warped.detach()
        return id, td, model

    def adjust_inputs_dict_for_loss(self, inputs_dict, Train_dict, model):
        #TODO: transfer to post-processing function like i did for the simple recursive case
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Recursion/Temporal Parameters: ###
        number_of_total_backward_steps_per_image = Train_dict.number_of_total_backward_steps_per_image  # number of backward steps before changing images batch
        number_of_time_steps = Train_dict.number_of_time_steps_in_one_backward  # number of time steps before doing a .backward()
        number_of_time_steps_from_batch_start = Train_dict.number_of_time_steps_from_batch_start

        ### Modify Dict To Follow Wanted Stuff: ### #TODO: pass to an internal function
        id.current_gt_clean_frame = id.output_frames_original[:, number_of_time_steps_from_batch_start:number_of_time_steps_from_batch_start + 1, :, :].to(Train_dict.device)
        id.current_noisy_frame = id.output_frames_noisy[:, number_of_time_steps_from_batch_start:number_of_time_steps_from_batch_start + 1, :, :].to(Train_dict.device)
        id.gt_noise_residual = id.current_gt_clean_frame - id.current_noisy_frame
        T = number_of_time_steps_from_batch_start + 1
        if hasattr(id, 'current_moving_average'):
            id.current_moving_average = (1 - 1 / T) * id.current_moving_average + 1 / T * id.current_noisy_frame
        else:
            id.current_moving_average = id.current_noisy_frame
        return id, td, model

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        number_of_time_steps_from_batch_start = td.number_of_time_steps_from_batch_start
        sigma = td.sigma_to_model

        ### Take care of initializations and such for first frame: ###
        #TODO: transfer to normalization function like i did in simple recursive case
        #(*). Normalization:
        if number_of_time_steps_from_batch_start == 0:
            id.output_frames_original = id.output_frames_original / td.max_value_possible
            id.output_frames_noisy = id.output_frames_noisy / td.max_value_possible
            id.current_gt_clean_frame = id.current_gt_clean_frame / td.max_value_possible
            id.current_moving_average = id.current_moving_average / td.max_value_possible
            id.output_frames_noisy_running_average = id.output_frames_noisy_running_average / td.max_value_possible

        ### Get Current Frames: ###
        id.original_frame_current = id.output_frames_original[:, number_of_time_steps_from_batch_start + 1:number_of_time_steps_from_batch_start + 2, :, :].data
        id.noisy_frame_current = id.output_frames_noisy[:, number_of_time_steps_from_batch_start + 1:number_of_time_steps_from_batch_start + 2, :, :].data
        #(3). Crop Frames (TODO: do this in the dataset object):
        id.noisy_frame_current_cropped = id.noisy_frame_current
        id.original_frame_current_cropped = id.original_frame_current

        ### To GPU: ###
        id = self.get_wanted_inputs_to_device(id, td.device)

        ### Network Input: ###
        network_input = id.noisy_frame_current_cropped

        ### Take Care Of Initialization At Start Of Movie: ###
        if number_of_time_steps_from_batch_start == 0:
            ### Initialize More Things For Internal Tracking: ###
            id.original_frame_previous = id.output_frames_original[:, 0:1, :, :].data
            id.noisy_frame_previous = id.output_frames_noisy[:, 0:1, :, :].data
            id.clean_frame_perfect_averaging = id.noisy_frame_previous
            id.clean_frame_simple_averaging = id.noisy_frame_previous

            ### Initialize Model Internal Variables: ###
            model = self.initialize_model_variables(model, id)

        return network_input, id, td, model

    def initialize_model_variables(self, model, id):
        # model.clean_frame_previous = id.noisy_frame_current_cropped  # initialize internal "cleaned" image to simply be first frame, which is noisy
        model.clean_frame_previous = id.noisy_frame_previous  # initialize internal "cleaned" image to simply be first frame, which is noisy
        model.pixel_counts_map_previous = torch.ones_like(id.noisy_frame_current_cropped)
        model.pixel_counts_map_previous_warped = torch.ones_like(id.noisy_frame_current_cropped)
        model.noise_map_previous = None
        return model

    def get_wanted_inputs_to_device(self, id, device):
        id.original_frame_current = id.original_frame_current.to(device)
        id.noisy_frame_current = id.noisy_frame_current.to(device)
        id.current_gt_clean_frame = id.current_gt_clean_frame.to(device)
        id.current_moving_average = id.current_moving_average.to(device)
        id.output_frames_noisy_running_average = id.output_frames_noisy_running_average.to(device)
        id.noisy_frame_current_cropped = id.noisy_frame_current_cropped.to(device)
        id.original_frame_current_cropped = id.original_frame_current_cropped.to(device)
        id.original_frame_current_cropped = id.original_frame_current_cropped.to(device)
        return id

    def forward(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Adjust Preprocessing Parameters: ###
        id, td, model = self.adjust_inner_states_for_new_iteration(id, td, model)
        id, td, model = self.adjust_inputs_dict_for_loss(id, td, model)
        network_input, id, td, model = self.prepare_inputs_for_net(id, td, model)

        return network_input, id, td, model


class PreProcessing_ClassicIIR(PreProcessing_ClassicIIR_AlphaBlend):
    def __init__(self, **kw):
        super(PreProcessing_ClassicIIR, self).__init__()

    def adjust_inner_states_for_new_iteration(self, id, td, model):
        #TODO: understand what ARE the internal memory states of the model and act accordingly
        if td.reset_hidden_states_flag == 0:
            # Zero/Initialize at start of batch: #
            #(*). TODO: The initialization is in the model script itself, maybe i should do it here?
            1

        ### Take care of internal states when something happens (backward step, start of batch etc'): ###
        if td.reset_hidden_states_flag == 1:
            # DO NOTHING!!!! - Keep Accumulating Graph: ###
            1

        ### In Between Loss Accumulation and Backward(), detach previous graph: ###
        if td.reset_hidden_states_flag == 2:
            # Use the previous hidden states but detach for memory usage limitation purposes: #
            model.clean_frame_current = model.clean_frame_current.detach()
            model.clean_frame_previous = model.clean_frame_previous.detach()
            model.clean_frame_previous_warped = model.clean_frame_previous_warped.detach()

            model.pixel_counts_map_current = model.pixel_counts_map_current.detach()
            model.pixel_counts_map_previous = model.pixel_counts_map_previous.detach()
            model.pixel_counts_map_previous_warped = model.pixel_counts_map_previous_warped.detach()

            model.reset_gates_classic = model.reset_gates_classic.detach()
            model.Reset_Gates_Combine = model.Reset_Gates_Combine.detach()

            model.noise_map_current = model.noise_map_current.detach()
            model.noise_map_previous = model.noise_map_previous.detach()

        return id, td, model


class PostProcessing_ClassicIIR(nn.Module):
    def __init__(self, **kw):
        super(PostProcessing_ClassicIIR, self).__init__()

    def adjust_target_output_for_model(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ####################################
        # (*). Besides incentivising optical_flow_estimate and occlusions_estimate outputs to be correct "in bulk"/"large sections", that still isn't everything.
        # for instance, let's say we are able to track the face of a person in general, but now it twitches/deforms the face in low resolution...we might have problems.
        # in general, there are, of course, problems when doing optical flow on small object / scales.

        # (*). moreover, we have the problem of "blinking". pixels can remain even non-moving but the overall intensity of them can change.
        # on of the best examples is blinking traffic lights.
        # for that reason, most probably we will need some additional network to go over the resulting output and take care of any additional stuff.
        # another, more wholistic, solution will be to use general recurrent networks with perhapse: optical flow on last clean frame + latent recurrent states on features.
        # another solution might be to reuse deep features from previous frame and do optical flow on those.
        # another solution might be to use something like EDVR's PCD+TSA but on features, which implicitely is optical flow but can be other things the network chooses.
        ####################################

        ### Get Noise Sigma Map: ###
        if td.noise_map_for_loss_source == 'GT':
            od.noise_sigma_map_for_loss = id.noise_sigma_map
        elif td.noise_map_for_loss_source == 'Estimate':
            od.noise_sigma_map_for_loss = od.noise_sigma_map  # Estimate

        ### Pixel Counts Map For Loss: ###
        if td.reset_gate_for_loss_source == 'network_output':
            #(1). Assuming network outputs reset-gate/alpha-blend simply use that (probably with some constraint for network to output 1 to avoid trivial solution):
            od.pixel_counts_map_for_loss = od.pixel_counts_map
        elif td.reset_gate_for_loss_source == 'GT_occlusions':
            #(2). GT occlusions: #This, however, means that every occlusion means reseting...theoretically occluded things could have appeared in previous
            #     frames and no need to "start all over again", i can "remember" them.
            od.pixel_counts_map_for_loss = od.GT_pixel_counts_map_previous * od.GT_occlusions_map + 1
        elif td.reset_gate_for_loss_source == 'GT_optical_flow_error':
            #(3). Heuristic/Unsupervised Reset Gate:
            #   (3.1). incorrect optical flow (assuming i have GT_optical_flow, which i might approximate from clean images flow estimation)
            reset_gate_incorrect_optical_flow = (id.optical_flow_GT - od.optical_flow_estimate).abs() > td.optical_flow_error_threshold
            od.pixel_counts_map_for_loss = od.GT_pixel_counts_map_previous * reset_gate_incorrect_optical_flow + 1
        elif td.reset_gate_for_loss_source == 'data_term_inconsistencies':
            #   (3.2). incorrect values flow (data term in unsupervised workflowes, perhapse use clean images)
            reset_gate_data_inconsistencies = (id.original_image_previous_warped - id.original_image_current).abs() > td.intensity_difference_error_threshold
            od.pixel_counts_map_for_loss = od.GT_pixel_counts_map_previous * reset_gate_data_inconsistencies + 1
        #   (3.3). occlusions from left-right/right-left inconsistencies on clean images (approximating occlusion inference from GT optical flows)
        elif td.reset_gate_for_loss_source == 'leftright_rightleft_optical_flow_inferred_occlusions':
            occlusions_from_optical_flow_inconsistencies = get_occlusions_from_GT_optical_flow(id.left_right_optical_flow, id.right_left_optical_flow)
            od.pixel_counts_map_for_loss = od.GT_pixel_counts_map_previous * occlusions_from_optical_flow_inconsistencies + 1

        ### Value For Loss: ###
        od.difference_value_term_for_loss = od.noise_sigma_map_for_loss * 1/torch.sqrt(od.pixel_counts_map_for_loss)

    def adjust_recursive_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Get Recursion/Temporal Parameters From Dictionary: ###
        time_steps_weights = Train_dict.time_steps_weights
        number_of_total_backward_steps_per_image = Train_dict.number_of_total_backward_steps_per_image  # number of backward steps before changing images batch
        number_of_time_steps = Train_dict.number_of_time_steps_in_one_backward  # number of time steps before doing a .backward()
        number_of_time_steps_from_batch_start = Train_dict.number_of_time_steps_from_batch_start
        current_time_step = Train_dict.current_time_step

        ### Get Lambda_Time: ###
        if Train_dict.now_training:
            if len(time_steps_weights) == Train_dict.number_of_time_steps_in_one_backward:
                lambda_time = time_steps_weights[current_time_step]
            elif len(time_steps_weights) == Train_dict.number_of_total_backward_steps_per_image * Train_dict.number_of_time_steps_in_one_backward:
                lambda_time = time_steps_weights[number_of_time_steps_from_batch_start]
            td.lambda_time = lambda_time
        else:
            td.lambda_time = 1

        ### Flags: ###
        flag_residual_learning = Train_dict.flag_residual_learning
        flag_learn_clean_or_running_average = Train_dict.flag_learn_clean_or_running_average
        flag_clip_noisy_tensor_after_noising = Train_dict.flag_clip_noisy_tensor_after_noising

        ### Use Residual Learning If Wanted: ###
        if flag_learn_clean_or_running_average == 'clean':
            id.gt_frame_for_loss = id.current_gt_clean_frame
            od.model_output = od.model_output[:, 0:1, :, :]
            od.clean_frame_for_loss = od.model_output[:, 0:1, :, :]
        else:
            id.gt_frame_for_loss = id.current_moving_average
            od.model_output = od.model_output[:, 0:1, :, :]
            od.clean_frame_for_loss = od.model_output[:, 0:1, :, :]


        ### Clip Network Output If Wanted: ###
        if flag_clip_noisy_tensor_after_noising:
            id.gt_frame_for_loss = id.gt_frame_for_loss.clamp(0,1)
            od.model_output = od.model_output.clamp(0,1)

        return id, od, td

    def adjust_internal_states_for_next_iteration(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        return id, od, td, model

    def keep_track_of_things_for_calllbacks(self, id, od, td, model):
        ### Keep Track Of Certain Things: ###  #TODO: make this easier to read
        if hasattr(od, 'Reset_Gates_Combine_list') == False:
            od.Reset_Gates_Combine_list = [model.Reset_Gates_Combine.detach()]
        else:
            od.Reset_Gates_Combine_list.append(model.Reset_Gates_Combine.detach())
        if hasattr(od, 'reset_gates_classic_list') == False:
            od.reset_gates_classic_list = [model.reset_gates_classic.detach()]
        else:
            od.reset_gates_classic_list.append(model.reset_gates_classic.detach())
        if hasattr(od, 'noise_map_current') == False:
            od.noise_map_current_list = [model.noise_map_current.detach()]
        else:
            od.noise_map_current_list.append(model.noise_map_current.detach())

        ### TODO: maybe simply pass in model itself into loss object? ....don't know...: ###
        od.Reset_Gates_Combine = model.Reset_Gates_Combine
        od.reset_gates_classic = model.reset_gates_classic
        od.noise_map_current = model.noise_map_current

        ### Keep Track Of Previous Frames For Later Saving/Printing: ###
        self.original_frame_previous = id.original_frame_current
        self.noisy_frame_previous = id.noisy_frame_current

        return id, od, td, model

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        od.model_output = od.model_output

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_recursive_loss_parameters(id, od, td)

        ### Keep Track Of Certain Things For Callback: ###
        id, od, td, model = self.keep_track_of_things_for_calllbacks(id, od, td, model)

        ### Assign Internal States: ###
        id, od, td, model = self.adjust_internal_states_for_next_iteration(id, od, td, model)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##


class PostProcessing_ClassicIIR_AlphaBlend(PostProcessing_ClassicIIR):
    def __init__(self, **kw):
        super(PostProcessing_ClassicIIR_AlphaBlend, self).__init__()


    def keep_track_of_things_for_calllbacks(self, id, od, td, model):
        ### Keep Track Of Certain Things: ###
        return id, od, td, model


class PostProcessing_ClassicIIR_ResetGate(PostProcessing_ClassicIIR):
    def __init__(self, **kw):
        super(PostProcessing_ClassicIIR_ResetGate, self).__init__()


    def keep_track_of_things_for_calllbacks(self, id, od, td, model):
        ### Keep Track Of Certain Things: ###
        return id, od, td, model

    def adjust_recursive_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Get Recursion/Temporal Parameters From Dictionary: ###
        time_steps_weights = Train_dict.time_steps_weights
        number_of_total_backward_steps_per_image = Train_dict.number_of_total_backward_steps_per_image  # number of backward steps before changing images batch
        number_of_time_steps = Train_dict.number_of_time_steps_in_one_backward  # number of time steps before doing a .backward()
        number_of_time_steps_from_batch_start = Train_dict.number_of_time_steps_from_batch_start
        current_time_step = Train_dict.current_time_step

        ### Get Lambda_Time: ###
        if Train_dict.now_training:
            if len(time_steps_weights) == Train_dict.number_of_time_steps_in_one_backward:
                lambda_time = time_steps_weights[current_time_step]
            elif len(time_steps_weights) == Train_dict.number_of_total_backward_steps_per_image * Train_dict.number_of_time_steps_in_one_backward:
                lambda_time = time_steps_weights[number_of_time_steps_from_batch_start]
            td.lambda_time = lambda_time
        else:
            td.lambda_time = 1

        ### Flags: ###
        flag_residual_learning = Train_dict.flag_residual_learning
        flag_learn_clean_or_running_average = Train_dict.flag_learn_clean_or_running_average
        flag_clip_noisy_tensor_after_noising = Train_dict.flag_clip_noisy_tensor_after_noising

        ### Use Residual Learning If Wanted: ###
        if flag_learn_clean_or_running_average == 'clean':
            id.gt_frame_for_loss = id.current_gt_clean_frame
            od.model_output = od.model_output[:, 0:1, :, :]
            od.clean_frame_for_loss = od.model_output[:, 0:1, :, :]
        else:
            id.gt_frame_for_loss = id.current_moving_average
            od.model_output = od.model_output[:, 0:1, :, :]
            od.clean_frame_for_loss = od.model_output[:, 0:1, :, :]


        ### Clip Network Output If Wanted: ###
        if flag_clip_noisy_tensor_after_noising:
            id.gt_frame_for_loss = id.gt_frame_for_loss.clamp(0,1)
            od.model_output = od.model_output.clamp(0,1)

        return id, od, td


####################################################################################################################################################


####################################################################################################################################################
class PreProcessing_NeuralIIR(PreProcessing_Denoise_Recursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_NeuralIIR, self).__init__()

    def adjust_inner_states_for_new_iteration(self, id, td, model):
        ### Start / Initiailizations: ###
        if td.reset_hidden_states_flag == 0:
            # Zero/Initialize at start of batch: #
            1

        ### In Between Iterations Keep Accumulating Graph: ###
        if td.reset_hidden_states_flag == 1:
            # DO NOTHING!!!! - Keep Accumulating Graph: ###
            1


        ### Between Backward Steps -> Detach Previous States: ###
        if td.reset_hidden_states_flag == 2:
            # Use the previous hidden states but detach for memory usage limitation purposes: #
            model.clean_frame_current = model.clean_frame_current.detach()
            model.clean_frame_previous = model.clean_frame_previous.detach()
            model.clean_frame_previous_warped = model.clean_frame_previous_warped.detach()

            model.pixel_counts_map_current = model.pixel_counts_map_current.detach()
            model.pixel_counts_map_previous = model.pixel_counts_map_previous.detach()
            model.pixel_counts_map_previous_warped = model.pixel_counts_map_previous_warped.detach()

            model.noise_map_current = model.noise_map_current.detach()
            model.noise_map_previous = model.noise_map_previous.detach()

        return id, td, model


    def adjust_inputs_dict_for_loss(self, inputs_dict, Train_dict, model):
        # TODO: transfer to post-processing function like i did for the simple recursive case
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Recursion/Temporal Parameters: ###
        number_of_total_backward_steps_per_image = Train_dict.number_of_total_backward_steps_per_image  # number of backward steps before changing images batch
        number_of_time_steps = Train_dict.number_of_time_steps_in_one_backward  # number of time steps before doing a .backward()
        number_of_time_steps_from_batch_start = Train_dict.number_of_time_steps_from_batch_start
        frame_index_to_use = number_of_time_steps_from_batch_start + 1  #(*). NOTICE!!!! i'm initializing things with first frame and starts predicting with the seoncd

        ### Modify Dict To Follow Wanted Stuff: ### #TODO: pass to an internal function
        id.current_gt_clean_frame = id.output_frames_original[:,frame_index_to_use:frame_index_to_use + 1, :,:].to(Train_dict.device)
        id.current_noisy_frame = id.output_frames_noisy[:,frame_index_to_use:frame_index_to_use + 1, :,:].to(Train_dict.device)
        id.gt_noise_residual = id.current_gt_clean_frame - id.current_noisy_frame
        T = number_of_time_steps_from_batch_start + 1
        if hasattr(id, 'current_moving_average'):
            id.current_moving_average = (1 - 1 / T) * id.current_moving_average + 1 / T * id.current_noisy_frame
        else:
            id.current_moving_average = id.current_noisy_frame
        return id, td, model

    def get_normalization_constant(self, network_input, id):
        # normalization_constant = get_max_correct_form(network_input, 0) # TODO: this normalization constant changes from working point to working point. i should do this, or get this from dataset object?
        normalization_constant = 255
        self.normalization_constant = normalization_constant
        # normalization_constant = 255 / 10
        return normalization_constant

    def normalize_noise_input(self, network_input, id, td):
        #TODO: get this shit in order!!!!
        td.sigma_to_model = np.float32(td.sigma_to_model / self.normalization_constant)
        id.sigma_to_model = td.sigma_to_model
        id.sigma_to_model = id.sigma_to_model * torch.ones_like(network_input).to(network_input.device)
        id.sigma_to_model = id.sigma_to_model[:, 0:1, :, :]
        return network_input, id, td

    def normalize_inputs(self, network_input, id, td):
        ### Normalize: ###
        normalization_constant = self.get_normalization_constant(network_input)
        network_input = network_input / normalization_constant
        id.output_frames_original = id.output_frames_original / normalization_constant
        id.output_frames_noisy = id.output_frames_noisy / normalization_constant
        id.center_frame_noisy = id.center_frame_noisy / normalization_constant
        id.center_frame_pseudo_running_mean = id.center_frame_pseudo_running_mean / normalization_constant
        id.center_frame_actual_mean = id.center_frame_actual_mean / normalization_constant
        id.center_frame_original = id.center_frame_original / normalization_constant
        network_input, id, td = self.normalize_noise_input(network_input, id, td)
        return network_input, id, td

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        number_of_time_steps_from_batch_start = td.number_of_time_steps_from_batch_start
        frame_index_to_use = number_of_time_steps_from_batch_start + 1

        ### Get Input To Model: ###
        input_frames = id.output_frames_noisy[:, frame_index_to_use, :, :]  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(input_frames)
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(input_frames)
        else:
            network_input = input_frames


        ### Normalize Inputs: ###
        if td.number_of_time_steps_from_batch_start == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### Take Care Of Initialization At Start Of Movie: ###
        if number_of_time_steps_from_batch_start == 0:
            ### Initialize More Things For Internal Tracking: ###
            id.original_frame_previous = id.output_frames_original[:, 0].data
            id.noisy_frame_previous = id.output_frames_noisy[:, 0].data
            id.clean_frame_perfect_averaging = id.noisy_frame_previous
            id.clean_frame_simple_averaging = id.noisy_frame_previous

            ### Initialize Model Internal Variables: ###
            model = self.initialize_model_variables(model, id)

        return network_input, id, td, model

    def get_wanted_inputs_to_device(self, id, td):
        for k, v in id.items():
            id[k] = id[k].to(td.device)
        return id, td

    def initialize_model_variables(self, model, id):
        # model.clean_frame_previous = id.noisy_frame_current_cropped  # initialize internal "cleaned" image to simply be first frame, which is noisy
        model.clean_frame_previous = id.noisy_frame_previous  # initialize internal "cleaned" image to simply be first frame, which is noisy
        model.pixel_counts_map_previous = torch.ones_like(model.clean_frame_previous)
        model.pixel_counts_map_previous_warped = torch.ones_like(model.clean_frame_previous)
        model.noise_map_previous = None
        return model

    def get_wanted_inputs_to_device(self, id, device):
        id.original_frame_current = id.original_frame_current.to(device)
        id.noisy_frame_current = id.noisy_frame_current.to(device)
        id.current_gt_clean_frame = id.current_gt_clean_frame.to(device)
        id.current_moving_average = id.current_moving_average.to(device)
        id.output_frames_noisy_running_average = id.output_frames_noisy_running_average.to(device)
        id.noisy_frame_current_cropped = id.noisy_frame_current_cropped.to(device)
        id.original_frame_current_cropped = id.original_frame_current_cropped.to(device)
        id.original_frame_current_cropped = id.original_frame_current_cropped.to(device)
        return id

    def forward(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Adjust Preprocessing Parameters: ###
        id, td, model = self.adjust_inner_states_for_new_iteration(id, td, model)
        id, td, model = self.adjust_inputs_dict_for_loss(id, td, model)
        network_input, id, td, model = self.prepare_inputs_for_net(id, td, model)

        return network_input, id, td, model


class PostProcessing_NeuralIIR(nn.Module):
    def __init__(self, **kw):
        super(PostProcessing_NeuralIIR, self).__init__()

    def adjust_target_output_for_model(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Get Noise Sigma Map: ###
        if td.noise_map_for_loss_source == 'GT':
            od.noise_sigma_map_for_loss = id.noise_sigma_map
        elif td.noise_map_for_loss_source == 'Estimate':
            od.noise_sigma_map_for_loss = od.noise_sigma_map  # Estimate

        ### Pixel Counts Map For Loss: ###
        if td.reset_gate_for_loss_source == 'network_output':
            # (1). Assuming network outputs reset-gate/alpha-blend simply use that (probably with some constraint for network to output 1 to avoid trivial solution):
            od.pixel_counts_map_for_loss = od.pixel_counts_map
        elif td.reset_gate_for_loss_source == 'GT_occlusions':
            # (2). GT occlusions: #This, however, means that every occlusion means reseting...theoretically occluded things could have appeared in previous
            #     frames and no need to "start all over again", i can "remember" them.
            od.pixel_counts_map_for_loss = od.GT_pixel_counts_map_previous * od.GT_occlusions_map + 1
        elif td.reset_gate_for_loss_source == 'GT_optical_flow_error':
            # (3). Heuristic/Unsupervised Reset Gate:
            #   (3.1). incorrect optical flow (assuming i have GT_optical_flow, which i might approximate from clean images flow estimation)
            reset_gate_incorrect_optical_flow = (id.optical_flow_GT - od.optical_flow_estimate).abs() > td.optical_flow_error_threshold
            od.pixel_counts_map_for_loss = od.GT_pixel_counts_map_previous * reset_gate_incorrect_optical_flow + 1
        elif td.reset_gate_for_loss_source == 'data_term_inconsistencies':
            #   (3.2). incorrect values flow (data term in unsupervised workflowes, perhapse use clean images)
            reset_gate_data_inconsistencies = (id.original_image_previous_warped - id.original_image_current).abs() > td.intensity_difference_error_threshold
            od.pixel_counts_map_for_loss = od.GT_pixel_counts_map_previous * reset_gate_data_inconsistencies + 1
        #   (3.3). occlusions from left-right/right-left inconsistencies on clean images (approximating occlusion inference from GT optical flows)
        elif td.reset_gate_for_loss_source == 'leftright_rightleft_optical_flow_inferred_occlusions':
            occlusions_from_optical_flow_inconsistencies = get_occlusions_from_GT_optical_flow(id.left_right_optical_flow, id.right_left_optical_flow)
            od.pixel_counts_map_for_loss = od.GT_pixel_counts_map_previous * occlusions_from_optical_flow_inconsistencies + 1

        ### Value For Loss: ###
        od.difference_value_term_for_loss = od.noise_sigma_map_for_loss * 1 / torch.sqrt(od.pixel_counts_map_for_loss)

    def adjust_recursive_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Get Recursion/Temporal Parameters From Dictionary: ###
        time_steps_weights = Train_dict.time_steps_weights
        number_of_total_backward_steps_per_image = Train_dict.number_of_total_backward_steps_per_image  # number of backward steps before changing images batch
        number_of_time_steps = Train_dict.number_of_time_steps_in_one_backward  # number of time steps before doing a .backward()
        number_of_time_steps_from_batch_start = Train_dict.number_of_time_steps_from_batch_start
        current_time_step = Train_dict.current_time_step

        ### Get Lambda_Time: ###
        if Train_dict.now_training:
            if len(time_steps_weights) == Train_dict.number_of_time_steps_in_one_backward:
                lambda_time = time_steps_weights[current_time_step]
            elif len(time_steps_weights) == Train_dict.number_of_total_backward_steps_per_image * Train_dict.number_of_time_steps_in_one_backward:
                lambda_time = time_steps_weights[number_of_time_steps_from_batch_start]
            td.lambda_time = lambda_time
        else:
            td.lambda_time = 1

        ### Flags: ###
        flag_residual_learning = Train_dict.flag_residual_learning
        flag_learn_clean_or_running_average = Train_dict.flag_learn_clean_or_running_average
        flag_clip_noisy_tensor_after_noising = Train_dict.flag_clip_noisy_tensor_after_noising

        ### Use Residual Learning If Wanted: ###
        if flag_learn_clean_or_running_average == 'clean':
            id.gt_frame_for_loss = id.current_gt_clean_frame
            od.model_output = od.model_output[:, 0:1, :, :]
        else:
            id.gt_frame_for_loss = id.current_moving_average
            od.model_output = od.model_output[:, 0:1, :, :]

        ### Clip Network Output If Wanted: ###
        if flag_clip_noisy_tensor_after_noising:
            id.gt_frame_for_loss = id.gt_frame_for_loss.clamp(0, 1)
            od.model_output = od.model_output.clamp(0, 1)

        return id, od, td

    def adjust_internal_states_for_next_iteration(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        return id, od, td, model

    def keep_track_of_things_for_calllbacks(self, id, od, td, model):
        ### Keep Track Of Certain Things: ###  #TODO: make this easier to read
        if hasattr(od, 'Reset_Gates_Combine_list') == False:
            od.Reset_Gates_Combine_list = [model.Reset_Gates_Combine.detach()]
        else:
            od.Reset_Gates_Combine_list.append(model.Reset_Gates_Combine.detach())
        if hasattr(od, 'reset_gates_classic_list') == False:
            od.reset_gates_classic_list = [model.reset_gates_classic.detach()]
        else:
            od.reset_gates_classic_list.append(model.reset_gates_classic.detach())
        if hasattr(od, 'noise_map_current') == False:
            od.noise_map_current_list = [model.noise_map_current.detach()]
        else:
            od.noise_map_current_list.append(model.noise_map_current.detach())

        ### TODO: maybe simply pass in model itself into loss object? ....don't know...: ###
        od.Reset_Gates_Combine = model.Reset_Gates_Combine
        od.reset_gates_classic = model.reset_gates_classic
        od.noise_map_current = model.noise_map_current

        ### Keep Track Of Previous Frames For Later Saving/Printing: ###
        self.original_frame_previous = id.original_frame_current
        self.noisy_frame_previous = id.noisy_frame_current

        return id, od, td, model

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        od.model_output = od.model_output

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_recursive_loss_parameters(id, od, td)

        ### Keep Track Of Certain Things For Callback: ###
        id, od, td, model = self.keep_track_of_things_for_calllbacks(id, od, td, model)

        ### Assign Internal States: ###
        id, od, td, model = self.adjust_internal_states_for_next_iteration(id, od, td, model)

        return od.model_output, od, id, td, model  # TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##
####################################################################################################################################################

####################################################################################################################################################

class PreProcessing_Gimbaless(PreProcessing_Denoise_NonRecursive_Base):

    def __init__(self, **kw):
        super(PreProcessing_Gimbaless, self).__init__()
        # self.flag_BW2RGB = True
        self.flag_BW2RGB = False

    def get_normalization_constant(self, network_input, id):
        # normalization_constant = get_max_correct_form(network_input, 0) # TODO: this normalization constant changes from working point to working point. i should do this, or get this from dataset object?

        if network_input.mean() < 256:
            normalization_constant = 255
        else:
            normalization_constant = 2**16 -1
        # print (network_input.mean(), normalization_constant)
        self.normalization_constant = normalization_constant
        # normalization_constant = 255 / 10
        return normalization_constant

    def forward(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict
        ### Adjust Preprocessing Parameters: ###
        network_input, id, td, model = self.prepare_inputs_for_net(id, td, model)
        id, td, model = self.adjust_inputs_dict_for_loss(id, td, model)
        ### Gimbaless - align and avearge images: ###
        output_tensors_list = []
        number_of_samples_per_sub_batch = 5
        B,T,C,H,W = id.output_frames_noisy.shape
        #(1). Loop over batches, and for each batch index perform gimbaless
        for i in np.arange(B):
            output_tensors_list.append(perform_gimbaless_cross_correlation_on_image_batch(id.output_frames_noisy[i].permute([1,0,2,3]), number_of_samples_per_sub_batch))
        output_tensors = torch.cat(output_tensors_list, 0)
        output_tensors = output_tensors.unsqueeze(2)
        #(2). Assign things properly to make everything in the tashtit consistent:
        # id.output_frames_noisy = output_tensors
        id.output_frames_noisy = id.output_frames_noisy[:,number_of_samples_per_sub_batch//2:id.output_frames_original.shape[1]:number_of_samples_per_sub_batch,:,:,:]
        id.output_frames_noisy_HR = id.output_frames_noisy_HR[:,number_of_samples_per_sub_batch//2:id.output_frames_original.shape[1]:number_of_samples_per_sub_batch,:,:,:]
        id.output_frames_original = id.output_frames_original[:,number_of_samples_per_sub_batch//2:id.output_frames_original.shape[1]:number_of_samples_per_sub_batch,:,:,:]
        [B,T,C,H,W] = id.output_frames_original.shape
        id.center_frame_noisy = id.output_frames_noisy_HR[:, T // 2]
        id.center_frame_noisy_HR = id.output_frames_noisy_HR[:, T // 2]
        id.center_frame_original = id.output_frames_original[:, T//2]
        id.center_frame_actual_mean = id.output_frames_noisy_HR.mean(1,True)

        network_input = output_tensors
        # imshow_torch(id.output_frames_noisy[0,0])
        # imshow_torch(network_input[0,0])
        return network_input, id, td, model


class PostProcessing_Gimbaless(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_Gimbaless, self).__init__()
        self.flag_RGB2BW = True

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        # (*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output.unsqueeze(1)
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output.unsqueeze(1)

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model

class PreProcessing_Gimbaless_VRT(PreProcessing_Denoise_NonRecursive_Base):

    def __init__(self, **kw):
        super(PreProcessing_Gimbaless_VRT, self).__init__()
        # self.flag_BW2RGB = True
        self.flag_BW2RGB = False
        self.flag_denoise = True

    def get_normalization_constant(self, network_input, id):
        # normalization_constant = get_max_correct_form(network_input, 0) # TODO: this normalization constant changes from working point to working point. i should do this, or get this from dataset object?
        self.normalization_constant = int(network_input.max().item())
        return self.normalization_constant

    def forward(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict
        ### Adjust Preprocessing Parameters: ###
        network_input, id, td, model = self.prepare_inputs_for_net(id, td, model)
        id, td, model = self.adjust_inputs_dict_for_loss(id, td, model)
        ### Gimbaless - align and avearge images: ###
        output_tensors_list = []
        number_of_samples_per_sub_batch = 5
        B,T,C,H,W = id.output_frames_noisy.shape
        #(1). Loop over batches, and for each batch index perform gimbaless
        for i in np.arange(B):
            output_tensors_list.append(perform_gimbaless_cross_correlation_on_image_batch(id.output_frames_noisy[i].permute([1,0,2,3]), number_of_samples_per_sub_batch))
        output_tensors = torch.cat(output_tensors_list, 0)
        output_tensors = output_tensors.unsqueeze(2)
        #(2). Assign things properly to make everything in the tashtit consistent:
        # id.output_frames_noisy = output_tensors
        id.output_frames_noisy = id.output_frames_noisy[:,number_of_samples_per_sub_batch//2:id.output_frames_original.shape[1]:number_of_samples_per_sub_batch,:,:,:]
        id.output_frames_noisy_HR = id.output_frames_noisy_HR[:,number_of_samples_per_sub_batch//2:id.output_frames_original.shape[1]:number_of_samples_per_sub_batch,:,:,:]
        id.output_frames_original = id.output_frames_original[:,number_of_samples_per_sub_batch//2:id.output_frames_original.shape[1]:number_of_samples_per_sub_batch,:,:,:]
        [B,T,C,H,W] = id.output_frames_original.shape
        id.center_frame_noisy = id.output_frames_noisy_HR[:, T // 2]
        id.center_frame_noisy_HR = id.output_frames_noisy_HR[:, T // 2]
        id.center_frame_original = id.output_frames_original[:, T//2]
        id.center_frame_actual_mean = id.output_frames_noisy_HR.mean(1,True)
        # save_image_torch(folder_path='/home/davidk/users/omer/Pytorch_Checkpoints/Inference/Denoise_1VRT_Frames=30--6_RGB_Pretrained.py',
        #                  filename='center_noisy.png', torch_tensor=id.center_frame_noisy*255,
        #                  flag_array_to_uint8=True)
        # save_image_torch(
        #     folder_path='/home/davidk/users/omer/Pytorch_Checkpoints/Inference/Denoise_1VRT_Frames=30--6_RGB_Pretrained.py',
        #     filename='center_original.png', torch_tensor=id.center_frame_original.clamp(0,1) * 255,
        #     flag_array_to_uint8=True)
        # save_image_torch(
        #     folder_path='/home/davidk/users/omer/Pytorch_Checkpoints/Inference/Denoise_1VRT_Frames=30--6_RGB_Pretrained.py',
        #     filename='output_tensor.png', torch_tensor=output_tensors[:,2] * 255,
        #     flag_array_to_uint8=True)
        network_input = BW2RGB(output_tensors)
        # imshow_torch(id.output_frames_noisy[0,0])
        # imshow_torch(network_input[0,0])

        ### Concat Frames & Sigma: ###
        if self.flag_denoise:
            B, T, C, H, W = network_input.shape
            network_input = torch.cat((network_input, torch.zeros((B, T, 1, H, W)).to(network_input.device)), 2)

        network_input = network_input, td  # For special vrt forward
        return network_input, id, td, model


class PostProcessing_Gimbaless_VRT(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_Gimbaless_VRT, self).__init__()
        self.flag_RGB2BW = True

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        # (*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output.unsqueeze(1)
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output.unsqueeze(1)

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model





class PreProcessing_EDVR(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_EDVR, self).__init__()
        # self.flag_BW2RGB = True
        self.flag_BW2RGB = False

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        return network_input, id, td, model

class PostProcessing_EDVR(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_EDVR, self).__init__()
        self.flag_RGB2BW = False

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##


##########################################################################################################
class PreProcessing_EDVR_15f(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_EDVR_15f, self).__init__()
        # self.flag_BW2RGB = True
        self.flag_BW2RGB = False

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        elif self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        network_input = torch.cat((network_input[:,0:6], network_input[:,6:7], network_input[:,6:7], network_input[:,6:7], network_input[:,7:15]), 1)
        b, t, c, h, w = network_input.size()
        network_input = network_input.reshape(b,5,3,h,w)
        return network_input, id, td, model

class PostProcessing_EDVR_15f(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_EDVR_15f, self).__init__()
        self.flag_RGB2BW = False

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##


##########################################################################################################
class PreProcessing_EDVR_Thermal(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_EDVR_Thermal, self).__init__()
        # self.flag_BW2RGB = True
        self.flag_BW2RGB = False

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        elif self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        network_input = torch.cat((network_input[:,:,0,:,:].unsqueeze(2), network_input[:,:,0,:,:].unsqueeze(2),network_input[:,:,0,:,:].unsqueeze(2)), 2)
        return network_input, id, td, model

class PostProcessing_EDVR_thermal(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_EDVR_thermal, self).__init__()
        self.flag_RGB2BW = False

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##

####################################################################################################################################################

class PreProcessing_Restformer(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_Restformer, self).__init__()
        # self.flag_BW2RGB = True
        self.flag_BW2RGB = False


class PostProcessing_Restformer(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_Restformer, self).__init__()
        self.flag_RGB2BW = False

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model



####################################################################################################################################################
class PreProcessing_RLSP(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_RLSP, self).__init__()
        self.flag_BW2RGB = False


class PostProcessing_RLSP(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_RLSP, self).__init__()
        self.flag_RGB2BW = False

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)


        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##
####################################################################################################################################################

####################################################################################################################################################
from RDND_proper.models.RLSP.RLSP.pytorch.functions import shuffle_down, shuffle_up
class PreProcessing_RLSP_Recursive(PreProcessing_Denoise_Recursive_Base):
    def __init__(self,number_of_frames, **kw):
        super(PreProcessing_RLSP_Recursive, self).__init__()
        self.number_of_frames=number_of_frames


    def adjust_inner_states_for_new_iteration(self, id, td, model):
        ### Start / Initiailizations: ###
        if td.reset_hidden_states_flag == 0:
            # Zero/Initialize at start of batch: #
            model.out = shuffle_up(torch.zeros_like(id.output_frames_noisy[:, 0, :, :, :]).repeat(1, model.number_of_input_channels * model.upsample_factor ** 2, 1, 1), model.upsample_factor)
            model.state = torch.zeros_like(id.output_frames_noisy[:, 0, 0:1, ...]).repeat(1, model.state_dimension, 1, 1)

        ### In Between Iterations Keep Accumulating Graph: ###
        if td.reset_hidden_states_flag == 1:
            # DO NOTHING!!!! - Keep Accumulating Graph: ###
            1

        ### Between Backward Steps -> Detach Previous States: ###
        if td.reset_hidden_states_flag == 2:
            # Use the previous hidden states but detach for memory usage limitation purposes: #
            model.out = model.out.detach()
            model.state = model.state.detach()

        return id, td, model

    def normalize_inputs(self, network_input, id, td):
        ### Normalize: ###
        network_input, id, td = super().normalize_inputs(network_input, id, td)
        id.output_frames_noisy_HR = id.output_frames_noisy_HR / self.normalization_constant
        id.center_frame_noisy_HR = id.center_frame_noisy_HR / self.normalization_constant
        return network_input, id, td

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        number_of_time_steps_from_batch_start = td.number_of_time_steps_from_batch_start

        ### Get Current Input Frames: ###, TODO(i changed 3 to self.number_of_frames)
        input_frames = id.output_frames_noisy[:, number_of_time_steps_from_batch_start:number_of_time_steps_from_batch_start + self.number_of_frames, :, :]  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(input_frames)
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(input_frames)
        else:
            network_input = input_frames

        #print(network_input.shape)
        ### Normalize Inputs: ###
        if td.number_of_time_steps_from_batch_start == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### Concat Previous State To Input Frames: ###
        return network_input, id, td, model

class PostProcessing_RLSP_Recursive(PostProcessing_Denoise_Recursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_RLSP_Recursive, self).__init__()
        #self.flag_RGB2BW = False

    def adjust_recursive_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Get Lambda_Time: ###
        if len(td.time_steps_weights) == Train_dict.number_of_time_steps_in_one_backward:
            lambda_time = td.time_steps_weights[td.current_time_step]
        else:
            lambda_time = td.time_steps_weights[td.number_of_time_steps_from_batch_start]
        td.lambda_time = lambda_time

        ### Modify Dict To Follow Wanted Stuff: ###
        #(*). Frame Index To Use/Predict:
        frame_index_to_use_in_loss = td.number_of_time_steps_from_batch_start + self.number_of_model_input_frames//2  # middle frame
        # frame_index_to_use_in_loss = self.number_of_model_input_frames - 1 # last frame
        #(*). Assign Wanted Frames For Loss:
        id.current_gt_clean_frame = id.output_frames_original[:, frame_index_to_use_in_loss, :, :]
        id.current_noisy_frame = id.output_frames_noisy_HR[:, frame_index_to_use_in_loss, :, :]
        id.gt_noise_residual = id.current_noisy_frame - id.current_gt_clean_frame
        T = td.number_of_time_steps_from_batch_start + 1
        if hasattr(id, 'current_moving_average'):
            id.current_moving_average = (1 - 1 / T) * id.current_moving_average + 1 / T * id.current_noisy_frame
        else:
            id.current_moving_average = id.current_noisy_frame

        ### Use Residual Learning If Wanted: ###
        if Train_dict.flag_learn_clean_or_running_average == 'clean':
            id.gt_frame_for_loss = id.current_gt_clean_frame
            od.model_output = od.model_output[:, 0:1, :, :]
        else:
            id.gt_frame_for_loss = id.current_moving_average
            od.model_output = od.model_output[:, 0:1, :, :]

        ### Clip Network Output If Wanted: ###
        if Train_dict.flag_clip_noisy_tensor_after_noising:
            id.gt_frame_for_loss = id.gt_frame_for_loss.clamp(0,1)
            od.model_output = od.model_output.clamp(0,1)

        return id, od, td

    def adjust_internal_states_for_next_iteration(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        return id, od, td, model

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        #(*). The model outputs three intermediate results (deep-supervision)!!!!
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.model_output)
        elif self.flag_BW2RGB:
            od.model_output = BW2RGB_interleave_T(od.model_output)
        
        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_recursive_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##
####################################################################################################################################################


####################################################################################################################################################
from RDND_proper.models.RLSP.RLSP.pytorch.functions import shuffle_down, shuffle_up
class PreProcessing_EGVSR_Recursive(PreProcessing_Denoise_Recursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_EGVSR_Recursive, self).__init__()

    def adjust_inner_states_for_new_iteration(self, id, td, model):
        ### Start / Initiailizations: ###
        if td.reset_hidden_states_flag == 0:
            # Zero/Initialize at start of batch: #
            model.hr_prev = shuffle_up(torch.zeros_like(id.output_frames_noisy[:, 0, :, :, :]).repeat(1, model.upsample_factor ** 2, 1, 1), model.upsample_factor)
            # id.lr_prev = torch.zeros_like(id.output_frames_noisy[:, 0, :, :, :])
            id.lr_prev = id.output_frames_noisy[:, 0, :, :, :]

        ### In Between Iterations Keep Accumulating Graph: ###
        if td.reset_hidden_states_flag == 1:
            # DO NOTHING!!!! - Keep Accumulating Graph: ###
            1

        ### Between Backward Steps -> Detach Previous States: ###
        if td.reset_hidden_states_flag == 2:
            # Use the previous hidden states but detach for memory usage limitation purposes: #
            model.hr_prev = model.hr_prev.detach()

        return id, td, model

    def get_normalization_constant(self, network_input, id):
        # normalization_constant = get_max_correct_form(network_input, 0) # TODO: this normalization constant changes from working point to working point. i should do this, or get this from dataset object?
        normalization_constant = 255
        self.normalization_constant = normalization_constant
        # normalization_constant = 255 / 10
        return normalization_constant

    def normalize_inputs(self, network_input, id, td):
        ### Normalize: ###
        network_input, id, td = super().normalize_inputs(network_input, id, td)
        id.output_frames_noisy_HR = id.output_frames_noisy_HR / self.normalization_constant
        id.center_frame_noisy_HR = id.center_frame_noisy_HR / self.normalization_constant
        id.lr_prev = id.lr_prev / self.normalization_constant
        return network_input, id, td

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        number_of_time_steps_from_batch_start = td.number_of_time_steps_from_batch_start

        ### Get Current Input Frames: ###
        input_frames = id.output_frames_noisy[:, number_of_time_steps_from_batch_start, :, :, :]  # Get relevant frame
        if self.flag_BW2RGB:
            input_frames = BW2RGB_interleave_T(input_frames)
            id.lr_prev = BW2RGB_interleave_T(id.lr_prev)
            model.hr_prev = BW2RGB_interleave_T(model.hr_prev)

        ### Normalize Inputs & Take Care Of Stuff Needed At Step 0: ###
        if td.number_of_time_steps_from_batch_start == 0:
            network_input, id, td = self.normalize_inputs(input_frames, id, td)
        else:
            network_input = input_frames

        ### Concat Previous State To Input Frames: ###
        lr_curr = network_input
        lr_prev = id.lr_prev
        id.lr_curr = network_input
        network_input = [lr_curr, lr_prev, model.hr_prev]

        ### Take Care Of Stuff For Next Iteration: ###


        return network_input, id, td, model

    def get_wanted_inputs_to_device(self, id, td):
        for k, v in id.items():
            id[k] = id[k].to(td.device)
        return id, td

    def forward(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Adjust Preprocessing Parameters: ###
        id, td = self.get_wanted_inputs_to_device(id, td)
        id, td, model = self.adjust_inner_states_for_new_iteration(id, td, model)
        network_input, id, td, model = self.prepare_inputs_for_net(id, td, model)
        return network_input, id, td, model

class PostProcessing_EGVSR_Recursive(PostProcessing_Denoise_Recursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_EGVSR_Recursive, self).__init__()
        self.number_of_model_input_frames = 1
        self.flag_RGB2BW = False

    def adjust_recursive_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Get Lambda_Time: ###
        if len(td.time_steps_weights) == Train_dict.number_of_time_steps_in_one_backward:
            lambda_time = td.time_steps_weights[td.current_time_step]
        else:
            lambda_time = td.time_steps_weights[td.number_of_time_steps_from_batch_start]
        td.lambda_time = lambda_time

        ### Modify Dict To Follow Wanted Stuff: ###
        #(*). Frame Index To Use/Predict:
        frame_index_to_use_in_loss = td.number_of_time_steps_from_batch_start + self.number_of_model_input_frames//2  # middle frame
        # frame_index_to_use_in_loss = self.number_of_model_input_frames - 1 # last frame
        #(*). Assign Wanted Frames For Loss:
        id.current_gt_clean_frame = id.output_frames_original[:, frame_index_to_use_in_loss, :, :]
        id.current_noisy_frame = id.output_frames_noisy_HR[:, frame_index_to_use_in_loss, :, :]
        id.gt_noise_residual = id.current_noisy_frame - id.current_gt_clean_frame
        T = td.number_of_time_steps_from_batch_start + 1
        if hasattr(id, 'current_moving_average'):
            id.current_moving_average = (1 - 1 / T) * id.current_moving_average + 1 / T * id.current_noisy_frame
        else:
            id.current_moving_average = id.current_noisy_frame

        ### Use Residual Learning If Wanted: ###
        if Train_dict.flag_learn_clean_or_running_average == 'clean':
            id.gt_frame_for_loss = id.current_gt_clean_frame
            od.model_output = od.model_output[:, :, :, :]
        else:
            id.gt_frame_for_loss = id.current_moving_average
            od.model_output = od.model_output[:, :, :, :]

        ### Clip Network Output If Wanted: ###
        if Train_dict.flag_clip_noisy_tensor_after_noising:
            id.gt_frame_for_loss = id.gt_frame_for_loss.clamp(0,1)
            od.model_output = od.model_output.clamp(0,1)

        return id, od, td

    def adjust_internal_states_for_next_iteration(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        return id, od, td, model

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        od.hr_output = od.model_output[0]
        od.flow_output = od.model_output[1]
        model.hr_prev = od.model_output[0]
        if self.flag_RGB2BW:
            od.model_output = RGB2BW_interleave_T(od.hr_output)
        else:
            od.model_output = od.hr_output

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_recursive_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##
####################################################################################################################################################

####################################################################################################################################################
class PreProcessing_DVDNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_DVDNet, self).__init__()

    def forward(self, inputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Prepare Input: ###
        # sigma = torch.full((1, 1, 1, 1), Train_dict.sigma_to_model / 255.).type_as(id.output_frames)
        sigma = Train_dict.sigma_to_model/255  #TODO: make this sigma cme from inputs_dict maybe?
        input_frames = id.output_frames_noisy

        ### (*). The reason i'm checking dim=2 is that in DVDNet what i'm getting is [B,T,C,H,W]: ###
        if input_frames.shape[2] == 1:  #currently DVDNet only accept "RGB" images with 3 channels. TODO: make function BW2RGB or something
            input_frames = torch.cat((input_frames,input_frames,input_frames),2)
        network_input = (input_frames, sigma)

        return network_input, id


class PreProcessing_DVDNet_ExtraDataset(PreProcessing_Base):
    def __init__(self, **kw):
        super(PreProcessing_DVDNet_ExtraDataset, self).__init__()

    def forward(self, inputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Prepare Input: ###
        # sigma = torch.full((1, 1, 1, 1), Train_dict.sigma_to_model / 255.).type_as(id.output_frames)
        sigma = Train_dict.sigma_to_model/255  #TODO: make this sigma cme from inputs_dict maybe?
        input_frames = id.output_frames_noisy


        ### (*). The reason i'm checking dim=2 is that in DVDNet what i'm getting is [B,T,C,H,W]: ###
        if input_frames.shape[2] == 1:  #currently DVDNet only accept "RGB" images with 3 channels. TODO: make function BW2RGB or something
            input_frames = torch.cat((input_frames,input_frames,input_frames),2)
        network_input = (input_frames, sigma)

        return network_input, id


class PreProcessing_DVDNet_BW(PreProcessing_Base):
    def __init__(self, **kw):
        super(PreProcessing_DVDNet_BW, self).__init__()

    def forward(self, inputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Prepare Input: ###
        # sigma = torch.full((1, 1, 1, 1), Train_dict.sigma_to_model / 255.).type_as(id.output_frames)
        sigma = Train_dict.sigma_to_model/255  #TODO: make this sigma cme from inputs_dict maybe?
        input_frames = id.output_frames_noisy

        network_input = (input_frames, sigma)

        return network_input, id


class PostProcessing_DVDNet(nn.Module):
    def __init__(self, **kw):
        super(PostProcessing_DVDNet, self).__init__()

    def forward(self, inputs_dict, outputs_dict, Train_dict):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### Scale: ###
        od.temporal_part_output_frame = od.model_output.temporal_part_output_frame
        od.model_output = od.model_output.temporal_part_output_frame
        od.spatial_part_output_frames = od.model_output.spatial_part_output_frames

        ### Assign Network Output Tensor: ###
        od.model_output_for_callback = od.model_output
        od.center_clean_frame_estimate_for_callback = od.model_output
        od.model_output_for_tensorboard = od.model_output

        return od.model_output, od



####################################################################################################################################################
class PreProcessing_RAFT(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_RAFT, self).__init__()
        self.flag_BW2RGB = False

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### The Network Accepts A Weird Combination Of Parameters: ###
        #(1). Center Around 0:
        network_input = 2 * network_input - 1
        flow_init = None
        upsample = True
        test_mode = False
        #(2). Set Up Network Input To Fit What Network Expects:
        network_input = (network_input[:,0:3], network_input[:,3:6], model.number_of_iterations, flow_init, upsample, test_mode)
        return network_input, id, td, model

from RapidBase.Utils.Warping_Shifting import shift_matrix_subpixel_torch
class PostProcessing_RAFT_Translation(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_RAFT_Translation, self).__init__()
        self.flag_RGB2BW = False
        self.warp_object = Warp_Object()
        self.affine_warp_object = Warp_Tensors_Affine_Layer()

    def adjust_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        return id, od, td

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### the model outputs optical flow - average everything out to get Translation: ###
        od.optical_flow_estimates_list = od.model_output
        od.final_optical_flow_estimate = od.model_output[-1]
        od.translation_estimate = od.final_optical_flow_estimate.mean(-1,True).mean(-2,True).squeeze()

        ### Assign Network Output Tensor: ###
        od.model_output = od.translation_estimate

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##

class PostProcessing_RAFT_OpticalFlow(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_RAFT_OpticalFlow, self).__init__()
        self.flag_RGB2BW = False

    def adjust_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        return id, od, td

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### the model outputs optical flow - average everything out to get Translation: ###
        od.optical_flow_estimate = od.model_output[-1]

        ### Assign Network Output Tensor: ###
        od.model_output = od.optical_flow_estimate

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##
####################################################################################################################################################



####################################################################################################################################################
class PreProcessing_STARFlow(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_STARFlow, self).__init__()
        self.flag_BW2RGB = False

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        elif self.flag_RGB2BW:
            network_input = BW2RGB_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### The Network Accepts A Weird Combination Of Parameters: ###
        network_input = (network_input[:,0], network_input[:,1])
        return network_input, id, td, model

from RapidBase.Utils.Warping_Shifting import shift_matrix_subpixel_torch
class PostProcessing_STARFlow_Translation(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_STARFlow_Translation, self).__init__()
        self.flag_RGB2BW = False
        self.warp_object = Warp_Object()
        self.affine_warp_object = Warp_Tensors_Affine_Layer()

    def adjust_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        return id, od, td

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### the model outputs optical flow - average everything out to get Translation: ###
        od.optical_flow_estimates_list = od.model_output
        od.final_optical_flow_estimate = od.model_output[-1]
        od.translation_estimate = od.final_optical_flow_estimate.mean(-1,True).mean(-2,True).squeeze()

        ### Assign Network Output Tensor: ###
        od.model_output = od.translation_estimate

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##

class PostProcessing_STARFlow_OpticalFlow(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_STARFlow_OpticalFlow, self).__init__()
        self.flag_RGB2BW = False

    def adjust_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        return id, od, td

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### the model outputs optical flow - average everything out to get Translation: ###
        od.optical_flow_estimate = od.model_output[-1]

        ### Assign Network Output Tensor: ###
        od.model_output = od.optical_flow_estimate

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##
####################################################################################################################################################


####################################################################################################################################################
class PreProcessing_FastFlowNet(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_FastFlowNet, self).__init__()
        self.flag_BW2RGB = False

    def prepare_inputs_for_net(self, id, td, model):
        ### Prepare Input: ###
        network_input = id.output_frames_noisy  # Get relevant frame
        if self.flag_BW2RGB:
            network_input = BW2RGB_interleave_T(network_input)  # the model accepts 3 channels but i deal with monochrome channels!!!!
        elif self.flag_RGB2BW:
            network_input = RGB2BW_interleave_T(network_input)

        ### Normalize Inputs: ###
        if td.current_time_step == 0:
            network_input, id, td = self.normalize_inputs(network_input, id, td)

        ### The Network Accepts A Weird Combination Of Parameters: ###
        network_input = (network_input[:,0], network_input[:,1])  #second input is the one warped inside the network
        id.left_image = network_input[0]
        id.right_image = network_input[1]
        return network_input, id, td, model

from RapidBase.Utils.Warping_Shifting import shift_matrix_subpixel_torch
class PostProcessing_FastFlowNet_Translation(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_FastFlowNet_Translation, self).__init__()
        self.flag_RGB2BW = False
        self.warp_object = Warp_Object()
        self.affine_warp_object = Warp_Tensors_Affine_Layer()

    def adjust_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        return id, od, td

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### the model outputs optical flow - average everything out to get Translation: ###
        od.optical_flow_estimates_list = od.model_output
        od.final_optical_flow_estimate = od.model_output[-1]
        od.translation_estimate = od.final_optical_flow_estimate.mean(-1,True).mean(-2,True).squeeze()

        ### Assign Network Output Tensor: ###
        od.model_output = od.translation_estimate

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##

class PostProcessing_FastFlowNet_OpticalFlow(PostProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PostProcessing_FastFlowNet_OpticalFlow, self).__init__()
        self.flag_RGB2BW = False

    def adjust_loss_parameters(self, inputs_dict, outputs_dict, Train_dict):
        ### Dicts Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        return id, od, td

    def forward(self, inputs_dict, outputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        od = outputs_dict
        td = Train_dict

        ### the model outputs optical flow - average everything out to get Translation: ###
        #TODO: the model outputs more then one optical flow, it outputs the entire pyramid, take care of that.
        od.optical_flow_estimate = od.model_output[0]

        ### Assign Network Output Tensor: ###
        od.model_output = od.optical_flow_estimate

        ### Adjust Some Parameters Before Loss: ###
        id, od, td = self.adjust_loss_parameters(id, od, td)

        return od.model_output, od, id, td, model  #TODO: currently the first argument here IS NOT USED in trainer.py!!@Q$@$$##
####################################################################################################################################################


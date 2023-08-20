from RapidBase.Models.Unets import Unet_large
from RapidBase.import_all import *

from torch.utils.data import DataLoader

from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.training_utils import path_make_path_if_none_exists
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

from RDND_proper.scenarios.Classic_TNR.Noise_Estimate.Noise_Estimate_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *



################################################################################################################################
class ClassicIIR_AlphaBlend_NoOpticalFlow_NoClassicalPrior(nn.Module):
    def __init__(self, model_dict=None, in_channels=None, out_channels=None):
        super(ClassicIIR_AlphaBlend_NoOpticalFlow_NoClassicalPrior, self).__init__()
        self.registration_algorithm = model_dict.registration_algorithm
        self.noise_estimation_algorithm = model_dict.noise_estimation_algorithm
        self.downsample_kernel_size_FrameCombine = model_dict.downsample_kernel_size_FrameCombine
        self.downsample_kernel_size_NoiseEstimation = model_dict.downsample_kernel_size_NoiseEstimation
        self.downsample_factor_registration = model_dict.downsample_factor_registration
        self.max_search_area = model_dict.max_search_area
        self.flag_do_initial_SAD = model_dict.flag_do_initial_SAD
        self.inner_crop_size = model_dict.inner_crop_size
        self.model_dict = model_dict
        self.in_channels = 3  #specific per network according to the number of inputs defined in forward
        self.out_channels = 1

        ### Initialize Model Internal Tracking: ###
        self.clean_frame_previous = None
        self.noise_map_previous = None
        self.pixel_counts_map_previous = None
        self.clean_frame_previous_warped = None
        self.shifts_x_sum = 0
        self.shifts_y_sum = 0

        ### Get Internal Network: ###
        self.network = self.get_network(self.in_channels, self.out_channels)

    def get_network(self, in_channels, out_channels):
        network = Unet_large(in_channels, out_channels)
        return network

    def forward(self, x):
        noisy_frame_current_cropped = x
        noisy_frame_current_cropped = noisy_frame_current_cropped.squeeze(2) #[B,T,C,H,W] -> [B,C,H,W]

        ### in the pre-processing function i'm populating self.clean_frame_previous and more with the proper things...but that's not right! do it in the model
        if self.clean_frame_previous is None:
            self.clean_frame_previous = noisy_frame_current_cropped  # initialize internal "cleaned" image to simply be first frame, which is noisy
            self.pixel_counts_map_previous = torch.ones_like(self.clean_frame_previous)
            self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_previous)
            self.noise_map_previous = None

        ### Forward: ###
        shifts_array = [0,0]
        self.clean_frame_previous_warped = self.clean_frame_previous + 0

        ### Use Network To Adjust Reset Gates And More Maybe: ###
        # (1). Prepare Input For Network:
        network_input = noisy_frame_current_cropped
        network_input = torch.cat([network_input, self.clean_frame_previous], 1)
        network_input = torch.cat([network_input, self.clean_frame_previous_warped], 1)
        # (2). Pass Input Through Network:
        network_output = self.network(network_input)
        # (4). Change Results According To Network Outputs:
        alpha_blend = network_output[:, 0:1, :, :]
        self.clean_frame_current = self.clean_frame_previous_warped * alpha_blend + noisy_frame_current_cropped * (1 - alpha_blend)

        ### Keep variables for next frame: ###
        self.clean_frame_previous = self.clean_frame_current
        self.shifts_x_sum += shifts_array[0]
        self.shifts_y_sum += shifts_array[1]

        ### Hachana Lemazgan - Take Care Of Variables Which Can Be Used In More Advanced Models: ###
        self.Reset_Gates_Combine = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.reset_gates_classic = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_current = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_previous = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_current = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_previous = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)

        return self.clean_frame_current #what i actually return is the clean frame output
    

    
class ClassicIIR_ResetGate_NoOpticalFlow_NoClassicalPrior(nn.Module):
    def __init__(self, model_dict=None, in_channels=None, out_channels=None):
        super(ClassicIIR_ResetGate_NoOpticalFlow_NoClassicalPrior, self).__init__()
        self.registration_algorithm = model_dict.registration_algorithm
        self.noise_estimation_algorithm = model_dict.noise_estimation_algorithm
        self.downsample_kernel_size_FrameCombine = model_dict.downsample_kernel_size_FrameCombine
        self.downsample_kernel_size_NoiseEstimation = model_dict.downsample_kernel_size_NoiseEstimation
        self.downsample_factor_registration = model_dict.downsample_factor_registration
        self.max_search_area = model_dict.max_search_area
        self.flag_do_initial_SAD = model_dict.flag_do_initial_SAD
        self.inner_crop_size = model_dict.inner_crop_size
        self.model_dict = model_dict
        self.in_channels = 3
        self.out_channels = 1

        ### Initialize Model Internal Tracking: ###
        self.clean_frame_previous = None
        self.noise_map_previous = None
        self.pixel_counts_map_previous = None
        self.clean_frame_previous_warped = None
        self.shifts_x_sum = 0
        self.shifts_y_sum = 0

        ### Get Internal Network: ###
        self.network = self.get_network(self.in_channels, self.out_channels)

    def get_network(self, in_channels, out_channels):
        network = Unet_large(in_channels, out_channels)
        return network

    def forward(self, x):
        noisy_frame_current_cropped = x
        noisy_frame_current_cropped = noisy_frame_current_cropped.squeeze(2)  # [B,T,C,H,W] -> [B,C,H,W]

        ### in the pre-processing function i'm populating self.clean_frame_previous and more with the proper things...but that's not right! do it in the model
        if self.clean_frame_previous is None:
            self.clean_frame_previous = noisy_frame_current_cropped  # initialize internal "cleaned" image to simply be first frame, which is noisy
            self.pixel_counts_map_previous = torch.ones_like(self.clean_frame_previous)
            self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_previous)
            self.noise_map_previous = None

        ### Forward: ###
        shifts_array = [0,0]
        self.clean_frame_previous_warped = self.clean_frame_previous + 0

        ### Use Network To Adjust Reset Gates And More Maybe: ###
        # (1). Prepare Input For Network:
        network_input = noisy_frame_current_cropped
        network_input = torch.cat([network_input, self.clean_frame_previous], 1)
        network_input = torch.cat([network_input, self.clean_frame_previous_warped], 1)
        # (2). Pass Input Through Network:
        network_output = self.network(network_input)
        # (4). Change Results According To Network Outputs:
        reset_gate = network_output[:, 0:1, :, :]
        self.pixel_counts_map_current = self.pixel_counts_map_previous * reset_gate + 1
        alpha_blend = 1 - 1/self.pixel_counts_map_current
        self.clean_frame_current = self.clean_frame_previous_warped * alpha_blend + noisy_frame_current_cropped * (1 - alpha_blend)

        ### Keep variables for next frame: ###
        self.clean_frame_previous = self.clean_frame_current
        self.pixel_counts_map_previous = self.pixel_counts_map_current
        self.shifts_x_sum += shifts_array[0]
        self.shifts_y_sum += shifts_array[1]

        ### Hachana Lemazgan - Take Care Of Variables Which Can Be Used In More Advanced Models: ###
        self.Reset_Gates_Combine = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.reset_gates_classic = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_current = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_previous = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_current).to(
            self.clean_frame_current.device)

        return self.clean_frame_current #what i actually return is the clean frame output
################################################################################################################################################


################################################################################################################################
class ClassicIIR_AlphaBlend_WithOpticalFlow_NoClassicalPrior(nn.Module):
    def __init__(self, model_dict=None, in_channels=None, out_channels=None):
        super(ClassicIIR_AlphaBlend_WithOpticalFlow_NoClassicalPrior, self).__init__()
        self.registration_algorithm = model_dict.registration_algorithm
        self.noise_estimation_algorithm = model_dict.noise_estimation_algorithm
        self.downsample_kernel_size_FrameCombine = model_dict.downsample_kernel_size_FrameCombine
        self.downsample_kernel_size_NoiseEstimation = model_dict.downsample_kernel_size_NoiseEstimation
        self.downsample_factor_registration = model_dict.downsample_factor_registration
        self.max_search_area = model_dict.max_search_area
        self.flag_do_initial_SAD = model_dict.flag_do_initial_SAD
        self.inner_crop_size = model_dict.inner_crop_size
        self.model_dict = model_dict
        self.in_channels = 3
        self.out_channels = 1

        ### Initialize Model Internal Tracking: ###
        self.clean_frame_previous = None
        self.noise_map_previous = None
        self.pixel_counts_map_previous = None
        self.clean_frame_previous_warped = None
        self.shifts_x_sum = 0
        self.shifts_y_sum = 0
        self.X = None
        self.warp_object = Warp_Object()

        ### Get Internal Network: ###
        self.network = self.get_network(self.in_channels, self.out_channels)

        ### OpticalFlow Networks: ###
        Train_dict = EasyDict()
        Train_dict.models_folder = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints\Model_New_Checkpoints")
        Train_dict.Network_checkpoint_step = 0
        # # (1). STARFlow:
        # Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = True
        # Train_dict.load_Network_filename = r'/home/mafat/PycharmProjects/IMOD/models/star_flow/saved_checkpoint/StarFlow_things/checkpoint_best.ckpt'
        # args = EasyDict()
        # div_flow = 0.05
        # STARFlow_Net = StarFlow(args)
        # STARFlow_Net, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=STARFlow_Net)
        # self.STARFlow_Net = STARFlow_Net.cuda()
        # (2). FastFlowNet:
        Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = False
        Train_dict.load_Network_filename = r'/home/mafat/PycharmProjects/IMOD/models/FastFlowNet/checkpoints/fastflownet_things3d.pth'
        FastFlowNet_model = FastFlowNet()
        FastFlowNet_model, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=FastFlowNet_model)
        self.FastFlowNet_model = FastFlowNet_model.cuda()
        # # (3). RAFT:
        # Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = True
        # Train_dict.load_Network_filename = '/home/mafat/PycharmProjects/IMOD/models/RAFT/Checkpoints/raft-small.pth'
        # args = EasyDict()
        # args.dropout = 0
        # args.alternate_corr = False
        # args.small = True
        # args.mixed_precision = False
        # args.number_of_iterations = 20
        # self.RAFT_model = RAFT_dudy(args).cuda()

    def get_network(self, in_channels, out_channels):
        network = Unet_large(in_channels, out_channels)
        return network

    def forward(self, x):
        noisy_frame_current_cropped = x
        noisy_frame_current_cropped = noisy_frame_current_cropped.squeeze(2)  # [B,T,C,H,W] -> [B,C,H,W]

        ### in the pre-processing function i'm populating self.clean_frame_previous and more with the proper things...but that's not right! do it in the model
        if self.clean_frame_previous is None:
            self.clean_frame_previous = noisy_frame_current_cropped  # initialize internal "cleaned" image to simply be first frame, which is noisy
            self.pixel_counts_map_previous = torch.ones_like(self.clean_frame_previous)
            self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_previous)
            self.noise_map_previous = None

        ### Forward: ###
        shifts_array = [0, 0]

        ### Optical Flow Network & Warping: ###
        ### Register Images Using Optical Flow NN: ###
        # self.clean_frame_previous_warped, self.flow, self.occlusions, self.global_shifts, self.NN_output_dict = self.register_images(noisy_frame_current_cropped)
        # (1). FastFlowNet:
        list_of_images = []
        list_of_images.append(BW2RGB(self.clean_frame_previous / 1))
        list_of_images.append(BW2RGB(noisy_frame_current_cropped / 1))
        output_flow_pyramid = self.FastFlowNet_model.forward(list_of_images)
        output_flow = F.interpolate(output_flow_pyramid[0], scale_factor=4)
        output_flow_magnitude = torch.sqrt(output_flow[:, 0:1, :, :] ** 2 + output_flow[:, 1:2, :, :, ] ** 2)
        output_flow_magnitude = output_flow_magnitude * 20 * 4
        delta_x = output_flow[:, 1:2, :, :]
        delta_y = output_flow[:, 0:1, :, :]
        # TODO: build warp object which accepts mapping and not delta
        self.clean_frame_previous_warped = self.warp_object.forward(self.clean_frame_previous, delta_x, delta_y, flag_bicubic_or_bilinear='bilinear')

        ### Use Network To Adjust Reset Gates And More Maybe: ###
        # (1). Prepare Input For Network:
        network_input = noisy_frame_current_cropped
        network_input = torch.cat([network_input, self.clean_frame_previous], 1)
        network_input = torch.cat([network_input, self.clean_frame_previous_warped], 1)
        # (2). Pass Input Through Network:
        network_output = self.network(network_input)
        # (4). Change Results According To Network Outputs:
        alpha_blend = network_output[:, 0:1, :, :]
        self.clean_frame_current = self.clean_frame_previous_warped * alpha_blend + \
                                   noisy_frame_current_cropped * (1 - alpha_blend)

        ### Keep variables for next frame: ###
        self.clean_frame_previous = self.clean_frame_current
        self.shifts_x_sum += shifts_array[0]
        self.shifts_y_sum += shifts_array[1]

        ### Hachana Lemazgan - Take Care Of Variables Which Can Be Used In More Advanced Models: ###
        self.Reset_Gates_Combine = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.reset_gates_classic = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_current = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_previous = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_current = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_previous = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)

        return self.clean_frame_current  # what i actually return is the clean frame output


class ClassicIIR_ResetGate_WithOpticalFlow_NoClassicalPrior(ClassicIIR_AlphaBlend_WithOpticalFlow_NoClassicalPrior):
    def __init__(self, model_dict=None, in_channels=5, out_channels=5):
        super(ClassicIIR_ResetGate_WithOpticalFlow_NoClassicalPrior, self).__init__(model_dict=None, in_channels=5, out_channels=5)

        ### Get Internal Network: ###
        self.network = self.get_network(self.in_channels, self.out_channels)

    def get_network(self, in_channels, out_channels):
        network = Unet_large(in_channels, out_channels)
        return network

    def forward(self, x):
        noisy_frame_current_cropped = x
        noisy_frame_current_cropped = noisy_frame_current_cropped.squeeze(2)  # [B,T,C,H,W] -> [B,C,H,W]

        ### in the pre-processing function i'm populating self.clean_frame_previous and more with the proper things...but that's not right! do it in the model
        if self.clean_frame_previous is None:
            self.clean_frame_previous = noisy_frame_current_cropped  # initialize internal "cleaned" image to simply be first frame, which is noisy
            self.pixel_counts_map_previous = torch.ones_like(self.clean_frame_previous)
            self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_previous)
            self.noise_map_previous = None

        ### Forward: ###
        shifts_array = [0, 0]
        ### Optical Flow Network & Warping: ###
        ### Register Images Using Optical Flow NN: ###
        # self.clean_frame_previous_warped, self.flow, self.occlusions, self.global_shifts, self.NN_output_dict = self.register_images(noisy_frame_current_cropped)
        # (1). FastFlowNet:
        list_of_images = []
        list_of_images.append(BW2RGB(self.clean_frame_previous / 1))
        list_of_images.append(BW2RGB(noisy_frame_current_cropped / 1))
        output_flow_pyramid = self.FastFlowNet_model.forward(list_of_images)
        output_flow = F.interpolate(output_flow_pyramid[0], scale_factor=4)
        output_flow_magnitude = torch.sqrt(output_flow[:, 0:1, :, :] ** 2 + output_flow[:, 1:2, :, :, ] ** 2)
        output_flow_magnitude = output_flow_magnitude * 20 * 4
        delta_x = output_flow[:, 1:2, :, :]
        delta_y = output_flow[:, 0:1, :, :]
        self.clean_frame_previous_warped = self.warp_object.forward(self.clean_frame_previous, delta_x, delta_y, flag_bicubic_or_bilinear='bilinear')

        ### Use Network To Adjust Reset Gates And More Maybe: ###
        # (1). Prepare Input For Network:
        network_input = noisy_frame_current_cropped
        network_input = torch.cat([network_input, self.clean_frame_previous], 1)
        network_input = torch.cat([network_input, self.clean_frame_previous_warped], 1)
        # (2). Pass Input Through Network:
        network_output = self.network(network_input)
        # (4). Change Results According To Network Outputs:
        reset_gate = network_output[:, 0:1, :, :]
        self.pixel_counts_map_current = self.pixel_counts_map_previous * reset_gate + 1
        alpha_blend = 1 - 1 / self.pixel_counts_map_current
        self.clean_frame_current = self.clean_frame_previous_warped * alpha_blend + \
                                   noisy_frame_current_cropped * (1 - alpha_blend)

        ### Keep variables for next frame: ###
        self.clean_frame_previous = self.clean_frame_current
        self.shifts_x_sum += shifts_array[0]
        self.shifts_y_sum += shifts_array[1]

        ### Hachana Lemazgan - Take Care Of Variables Which Can Be Used In More Advanced Models: ###
        self.Reset_Gates_Combine = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.reset_gates_classic = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_current = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_previous = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)

        return self.clean_frame_current  # what i actually return is the clean frame output
################################################################################################################################################


################################################################################################################################
class ClassicIIR_AlphaBlend_SimpleNetworkNoOpticalFlow_NoClassicalPrior(nn.Module):
    def __init__(self, model_dict=None, in_channels=None, out_channels=None):
        super(ClassicIIR_AlphaBlend_SimpleNetworkNoOpticalFlow_NoClassicalPrior, self).__init__()
        self.registration_algorithm = model_dict.registration_algorithm
        self.noise_estimation_algorithm = model_dict.noise_estimation_algorithm
        self.downsample_kernel_size_FrameCombine = model_dict.downsample_kernel_size_FrameCombine
        self.downsample_kernel_size_NoiseEstimation = model_dict.downsample_kernel_size_NoiseEstimation
        self.downsample_factor_registration = model_dict.downsample_factor_registration
        self.max_search_area = model_dict.max_search_area
        self.flag_do_initial_SAD = model_dict.flag_do_initial_SAD
        self.inner_crop_size = model_dict.inner_crop_size
        self.model_dict = model_dict
        self.in_channels = 3
        self.out_channels = 1

        ### Initialize Model Internal Tracking: ###
        self.clean_frame_previous = None
        self.noise_map_previous = None
        self.pixel_counts_map_previous = None
        self.clean_frame_previous_warped = None
        self.shifts_x_sum = 0
        self.shifts_y_sum = 0

        ### Get Internal Network: ###
        self.input_network = self.get_network(2, 1)
        self.gate_network = self.get_network(3, 1)

    def get_network(self, in_channels, out_channels):
        network = Unet_large(in_channels, out_channels)
        return network

    def forward(self, x):
        noisy_frame_current_cropped = x
        noisy_frame_current_cropped = noisy_frame_current_cropped.squeeze(2)  # [B,T,C,H,W] -> [B,C,H,W]

        ### in the pre-processing function i'm populating self.clean_frame_previous and more with the proper things...but that's not right! do it in the model
        if self.clean_frame_previous is None:
            self.clean_frame_previous = noisy_frame_current_cropped  # initialize internal "cleaned" image to simply be first frame, which is noisy
            self.pixel_counts_map_previous = torch.ones_like(self.clean_frame_previous)
            self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_previous)
            self.noise_map_previous = None

        ### Forward: ###
        shifts_array = [0, 0]
        ### Pass input through network to get output to be blended with previous_clean: ###
        network_input = torch.cat([noisy_frame_current_cropped,self.clean_frame_previous], 1)  # TODO: maybe change to previous noisy or something
        self.clean_frame_previous_warped = self.input_network(network_input)

        ### Use Network To Adjust Reset Gates And More Maybe: ###
        # (1). Prepare Input For Network:
        network_input_gate = noisy_frame_current_cropped
        network_input_gate = torch.cat([network_input_gate, self.clean_frame_previous], 1)
        network_input_gate = torch.cat([network_input_gate, self.clean_frame_previous_warped], 1)
        # (2). Pass Input Through Network:
        network_output = self.gate_network(network_input_gate)
        # (4). Change Results According To Network Outputs:
        alpha_blend = network_output[:, 0:1, :, :]
        self.clean_frame_current = self.clean_frame_previous_warped * alpha_blend + \
                                   noisy_frame_current_cropped * (1 - alpha_blend)

        ### Keep variables for next frame: ###
        self.clean_frame_previous = self.clean_frame_current
        self.shifts_x_sum += shifts_array[0]
        self.shifts_y_sum += shifts_array[1]

        ### Hachana Lemazgan - Take Care Of Variables Which Can Be Used In More Advanced Models: ###
        self.Reset_Gates_Combine = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.reset_gates_classic = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_current = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_previous = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_current = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_previous = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)

        return self.clean_frame_current  # what i actually return is the clean frame output


class ClassicIIR_ResetGate_SimpleNetworkNoOpticalFlow_NoClassicalPrior(ClassicIIR_AlphaBlend_WithOpticalFlow_NoClassicalPrior):
    def __init__(self, model_dict=None, in_channels=None, out_channels=None):
        super(ClassicIIR_ResetGate_SimpleNetworkNoOpticalFlow_NoClassicalPrior, self).__init__(model_dict=None, in_channels=None,
                                                                                    out_channels=None)

        ### Get Internal Network: ###
        self.input_network = self.get_network(2, 1)
        self.gate_network = self.get_network(3, 1)

    def get_network(self, in_channels, out_channels):
        network = Unet_large(in_channels, out_channels)
        return network

    def forward(self, x):
        noisy_frame_current_cropped = x
        noisy_frame_current_cropped = noisy_frame_current_cropped.squeeze(2)  # [B,T,C,H,W] -> [B,C,H,W]

        ### in the pre-processing function i'm populating self.clean_frame_previous and more with the proper things...but that's not right! do it in the model
        if self.clean_frame_previous is None:
            self.clean_frame_previous = noisy_frame_current_cropped  # initialize internal "cleaned" image to simply be first frame, which is noisy
            self.pixel_counts_map_previous = torch.ones_like(self.clean_frame_previous)
            self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_previous)
            self.noise_map_previous = None

        ### Forward: ###
        shifts_array = [0, 0]
        ### Pass input through network to get output to be blended with previous_clean: ###
        network_input = torch.cat([noisy_frame_current_cropped, self.clean_frame_previous])  #TODO: maybe change to previous noisy or something
        self.clean_frame_previous_warped = self.input_network(network_input)

        ### Use Network To Adjust Reset Gates And More Maybe: ###
        # (1). Prepare Input For Network:
        network_input = noisy_frame_current_cropped
        network_input = torch.cat([network_input, self.clean_frame_previous], 1)
        network_input = torch.cat([network_input, self.clean_frame_previous_warped], 1)
        # (2). Pass Input Through Network:
        network_output = self.gate_network(network_input)
        # (4). Change Results According To Network Outputs:
        reset_gate = network_output[:, 0:1, :, :]
        self.pixel_counts_map_current = self.pixel_counts_map_previous * reset_gate
        alpha_blend = 1 - 1 / self.pixel_counts_map_current
        self.clean_frame_current = self.clean_frame_previous_warped * alpha_blend + \
                                   noisy_frame_current_cropped * (1 - alpha_blend)

        ### Keep variables for next frame: ###
        self.clean_frame_previous = self.clean_frame_current
        self.shifts_x_sum += shifts_array[0]
        self.shifts_y_sum += shifts_array[1]

        ### Hachana Lemazgan - Take Care Of Variables Which Can Be Used In More Advanced Models: ###
        self.Reset_Gates_Combine = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.reset_gates_classic = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_current = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.noise_map_previous = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.pixel_counts_map_previous_warped = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)

        return self.clean_frame_current  # what i actually return is the clean frame output
################################################################################################################################################


################################################################################################################################################
class Classic_IIR_alpha_blend(nn.Module):
    def __init__(self, model_dict=None, in_channels=5, out_channels=5):
        super(Classic_IIR_alpha_blend, self).__init__()
        self.registration_algorithm = model_dict.registration_algorithm
        self.noise_estimation_algorithm = model_dict.noise_estimation_algorithm
        self.downsample_kernel_size_FrameCombine = model_dict.downsample_kernel_size_FrameCombine
        self.downsample_kernel_size_NoiseEstimation = model_dict.downsample_kernel_size_NoiseEstimation
        self.downsample_factor_registration = model_dict.downsample_factor_registration
        self.max_search_area = model_dict.max_search_area
        self.flag_do_initial_SAD = model_dict.flag_do_initial_SAD
        self.inner_crop_size = model_dict.inner_crop_size
        self.model_dict = model_dict
        self.in_channels = in_channels
        self.out_channels = out_channels

        ### Initialize Model Internal Tracking: ###
        self.clean_frame_previous = None
        self.noise_map_previous = None
        self.pixel_counts_map_previous = None
        self.clean_frame_previous_warped = None
        self.shifts_x_sum = 0
        self.shifts_y_sum = 0

        ### Get Internal Network: ###
        self.network = self.get_network(self.in_channels, self.out_channels)

    def get_network(self, in_channels, out_channels):
        network = Unet_large(in_channels, out_channels)
        return network

    def forward(self, x):
        noisy_frame_current_cropped = x
        if self.clean_frame_previous is None:
            self.clean_frame_previous = noisy_frame_current_cropped
        else:
            # ### Register Images (return numpy arrays): ###
            # # registration_algorithm = 'CrossCorrelation' #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR
            # shifts_array, self.clean_frame_previous_warped = register_images(noisy_frame_current_cropped.data,
            #                                                                  self.clean_frame_previous.data,
            #                                                                  self.model_dict.registration_algorithm,
            #                                                                  self.model_dict.max_search_area,
            #                                                                  self.model_dict.inner_crop_size,
            #                                                                  self.model_dict.downsample_factor_registration,
            #                                                                  flag_do_initial_SAD=self.model_dict.flag_do_initial_SAD,
            #                                                                  initial_SAD_search_area=self.model_dict.max_search_area)
            ### TODO: TEMP - will incorporate image registration in a second: ###
            #TODO: use flag to decide whether or not to register
            shifts_array = [0,0]
            self.clean_frame_previous_warped = self.clean_frame_previous + 0


            ### Use Network To Adjust Reset Gates And More Maybe: ###
            # (1). Prepare Input For Network:
            network_input = noisy_frame_current_cropped
            network_input = torch.cat([network_input, self.clean_frame_previous], 1)
            network_input = torch.cat([network_input, self.clean_frame_previous_warped], 1)
            # (2). Pass Input Through Network:
            network_output = self.network(network_input)
            # (4). Change Results According To Network Outputs:
            alpha_blend = network_output[:, 0:1, :, :]
            self.clean_frame_current = self.clean_frame_previous_warped * alpha_blend + noisy_frame_current_cropped * (1 - alpha_blend)

            ### Keep variables for next frame: ###
            self.clean_frame_previous = self.clean_frame_current
            self.shifts_x_sum += shifts_array[0]
            self.shifts_y_sum += shifts_array[1]

        return self.clean_frame_current #what i actually return is the clean frame output


class Classic_IIR_1(nn.Module):
    def __init__(self, model_dict=None, in_channels=5, out_channels=5):
        super(Classic_IIR_1, self).__init__()
        self.registration_algorithm = model_dict.registration_algorithm
        self.noise_estimation_algorithm = model_dict.noise_estimation_algorithm
        self.downsample_kernel_size_FrameCombine = model_dict.downsample_kernel_size_FrameCombine
        self.downsample_kernel_size_NoiseEstimation = model_dict.downsample_kernel_size_NoiseEstimation
        self.downsample_factor_registration = model_dict.downsample_factor_registration
        self.max_search_area = model_dict.max_search_area
        self.flag_do_initial_SAD = model_dict.flag_do_initial_SAD
        self.inner_crop_size = model_dict.inner_crop_size
        self.model_dict = model_dict
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.noise_sigma = model_dict.sigma_to_model

        ### Initialize Model Internal Tracking: ###
        self.clean_frame_previous = None
        self.noise_map_previous = None
        self.pixel_counts_map_previous = None
        self.clean_frame_previous_warped = None
        self.shifts_x_sum = 0
        self.shifts_y_sum = 0

        ### Get Internal Network: ###
        self.network = self.get_network(self.in_channels, self.out_channels)

        ### Warp Object: ###
        self.warp_object = Warp_Object()
        self.X = None
        self.Y = None

    def get_network(self, in_channels, out_channels):
        return Unet_large(in_channels, out_channels)

    def forward(self, x):
        noisy_frame_current_cropped = x

        # ### Register Images (return numpy arrays): ###
        # # registration_algorithm = 'CrossCorrelation' #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR
        shifts_array, self.clean_frame_previous_warped_classic, flow = register_images(noisy_frame_current_cropped.data,
                                                                    self.clean_frame_previous.data,
                                                                    self.model_dict.registration_algorithm,
                                                                    self.model_dict.max_search_area,
                                                                    self.model_dict.inner_crop_size,
                                                                    self.model_dict.downsample_factor_registration,
                                                                    flag_do_initial_SAD=self.model_dict.flag_do_initial_SAD,
                                                                    initial_SAD_search_area=self.model_dict.max_search_area)
        # ### TODO: TEMP - will incorporate image registration in a second: ###
        # shifts_array = [0, 0]
        # self.clean_frame_previous_warped = self.clean_frame_previous + 0

        ### Inv-Flow Image: ###
        if self.X is None:
            B,C,H,W = noisy_frame_current_cropped.shape
            h_vec = np.arange(H)
            w_vec = np.arange(W)
            X, Y = np.meshgrid(w_vec, h_vec)
            self.X = X
            self.Y = Y
        delta_x = flow[:, :, 0] - self.X
        delta_y = flow[:, :, 1] - self.Y
        delta_x = torch.Tensor(delta_x).unsqueeze(0).unsqueeze(0).cuda()
        delta_y = torch.Tensor(delta_y).unsqueeze(0).unsqueeze(0).cuda()
        self.clean_frame_previous_warped = self.warp_object.forward(self.clean_frame_previous.cuda(), delta_x, delta_y)

        # ### Estimate noise: ###
        # self.noise_map_current, self.noise_pixel_counts_map_current = estimate_noise_IterativeResidual(
        #     noisy_frame_current_cropped.data,
        #     self.noise_map_previous,
        #     self.pixel_counts_map_previous,
        #     self.clean_frame_previous_warped.data,
        #     downsample_kernel_size=self.downsample_kernel_size_NoiseEstimation)  # original_frame_current_cropped
        ### TODO: TEMP - will incorporate image registration in a second: ###
        #TODO: add flag to toggle estimation and constant
        self.noise_map_current = self.noise_sigma * torch.ones_like(self.clean_frame_previous.cuda()) / 255 #TODO: there's a problem with the noise sigma in dataset_object and other places

        ### Combine Images: ###
        self.clean_frame_current_classic, self.pixel_counts_map_current_classic, self.reset_gates_classic = combine_frames(noisy_frame_current_cropped,
                                                                                    self.clean_frame_previous_warped.detach(), #TODO: remove .detach()
                                                                                    noise_map_current=self.noise_map_current.detach(),
                                                                                    noise_map_last_warped=None,
                                                                                    pixel_counts_map_previous=self.pixel_counts_map_previous.detach(), #TODO: remove .detach()
                                                                                    downsample_kernel_size=self.downsample_kernel_size_FrameCombine)

        ### Use Network To Adjust Reset Gates And More Maybe: ###
        #(1). Prepare Input For Network:
        network_input = noisy_frame_current_cropped
        network_input = torch.cat([network_input, self.clean_frame_current_classic.cuda()], 1)
        network_input = torch.cat([network_input, self.clean_frame_previous.cuda()], 1)
        network_input = torch.cat([network_input, self.clean_frame_previous_warped.cuda()], 1)
        network_input = torch.cat([network_input, self.noise_map_current.cuda()], 1)
        network_input = torch.cat([network_input, self.reset_gates_classic.cuda()], 1)
        #(2). Pass Input Through Network:
        network_output = self.network(network_input)
        #(3). Get Proper Outputs From Network Output:
        reset_gates_from_network = network_output[:, 0:1, :, :]
        #(4). Change Results According To Network Outputs:
        # self.pixel_counts_map_current = self.pixel_counts_map_previous * reset_gates_from_network + 1
        self.pixel_counts_map_current = self.pixel_counts_map_previous * reset_gates_from_network + 1
        self.clean_frame_current = noisy_frame_current_cropped * (1 / self.pixel_counts_map_current) + \
                                   self.clean_frame_previous_warped * (1 - 1 / self.pixel_counts_map_current)
        self.Reset_Gates_Combine = reset_gates_from_network

        ### Keep variables for next frame: ###
        # self.clean_frame_previous = self.clean_frame_current
        self.clean_frame_previous = self.clean_frame_current_classic
        # self.pixel_counts_map_previous = self.pixel_counts_map_current
        self.pixel_counts_map_previous = self.pixel_counts_map_current_classic
        self.noise_map_previous = self.noise_map_current
        self.shifts_x_sum += shifts_array[0]
        self.shifts_y_sum += shifts_array[1]

        return self.clean_frame_current





class ClassicIIR_Final(nn.Module):
    def __init__(self, model_dict=None, in_channels=5, out_channels=5):
        super(ClassicIIR_Final, self).__init__()
        self.registration_algorithm = model_dict.registration_algorithm
        self.noise_estimation_algorithm = model_dict.noise_estimation_algorithm
        self.downsample_kernel_size_FrameCombine = model_dict.downsample_kernel_size_FrameCombine
        self.downsample_kernel_size_NoiseEstimation = model_dict.downsample_kernel_size_NoiseEstimation
        self.downsample_factor_registration = model_dict.downsample_factor_registration
        self.max_search_area = model_dict.max_search_area
        self.flag_do_initial_SAD = model_dict.flag_do_initial_SAD
        self.inner_crop_size = model_dict.inner_crop_size
        self.model_dict = model_dict
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.noise_sigma = model_dict.sigma_to_model

        ### Initialize Model Internal Tracking: ###
        self.clean_frame_previous = None
        self.noise_map_previous = None
        self.pixel_counts_map_previous = None
        self.clean_frame_previous_warped = None
        self.shifts_x_sum = 0
        self.shifts_y_sum = 0

        ### Get Internal Network: ###
        self.network = self.get_network(self.in_channels, self.out_channels)

        ### Warp Object: ###
        self.warp_object = Warp_Object()
        self.X = None
        self.Y = None

    def get_network(self, in_channels, out_channels):
        return Unet_large(in_channels, out_channels)
    
    def register_images_ClassicalOpticalFlow(self, noisy_frame_current_cropped):
        # # registration_algorithm = 'CrossCorrelation' #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR
        global_shifts_array, clean_frame_previous_warped_classic, flow = register_images(noisy_frame_current_cropped.data,
                                                                                       self.clean_frame_previous.data,
                                                                                       self.model_dict.registration_algorithm,
                                                                                       self.model_dict.max_search_area,
                                                                                       self.model_dict.inner_crop_size,
                                                                                       self.model_dict.downsample_factor_registration,
                                                                                       flag_do_initial_SAD=self.model_dict.flag_do_initial_SAD,
                                                                                       initial_SAD_search_area=self.model_dict.max_search_area)
        return global_shifts_array, clean_frame_previous_warped_classic, flow
    
    def register_images_NeuralOpticalFlow(self, noisy_frame_current_cropped):
        flow = self.OpticalFlow_Network(noisy_frame_current_cropped, self.clean_frame_previous)
        global_shifts_array = flow.mean(-1) #TODO: make sure this works

        ### Inv-Flow Image: ###
        if self.X is None:
            B, C, H, W = noisy_frame_current_cropped.shape
            h_vec = np.arange(H)
            w_vec = np.arange(W)
            X, Y = np.meshgrid(w_vec, h_vec)
            self.X = X
            self.Y = Y
        delta_x = flow[:, :, :, 0] - self.X
        delta_y = flow[:, :, :, 1] - self.Y
        delta_x = torch.Tensor(delta_x).unsqueeze(0).unsqueeze(0)
        delta_y = torch.Tensor(delta_y).unsqueeze(0).unsqueeze(0)
        clean_frame_previous_warped = self.warp_object.forward(self.clean_frame_previous, delta_x, delta_y)
        return global_shifts_array, clean_frame_previous_warped, flow
    
    def register_images_ZeroShift(self, noisy_frame_current_cropped):
        B,C,H,W = noisy_frame_current_cropped.shape
        flow = torch.zeros_like(B, H, W, 2)
        global_shifts_array = [0, 0]
        clean_frame_previous_warped = self.clean_frame_previous + 0
        return global_shifts_array, clean_frame_previous_warped, flow
    
    def register_images(self, noisy_frame_current_cropped):
        if self.flag_register_images_method == 'classical':
            return self.register_images_ClassicalOpticalFlow(noisy_frame_current_cropped)
        elif self.flag_register_images_method == 'neural':
            return self.register_images_NeuralOpticalFlow(noisy_frame_current_cropped)
        elif self.flag_register_images_method == 'zero_shift':
            return self.register_images_ZeroShift(noisy_frame_current_cropped)
    
    def estimate_noise_IterativeResidualAlgorithm(self, noisy_frame_current_cropped):
        noise_map_current, noise_pixel_counts_map_current = estimate_noise_IterativeResidual(
            noisy_frame_current_cropped.data,
            self.noise_map_previous,
            self.pixel_counts_map_previous,
            self.clean_frame_previous_warped.data,
            downsample_kernel_size=self.downsample_kernel_size_NoiseEstimation)  # original_frame_current_cropped
        self.noise_pixel_counts_map_current = noise_pixel_counts_map_current
        return noise_map_current
    
    def estimate_noise_KnownLevel(self, noisy_frame_current_cropped):
        # TODO: there's a problem with the noise sigma in dataset_object and other places
        noise_map_current = self.noise_sigma * torch.ones_like(self.clean_frame_previous.cuda()) / 255
        return noise_map_current
    
    def estimate_noise(self, noisy_frame_current_cropped):
        if self.flag_estimate_noise_method == 'known_level':
            return self.estimate_noise_KnownLevel(noisy_frame_current_cropped)
        elif self.flag_estimate_noise_method == 'classic_iterative_residual':
            return self.estimate_noise_IterativeResidualAlgorithm(noisy_frame_current_cropped)
    
    def combine_frames_Classical(self, noisy_frame_current_cropped):
        self.clean_frame_current_classic, self.pixel_counts_map_current_classic, self.reset_gates_classic = combine_frames(
            noisy_frame_current_cropped,
            self.clean_frame_previous_warped.detach(),  # TODO: remove .detach().... i think it's problematic as far as gradient calculations are concerned
            noise_map_current=self.noise_map_current.detach(),
            noise_map_last_warped=None,
            pixel_counts_map_previous=self.pixel_counts_map_previous.detach(),  # TODO: remove .detach()
            downsample_kernel_size=self.downsample_kernel_size_FrameCombine)
    
    def combine_frames_Neural(self, network_output, noisy_frame_current_cropped):
        if self.flag_estimate_reset_gate_or_alpha_blend == 'reset_gate':
            reset_gates_from_network = network_output[:, 0:1, :, :]
            self.pixel_counts_map_current = self.pixel_counts_map_previous * reset_gates_from_network + 1
            alpha_blend = 1 / self.pixel_counts_map_current
            clean_frame_current = noisy_frame_current_cropped * alpha_blend + self.clean_frame_previous_warped * (1 - alpha_blend)
            self.Reset_Gates_Combine = reset_gates_from_network
        elif self.flag_estimate_reset_gate_or_alpha_blend == 'alpha_blend':
            alpha_blend = network_output[:, 0:1, :, :]
            clean_frame_current = noisy_frame_current_cropped * alpha_blend + self.clean_frame_previous_warped * (1 - alpha_blend)
        return clean_frame_current
    

    def get_GateNetworkPrediction_1(self, noisy_frame_current_cropped):
        # (*). Prepare Input For Network:
        network_input = noisy_frame_current_cropped
        network_input = torch.cat([network_input, self.clean_frame_current_classic.cuda()], 1)
        network_input = torch.cat([network_input, self.clean_frame_previous.cuda()], 1)
        network_input = torch.cat([network_input, self.clean_frame_previous_warped.cuda()], 1)
        network_input = torch.cat([network_input, self.noise_map_current.cuda()], 1)
        network_input = torch.cat([network_input, self.reset_gates_classic.cuda()], 1)
        # (*). Pass Input Through Network:
        network_output = self.network(network_input)
        return network_output
    
    def get_GateNetworkPrediction(self, noisy_frame_current_cropped):
        network_output = self.gate_prediction_network(noisy_frame_current_cropped)  #TODO: either insert network/function from outside or inside
        return network_output
    
    def forward(self, x):
        noisy_frame_current_cropped = x

        ### Register Images (return numpy arrays): ###
        self.global_shifts_array, self.clean_frame_previous_warped, self.flow = self.register_images(noisy_frame_current_cropped)
        
        ### Estimate noise: ###
        self.noise_map_current = self.estimate_noise(noisy_frame_current_cropped)
        
        ### Combine Images In A Classical Way: ###
        if self.flag_get_classical_combine_predictions:
            self.combine_frames_Classical(noisy_frame_current_cropped)

        ### Use Network To Adjust Reset-Gates Or Alpha-Blend And More Maybe: ###
        network_output = self.get_GateNetworkPrediction(noisy_frame_current_cropped)
        
        #(3). Get Proper Outputs From Network Output:
        self.clean_frame_current = self.combine_frames_Neural(network_output, noisy_frame_current_cropped)
        
        ### Keep variables for next frame: ###
        self.clean_frame_previous = self.clean_frame_current
        self.pixel_counts_map_previous = self.pixel_counts_map_current
        self.noise_map_previous = self.noise_map_current
        self.shifts_x_sum += self.global_shifts_array[0]
        self.shifts_y_sum += self.global_shifts_array[1]

        return self.clean_frame_current


from RDND_proper.models.star_flow.models.STAR import *
from RDND_proper.models.FastFlowNet.models.FastFlowNet import *
from RDND_proper.models.RAFT.core.raft import *
import RapidBase.TrainingCore as core
class ClassicIIR_NN_Final(nn.Module):
    def __init__(self, model_dict=None):
        super(ClassicIIR_NN_Final, self).__init__()
        # self.registration_algorithm = model_dict.registration_algorithm
        # self.noise_estimation_algorithm = model_dict.noise_estimation_algorithm
        # self.downsample_kernel_size_FrameCombine = model_dict.downsample_kernel_size_FrameCombine
        # self.downsample_kernel_size_NoiseEstimation = model_dict.downsample_kernel_size_NoiseEstimation
        # self.downsample_factor_registration = model_dict.downsample_factor_registration
        # self.max_search_area = model_dict.max_search_area
        # self.flag_do_initial_SAD = model_dict.flag_do_initial_SAD
        # self.inner_crop_size = model_dict.inner_crop_size
        # self.model_dict = model_dict

        ### Designating Gate Network Output - AlphaBlend Or ResetGate (for per-pixel counter): ###
        self.flag_estimate_reset_gate_or_alpha_blend = 'alpha_blend' #'alpha_blend' , 'reset_gate'

        ### Gate Network: ###
        self.gate_network_input_channels = 2
        self.gate_network_output_channels = 1
        self.gate_network = self.get_network(self.gate_network_input_channels, self.gate_network_output_channels)

        ### OpticalFlow Network: ###
        Train_dict = EasyDict()
        Train_dict.models_folder = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints\Model_New_Checkpoints")
        Train_dict.Network_checkpoint_step = 0
        # # (1). STARFlow:
        # Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = True
        # Train_dict.load_Network_filename = r'/home/mafat/PycharmProjects/IMOD/models/star_flow/saved_checkpoint/StarFlow_things/checkpoint_best.ckpt'
        # args = EasyDict()
        # div_flow = 0.05
        # STARFlow_Net = StarFlow(args)
        # STARFlow_Net, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=STARFlow_Net)
        # self.STARFlow_Net = STARFlow_Net.cuda()
        # (2). FastFlowNet:
        Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = False
        Train_dict.load_Network_filename = r'/home/mafat/PycharmProjects/IMOD/models/FastFlowNet/checkpoints/fastflownet_things3d.pth'
        FastFlowNet_model = FastFlowNet()
        FastFlowNet_model, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=FastFlowNet_model)
        self.FastFlowNet_model = FastFlowNet_model.cuda()
        # # (3). RAFT:
        # Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = True
        # Train_dict.load_Network_filename = '/home/mafat/PycharmProjects/IMOD/models/RAFT/Checkpoints/raft-small.pth'
        # args = EasyDict()
        # args.dropout = 0
        # args.alternate_corr = False
        # args.small = True
        # args.mixed_precision = False
        # args.number_of_iterations = 20
        # self.RAFT_model = RAFT_dudy(args).cuda()

        ### Noise Estimation: ###
        self.flag_estimate_noise_method = 'known_level'
        self.noise_SNR = 30
        self.noise_sigma = 1/np.sqrt(self.noise_SNR) * 1/255

        ### Initialize Model Internal Tracking: ###
        self.clean_frame_previous = None
        self.noise_map_previous = None
        self.pixel_counts_map_previous = None
        self.clean_frame_previous_warped = None
        self.shifts_x_sum = 0
        self.shifts_y_sum = 0

        ### Warp Object: ###
        self.warp_object = Warp_Object()
        self.X = None
        self.Y = None

    def get_network(self, in_channels, out_channels):
        return Unet_large(in_channels, out_channels)

    def register_images_NeuralOpticalFlow(self, noisy_frame_current_cropped):
        flow = self.OpticalFlow_Network(noisy_frame_current_cropped, self.clean_frame_previous)
        global_shifts_array = flow.mean(-1) #TODO: make sure this works

        ### Inv-Flow Image: ###
        if self.X is None:
            B, C, H, W = noisy_frame_current_cropped.shape
            h_vec = np.arange(H)
            w_vec = np.arange(W)
            X, Y = np.meshgrid(w_vec, h_vec)
            self.X = X
            self.Y = Y
        delta_x = flow[:, :, :, 0] - self.X
        delta_y = flow[:, :, :, 1] - self.Y
        delta_x = torch.Tensor(delta_x).unsqueeze(0).unsqueeze(0)
        delta_y = torch.Tensor(delta_y).unsqueeze(0).unsqueeze(0)
        clean_frame_previous_warped = self.warp_object.forward(self.clean_frame_previous, delta_x, delta_y)
        return global_shifts_array, clean_frame_previous_warped, flow

    def estimate_noise_IterativeResidualAlgorithm(self, noisy_frame_current_cropped):
        noise_map_current, noise_pixel_counts_map_current = estimate_noise_IterativeResidual(
            noisy_frame_current_cropped.data,
            self.noise_map_previous,
            self.pixel_counts_map_previous,
            self.clean_frame_previous_warped.data,
            downsample_kernel_size=self.downsample_kernel_size_NoiseEstimation)  # original_frame_current_cropped
        self.noise_pixel_counts_map_current = noise_pixel_counts_map_current
        return noise_map_current

    def estimate_noise_KnownLevel(self, noisy_frame_current_cropped):
        noise_map_current = self.noise_sigma * torch.ones_like(self.clean_frame_previous.cuda())
        return noise_map_current

    def estimate_noise(self, noisy_frame_current_cropped):
        if self.flag_estimate_noise_method == 'known_level':
            return self.estimate_noise_KnownLevel(noisy_frame_current_cropped)
        elif self.flag_estimate_noise_method == 'classic_iterative_residual':
            return self.estimate_noise_IterativeResidualAlgorithm(noisy_frame_current_cropped)

    def combine_frames_Neural(self, network_output, noisy_frame_current_cropped):
        if self.flag_estimate_reset_gate_or_alpha_blend == 'reset_gate':
            reset_gates_from_network = network_output[:, 0:1, :, :]
            self.pixel_counts_map_current = self.pixel_counts_map_previous * reset_gates_from_network + 1
            alpha_blend = 1 / self.pixel_counts_map_current
            clean_frame_current = noisy_frame_current_cropped * alpha_blend + self.clean_frame_previous_warped * (1 - alpha_blend)
            self.Reset_Gates_Combine = reset_gates_from_network
        elif self.flag_estimate_reset_gate_or_alpha_blend == 'alpha_blend':
            alpha_blend = network_output[:, 0:1, :, :]
            clean_frame_current = noisy_frame_current_cropped * alpha_blend + self.clean_frame_previous_warped * (1 - alpha_blend)
        return clean_frame_current

    def forward(self, x):
        noisy_frame_current_cropped = x

        ### Register Images Using Optical Flow NN: ###
        # self.clean_frame_previous_warped, self.flow, self.occlusions, self.global_shifts, self.NN_output_dict = self.register_images(noisy_frame_current_cropped)
        #(1). FastFlowNet:
        list_of_images = []
        list_of_images.append(BW2RGB(self.clean_frame_previous / 1))
        list_of_images.append(BW2RGB(noisy_frame_current_cropped / 1))
        output_flow_pyramid = self.FastFlowNet_model.forward(list_of_images)
        output_flow = F.interpolate(output_flow_pyramid[0], scale_factor=4)
        output_flow_magnitude = torch.sqrt(output_flow[:, 0:1, :, :] ** 2 + output_flow[:, 1:2, :, :, ] ** 2)
        output_flow_magnitude = output_flow_magnitude * 20 * 4
        ### Inv-Flow Image: ###
        if self.X is None:
            B, C, H, W = noisy_frame_current_cropped.shape
            h_vec = np.arange(H)
            w_vec = np.arange(W)
            X, Y = np.meshgrid(w_vec, h_vec)
            self.X = X
            self.Y = Y
        delta_x = output_flow[:, 1:2, :, :] - self.X
        delta_y = output_flow[:, 0:1, :, :] - self.Y
        delta_x = torch.Tensor(delta_x).unsqueeze(0).unsqueeze(0)
        delta_y = torch.Tensor(delta_y).unsqueeze(0).unsqueeze(0)
        #TODO: build warp object which accepts mapping and not delta
        self.clean_frame_previous_warped = self.warp_object.forward(self.clean_frame_previous, delta_x, delta_y, flag_bicubic_or_bilinear='bilinear')

        ### Estimate noise: ###
        self.noise_map_current = self.estimate_noise(noisy_frame_current_cropped)

        ### Use Network To Adjust Reset-Gates Or Alpha-Blend And More Maybe: ###
        gate_output = self.gate_network(torch.cat([self.clean_frame_previous_warped, noisy_frame_current_cropped], 1))
        self.clean_frame_current = self.combine_frames_Neural(gate_output, noisy_frame_current_cropped)

        ### Keep variables for next frame: ###
        self.clean_frame_previous = self.clean_frame_current
        self.pixel_counts_map_previous = self.pixel_counts_map_current
        self.noise_map_previous = self.noise_map_current
        self.shifts_x_sum += self.global_shifts_array[0]
        self.shifts_y_sum += self.global_shifts_array[1]

        self.Reset_Gates_Combine = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)
        self.reset_gates_classic = torch.ones_like(self.clean_frame_current).to(self.clean_frame_current.device)









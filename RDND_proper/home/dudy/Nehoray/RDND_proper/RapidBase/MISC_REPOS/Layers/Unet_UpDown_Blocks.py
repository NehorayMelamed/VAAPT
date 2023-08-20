import RapidBase.Basic_Import_Libs
from RapidBase.Basic_Import_Libs import *

import RapidBase.Utils_Import_Libs
from RapidBase.Utils_Import_Libs import *


# #(5). Layers:
import RapidBase.MISC_REPOS.Layers.Activations
import RapidBase.MISC_REPOS.Layers.Basic_Layers
import RapidBase.MISC_REPOS.Layers.Conv_Blocks
import RapidBase.MISC_REPOS.Layers.DownSample_UpSample
import RapidBase.MISC_REPOS.Layers.Memory_and_RNN
# import RapidBase.MISC_REPOS.Layers.Refinement_Modules
import RapidBase.MISC_REPOS.Layers.Special_Layers
# # import RapidBase.MISC_REPOS.Layers.Unet_UpDown_Blocks
import RapidBase.Utils.Registration.Warp_Layers
import RapidBase.MISC_REPOS.Layers.Wrappers
from RapidBase.MISC_REPOS.Layers.Activations import *
from RapidBase.MISC_REPOS.Layers.Basic_Layers import *
from RapidBase.MISC_REPOS.Layers.Conv_Blocks import *
from RapidBase.MISC_REPOS.Layers.DownSample_UpSample import *
from RapidBase.MISC_REPOS.Layers.Memory_and_RNN import *
# from RapidBase.MISC_REPOS.Layers.Refinement_Modules import *
from RapidBase.MISC_REPOS.Layers.Special_Layers import *
# # from RapidBase.MISC_REPOS.Layers.Unet_UpDown_Blocks import *
from RapidBase.Utils.Registration.Warp_Layers import *
from RapidBase.MISC_REPOS.Layers.Wrappers import *









class UNET_Down_Fusion_General_V1(nn.Module):
    #TODO: maybe add the possiblity of a "memory" which simply holds the previous output ???
    def __init__(self,
                 number_of_input_channels_from_upper_layer,
                 number_of_input_channels_from_cross_connection,
                 number_of_output_channels,
                 flag_use_final_projection_block = False,
                 number_of_output_channels_after_projection_block = 32,
                 flag_use_cross_connection=True,
                 flag_Sequential_or_RDB = 'sequential', #'sequential' / 'rdb'
                 flag_sequential_dense = False,
                 flag_sequential_resnet = False,
                 flag_sequential_concat = True,
                 stack_residual_scale = 1,
                 kernel_sizes=3, strides=1, dilations=1, groups=1,
                 normalization_function='instance_normalization',
                 activation_function='prelu',
                 initialization_method = 'dirac',
                 flag_downsample_strategy='maxpool',  #'maxpool' / 'unshuffle'
                 flag_add_unshuffled_input_to_lower_level=False,
                 ##############################################   --  1-K-1 --(*+)---
                 ### FIRE/Super Block SFT Self-Interaction: ###   ------------|
                 flag_SuperBlock_SFT=False,
                 flag_SuperBlock_SFT_use_outside_conditional=False,
                 flag_SuperBlock_SFT_same_on_all_channels=False,
                 flag_SuperBlock_SFT_base_convs_mix='x',  # 'x', 'y', 'xy'
                 flag_SuperBlock_SFT_SFT_convs_mix='x',  # 'x', 'y', 'xy'
                 flag_SuperBlock_SFT_add_y_to_output=False,
                 flag_SuperBlock_SFT_shift=False,
                 flag_SuperBlock_SFT_scale=False,
                 ### Deformable Convolution: ###
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v3',
                 flag_deformable_convolution_before_or_after_main_convolution='before',  # 'before' / 'after'
                 flag_deformable_convolution_modulation=True,
                 flag_deformable_kernel_size=5,
                 flag_deformable_number_of_deformable_groups=-1,
                 flag_deformable_number_of_channels_in_group1=-1,
                 flag_deformable_same_on_all_channels=True,
                 ### Deformable SFT: ###
                 flag_deformable_SFT_use_outside_conditional=False,
                 flag_deformable_SFT_same_on_all_channels=False,  ### The interactive shift/scale convs by default spits-out x.shape[1] number_of_channels....but maybe to save memory and parameters we want the same interaction on all channels
                 flag_deformable_SFT_base_convs_mix='x',
                 flag_deformable_SFT_SFT_convs_mix='x',
                 flag_deformable_SFT_shift=False,
                 flag_deformable_SFT_scale=False,
                 flag_deformable_SFT_add_y_to_output=False,
                 flag_deformable_for_each_sub_block_or_for_super_block='super_block',  # 'super_block' / 'sub_block'
                 #####################################
                 ### Cell and Super-Cell Types: ######
                 flag_single_cell_block_type='simple',  # 'simple'/ 'standard_residual'/ '131_residual'
                 flag_super_cell_block_type='131_collapse_residual', # 'concat' / '131' / '131_collapse' / 'concat_standard_residual' / '131_residual' / '131_collapse_residual'
                 ):
        super(UNET_Down_Fusion_General_V1, self).__init__()




        # Take Care Of Variables:
        if type(number_of_output_channels)!=list and type(number_of_output_channels)!=tuple:
            number_of_output_channels = [number_of_output_channels]
        number_of_layers = len(number_of_output_channels);
        kernel_sizes = to_list_of_certain_size(kernel_sizes, number_of_layers)
        strides = to_list_of_certain_size(strides, number_of_layers)
        dilations = to_list_of_certain_size(dilations, number_of_layers)
        groups = to_list_of_certain_size(groups, number_of_layers)

        self.flag_add_unshuffled_input_to_lower_level = flag_add_unshuffled_input_to_lower_level
        self.flag_use_cross_connection = flag_use_cross_connection;
        if flag_use_cross_connection==False:
            number_of_input_channels_from_cross_connection = 0;

        #Process Input From Upper Layer:
        if flag_Sequential_or_RDB == 'sequential':
            self.conv1 = Sequential_Conv_Block_General(number_of_input_channels=number_of_input_channels_from_upper_layer + number_of_input_channels_from_cross_connection,
                                               number_of_output_channels=number_of_output_channels,
                                               kernel_sizes=kernel_sizes,
                                               strides=strides,
                                               dilations=dilations,
                                               groups=groups,
                                               padding_type='reflect',
                                               normalization_function=normalization_function,
                                               activation_function=activation_function,
                                               mode='CNA',
                                               initialization_method=initialization_method,
                                               flag_dense=flag_sequential_dense,
                                               flag_resnet=flag_sequential_resnet,
                                               flag_concat=flag_sequential_concat,
                                               stack_residual_scale = stack_residual_scale,
                                               ##############################################   --  1-K-1 --(*+)---
                                               ### FIRE/Super Block SFT Self-Interaction: ###   ------------|
                                               flag_SuperBlock_SFT=flag_SuperBlock_SFT,
                                               flag_SuperBlock_SFT_use_outside_conditional=flag_SuperBlock_SFT_use_outside_conditional,
                                               flag_SuperBlock_SFT_same_on_all_channels=flag_SuperBlock_SFT_same_on_all_channels,
                                               flag_SuperBlock_SFT_base_convs_mix=flag_SuperBlock_SFT_base_convs_mix,  # 'x', 'y', 'xy'
                                               flag_SuperBlock_SFT_SFT_convs_mix=flag_SuperBlock_SFT_SFT_convs_mix,  # 'x', 'y', 'xy'
                                               flag_SuperBlock_SFT_add_y_to_output=flag_SuperBlock_SFT_add_y_to_output,
                                               flag_SuperBlock_SFT_shift=flag_SuperBlock_SFT_shift,
                                               flag_SuperBlock_SFT_scale=flag_SuperBlock_SFT_scale,
                                               ### Deformable Convolution: ###
                                               flag_deformable_convolution=flag_deformable_convolution,
                                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation,
                                               flag_deformable_kernel_size=flag_deformable_kernel_size,
                                               flag_deformable_number_of_deformable_groups=flag_deformable_number_of_deformable_groups,
                                               flag_deformable_number_of_channels_in_group1=flag_deformable_number_of_channels_in_group1,
                                               flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                               ### Deformable SFT: ###
                                               flag_deformable_SFT_use_outside_conditional=flag_deformable_SFT_use_outside_conditional,
                                               flag_deformable_SFT_same_on_all_channels=flag_deformable_SFT_same_on_all_channels,  ### The interactive shift/scale convs by default spits-out x.shape[1] number_of_channels....but maybe to save memory and parameters we want the same interaction on all channels
                                               flag_deformable_SFT_base_convs_mix=flag_deformable_SFT_base_convs_mix,
                                               flag_deformable_SFT_SFT_convs_mix=flag_deformable_SFT_SFT_convs_mix,
                                               flag_deformable_SFT_shift=flag_deformable_SFT_shift,
                                               flag_deformable_SFT_scale=flag_deformable_SFT_scale,
                                               flag_deformable_SFT_add_y_to_output=flag_deformable_SFT_add_y_to_output,
                                               flag_deformable_for_each_sub_block_or_for_super_block=flag_deformable_for_each_sub_block_or_for_super_block,  # 'super_block' / 'sub_block'
                                               ### Cell & Super-Cell Types: ###
                                               flag_single_cell_block_type=flag_single_cell_block_type,
                                               flag_super_cell_block_type=flag_super_cell_block_type,
                                                       )
        elif flag_Sequential_or_RDB == 'rdb':
            self.conv1 = RDB(number_of_input_channels=number_of_input_channels_from_upper_layer + number_of_input_channels_from_cross_connection,
                             number_of_output_channels_for_each_conv_block=number_of_output_channels,
                             number_of_conv_blocks=number_of_layers,
                             kernel_size=kernel_sizes, stride=1, bias=True, padding_type='reflect', normalization_function=None, activation_function='leakyrelu', mode='CNA',
                             final_residual_branch_scale_factor=1/number_of_layers)


        ### Final Projection Block: ###
        if flag_use_final_projection_block:
            self.projection_block = Color_Space_Conversion_Layer(number_of_input_channels=self.conv1.final_number_of_channels,number_of_output_channels=number_of_output_channels_after_projection_block)
            self.conv1 = nn.Sequential(self.conv1, self.projection_block)
            self.final_number_of_channels = number_of_output_channels_after_projection_block
        else:
            self.final_number_of_channels = self.conv1.final_number_of_channels

        # if flag_sequential_concat:
        #     self.final_number_of_channels += number_of_input_channels_from_cross_connection + number_of_input_channels_from_upper_layer

        #DownSample Strategy:
        if flag_downsample_strategy == 'maxpool':
            self.downsample = nn.MaxPool2d(kernel_size=2)
        elif flag_downsample_strategy == 'unshuffle':
            self.downsample = UnshufflePixels(2); #less spatial extent more channels!!!!
        elif flag_downsample_strategy == 'simple_downsample':
            self.downsample = DownSample_Simple(2); #TODO: implement
        elif flag_downsample_strategy == 'DPP_learned':
            self.downsample = DPP_learned();
        elif flag_downsample_strategy == 'DPP':
            self.downsample = DPP();
        elif flag_downsample_strategy == 'avgpool':
            self.downsample = nn.AvgPool2d(kernel_size=2);


    def forward(self, x, outside_connection_input=None, y_cell_conditional=None, y_deformable_conditional=None):
        ### if there is a cross connection (or outside input) -> concat it to input: ###
        if self.flag_use_cross_connection and outside_connection_input is not None:
            output_before_maxpool = torch.cat([outside_connection_input, x], dim=1)
        else:
            output_before_maxpool = x;

        ### Pass Forward through the layers: ###
        # print('output_before_maxpool: ' + str(output_before_maxpool.shape))
        # print('down layer' + str(y_cell_conditional.shape))
        output_before_maxpool = self.conv1(output_before_maxpool, y_cell_conditional, y_deformable_conditional)

        ### Prepare output to lower layer: ###
        output_after_maxpool = self.downsample(output_before_maxpool)
        if self.flag_add_unshuffled_input_to_lower_level:
            unshuffled_input = UnshufflePixels(x,2)
            output_after_maxpool = torch.cat([output_after_maxpool, unshuffled_input], dim=1)

        return output_before_maxpool, output_after_maxpool









class UNET_Down_Memory_V1(nn.Module):
    def __init__(self,
                 memory_unit_name,
                 number_of_input_channels_from_upper_layer,
                 number_of_hidden_states_channels,
                 kernel_sizes=[3,3], strides=[1,1], dilations=[1,1], groups=[1,1],
                 normalization_function='instance_normalization',
                 activation_function='prelu',
                 flag_deformable_convolution=False):
        super(UNET_Down_Memory_V1, self).__init__()

        # Take Care Of Variables:
        if type(number_of_hidden_states_channels)!=list and type(number_of_hidden_states_channels)!=tuple:
            number_of_hidden_states_channels = [number_of_hidden_states_channels]
        number_of_layers = len(number_of_hidden_states_channels);
        kernel_sizes = to_list_of_certain_size(kernel_sizes, number_of_layers);
        strides = to_list_of_certain_size(strides, number_of_layers)
        dilations = to_list_of_certain_size(dilations, number_of_layers)
        groups = to_list_of_certain_size(groups, number_of_layers)
        activation_function = to_list_of_certain_size(activation_function, number_of_layers)
        flag_deformable_convolution = to_list_of_certain_size(flag_deformable_convolution, number_of_layers)

        #Process Input From Upper Layer:
        self.ConvMemory = Get_Memory_Unit(memory_unit_name=memory_unit_name,
                                          input_size = number_of_input_channels_from_upper_layer,
                                          hidden_sizes=number_of_hidden_states_channels,
                                          n_layers=number_of_layers,
                                          kernel_sizes=kernel_sizes,
                                          strides=strides,
                                          dilations=dilations,
                                          groups=groups,
                                          normalization_function=normalization_function,
                                          activation_function = activation_function,
                                          flag_deformable_convolution=flag_deformable_convolution)


        #DownSample Strategy:
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)


    def reset_hidden_states(self):
        self.ConvMemory.reset_hidden_states()

    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.ConvMemory.reset_or_detach_hidden_states(reset_flags_list)

    def hidden_states_to_device(self,device):
        self.ConvMemory.hidden_states_to_device(device);

    def forward(self, input_tensor,reset_flags_list):
        output_before_maxpool = self.ConvMemory(input_tensor,reset_flags_list)
        output_after_maxpool = self.maxpool1(output_before_maxpool)
        return output_before_maxpool, output_after_maxpool














class UNET_Up_General_V1(nn.Module):
    def __init__(self,
                 number_of_lower_level_channels,
                 number_of_lower_level_channels_after_upsample,
                 number_of_cross_connection_channels,
                 number_of_output_channels,
                 flag_use_final_projection_block=True,
                 number_of_output_channels_after_projection_block=32,
                 flag_use_cross_connection=True,
                 flag_Sequential_or_RDB='sequential',  # 'sequential' / 'rdb'
                 flag_sequential_dense=False,
                 flag_sequential_resnet=False,
                 flag_sequential_concat=False,
                 stack_residual_scale=1,
                 kernel_sizes=3, strides=1, dilations=1, groups=1, normalization_function='instance_normalization', activation_function='prelu',
                 flag_upsample_method='bilinear',
                 flag_add_unshuffled_input_to_upper_level=False,
                 initialization_method='xavier',
                 ##############################################   --  1-K-1 --(*+)---
                 ### FIRE/Super Block SFT Self-Interaction: ###   ------------|
                 flag_SuperBlock_SFT=False,
                 flag_SuperBlock_SFT_use_outside_conditional=False,
                 flag_SuperBlock_SFT_same_on_all_channels=False,
                 flag_SuperBlock_SFT_base_convs_mix='x',  # 'x', 'y', 'xy'
                 flag_SuperBlock_SFT_SFT_convs_mix='x',  # 'x', 'y', 'xy'
                 flag_SuperBlock_SFT_add_y_to_output=False,
                 flag_SuperBlock_SFT_shift=False,
                 flag_SuperBlock_SFT_scale=False,
                 ### Deformable Convolution: ###
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v3',
                 flag_deformable_convolution_before_or_after_main_convolution='before',  # 'before' / 'after'
                 flag_deformable_convolution_modulation=True,
                 flag_deformable_kernel_size=5,
                 flag_deformable_number_of_deformable_groups=-1,
                 flag_deformable_number_of_channels_in_group1=-1,
                 flag_deformable_same_on_all_channels=True,
                 ### Deformable SFT: ###
                 flag_deformable_SFT_use_outside_conditional=False,
                 flag_deformable_SFT_same_on_all_channels=False,  ### The interactive shift/scale convs by default spits-out x.shape[1] number_of_channels....but maybe to save memory and parameters we want the same interaction on all channels
                 flag_deformable_SFT_base_convs_mix='x',
                 flag_deformable_SFT_SFT_convs_mix='x',
                 flag_deformable_SFT_shift=False,
                 flag_deformable_SFT_scale=False,
                 flag_deformable_SFT_add_y_to_output=False,
                 flag_deformable_for_each_sub_block_or_for_super_block='super_block',  # 'super_block' / 'sub_block'
                 #####################################
                 flag_single_cell_block_type='simple',  # 'simple'/ 'standard_residual'/ '131_residual'
                 flag_super_cell_block_type='131_collapse_residual', # 'concat' / '131' / '131_collapse' / 'concat_standard_residual' / '131_residual' / '131_collapse_residual'
                 ):
        super(UNET_Up_General_V1, self).__init__()


        #Take Care Of Variables:
        if type(number_of_output_channels)!=list and type(number_of_output_channels)!=tuple:
            number_of_output_channels = [number_of_output_channels]
        number_of_layers = len(number_of_output_channels);
        kernel_sizes = to_list_of_certain_size(kernel_sizes, number_of_layers)
        strides = to_list_of_certain_size(strides, number_of_layers)
        dilations = to_list_of_certain_size(dilations, number_of_layers)
        groups = to_list_of_certain_size(groups, number_of_layers)

        self.number_of_cross_connection_channels = number_of_cross_connection_channels;
        self.flag_use_cross_connection = flag_use_cross_connection;
        if flag_use_cross_connection==False:
            number_of_cross_connection_channels = 0;

        #(*). Upsample Method For Low Layer Input:
        if flag_upsample_method=='deconvolution' or 'transpose' in flag_upsample_method:
            self.up = nn.ConvTranspose2d(number_of_lower_level_channels, number_of_lower_level_channels_after_upsample, kernel_size=2, stride=2)
        elif flag_upsample_method=='bilinear':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        elif 'pytorch_smart_shuffle' in flag_upsample_method:
            self.up = nn.PixelShuffle(upscale_factor=2)
        elif flag_upsample_method == 'my_smart_shuffle':
            self.up = Pixel_Shuffle_Block(number_of_lower_level_channels,
                                          number_of_lower_level_channels_after_upsample,
                                          upscale_factor=2,
                                          kernel_size=3, stride=1, bias=True, padding_type='zero',
                                          normalization_function=None,
                                          activation_function=activation_function)
        elif flag_upsample_method == 'my_smart_shuffle_stacked':
            self.up = Pixel_Shuffle_Block_Stacked(number_of_lower_level_channels, number_of_lower_level_channels_after_upsample,
                                                  upscale_factor=2, number_of_layers = 2,
                                                  kernel_size=3, stride=1, bias=True, padding_type='zero', normalization_function=None, activation_function=activation_function)
        elif flag_upsample_method == 'simple_shuffle':
            self.up = ShufflePixels(2)
        elif flag_upsample_method == 'none':
            self.up = Identity_Layer()


        ### If the upsampling method is using shuffle then the number of channels goes down by 4: ###
        self.flag_lower_level_shuffled = ('pytorch_smart_shuffle' in flag_upsample_method or 'simple_shuffle' in flag_upsample_method)
        if self.flag_lower_level_shuffled:
            number_of_lower_level_channels_after_upsample = int(number_of_lower_level_channels/4);


        #(*). Fusion Strategy (concat -> conv):
        if flag_Sequential_or_RDB == 'sequential':
            self.conv_block_on_fused_inputs = Sequential_Conv_Block_General(
                                              number_of_input_channels=number_of_lower_level_channels_after_upsample + number_of_cross_connection_channels,
                                               number_of_output_channels=number_of_output_channels,
                                               kernel_sizes=kernel_sizes,
                                               strides=strides,
                                               dilations=dilations,
                                               groups=groups,
                                               padding_type='reflect',
                                               normalization_function=normalization_function,
                                               activation_function=activation_function,
                                               mode='CNA',
                                               initialization_method=initialization_method,
                                               flag_dense=flag_sequential_dense,
                                               flag_resnet=flag_sequential_resnet,
                                               flag_concat=flag_sequential_concat,
                                               stack_residual_scale=stack_residual_scale,
                                               ##############################################   --  1-K-1 --(*+)---
                                               ### FIRE/Super Block SFT Self-Interaction: ###   ------------|
                                               flag_SuperBlock_SFT=flag_SuperBlock_SFT,
                                               flag_SuperBlock_SFT_use_outside_conditional=flag_SuperBlock_SFT_use_outside_conditional,
                                               flag_SuperBlock_SFT_same_on_all_channels=flag_SuperBlock_SFT_same_on_all_channels,
                                               flag_SuperBlock_SFT_base_convs_mix=flag_SuperBlock_SFT_base_convs_mix,  # 'x', 'y', 'xy'
                                               flag_SuperBlock_SFT_SFT_convs_mix=flag_SuperBlock_SFT_SFT_convs_mix,  # 'x', 'y', 'xy'
                                               flag_SuperBlock_SFT_add_y_to_output=flag_SuperBlock_SFT_add_y_to_output,
                                               flag_SuperBlock_SFT_shift=flag_SuperBlock_SFT_shift,
                                               flag_SuperBlock_SFT_scale=flag_SuperBlock_SFT_scale,
                                               ### Deformable Convolution: ###
                                               flag_deformable_convolution=flag_deformable_convolution,
                                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation,
                                               flag_deformable_kernel_size=flag_deformable_convolution_modulation,
                                               flag_deformable_number_of_deformable_groups=flag_deformable_number_of_deformable_groups,
                                               flag_deformable_number_of_channels_in_group1=flag_deformable_number_of_channels_in_group1,
                                               flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                               ### Deformable SFT: ###
                                               flag_deformable_SFT_use_outside_conditional=flag_deformable_SFT_use_outside_conditional,
                                               flag_deformable_SFT_same_on_all_channels=flag_deformable_SFT_same_on_all_channels,  ### The interactive shift/scale convs by default spits-out x.shape[1] number_of_channels....but maybe to save memory and parameters we want the same interaction on all channels
                                               flag_deformable_SFT_base_convs_mix=flag_deformable_SFT_base_convs_mix,
                                               flag_deformable_SFT_SFT_convs_mix=flag_deformable_SFT_SFT_convs_mix,
                                               flag_deformable_SFT_shift=flag_deformable_SFT_shift,
                                               flag_deformable_SFT_scale=flag_deformable_SFT_scale,
                                               flag_deformable_SFT_add_y_to_output=flag_deformable_SFT_add_y_to_output,
                                               flag_deformable_for_each_sub_block_or_for_super_block=flag_deformable_for_each_sub_block_or_for_super_block,  # 'super_block' / 'sub_block'
                                               ### Cell & Super-Cell Types: ###
                                               flag_single_cell_block_type=flag_single_cell_block_type,
                                               flag_super_cell_block_type=flag_super_cell_block_type,
                                               )
        elif flag_Sequential_or_RDB == 'rdb':
            self.conv_block_on_fused_inputs = RDB(number_of_input_channels=number_of_input_channels_from_upper_layer + number_of_input_channels_from_cross_connection,
                             number_of_output_channels_for_each_conv_block=number_of_output_channels,
                             number_of_conv_blocks=number_of_layers,
                             kernel_size=kernel_sizes, stride=1, bias=True, padding_type='reflect', normalization_function=None, activation_function='leakyrelu', mode='CNA',
                             final_residual_branch_scale_factor=1/number_of_layers)

        ### Projection Block at the end: ###
        if flag_use_final_projection_block:
            self.projection_block = Color_Space_Conversion_Layer(number_of_input_channels=self.conv_block_on_fused_inputs.final_number_of_channels,
                                                                 number_of_output_channels=number_of_output_channels_after_projection_block)
            self.conv_block_on_fused_inputs = nn.Sequential(self.conv_block_on_fused_inputs, self.projection_block)
            self.final_number_of_channels = number_of_output_channels_after_projection_block
        else:
            self.final_number_of_channels = self.conv_block_on_fused_inputs.final_number_of_channels

        # if flag_sequential_concat:
        #     self.final_number_of_channels += number_of_lower_level_channels_after_upsample + number_of_cross_connection_channels


    def forward(self, input_cross_connection, input_low_layer, y_cell_conditional=None, y_deformable_conditional=None):
        #(1). upsample input coming from lower layer:
        lower_level_upsampled = self.up(input_low_layer)

        if self.flag_use_cross_connection == False:
            ### No Cross Conection!: ###
            return self.conv_block_on_fused_inputs(lower_level_upsampled, y_cell_conditional, y_deformable_conditional)
        else:
            ### With Cross Connection!: ###
            # Correct For cross connection and upsample layer output size discrepency:
            offset = input_cross_connection.size()[2] - lower_level_upsampled.size()[2]  # offset SHOULD BE either 0 or 1 (for instance downsampling from 25 to 12 and the upscaling from 12 to 24)
            padding = [offset, 0, offset, 0]
            lower_level_upsampled_padded = F.pad(lower_level_upsampled, padding, 'reflect')
            # Fuse Information by concatenating and convolving:
            return self.conv_block_on_fused_inputs(torch.cat([input_cross_connection, lower_level_upsampled_padded], 1), y_cell_conditional, y_deformable_conditional)










class UNET_Up_Memory_V1(nn.Module):
    def __init__(self,
                 memory_unit_name,
                 number_of_lower_level_channels,
                 number_of_lower_level_channels_after_upsample,
                 number_of_cross_connection_channels,
                 number_of_hidden_states_channels,
                 flag_use_cross_connection=True,
                 kernel_sizes=3, strides=1, dilations=1, groups=1, normalization_function='instance_normalization', flag_upsample_method='bilinear', flag_deformable_convolution=False,
                 activation_function='none'):
        super(UNET_Up_Memory_V1, self).__init__()

        # Take Care Of Variables:
        if type(kernel_sizes)!=list and type(kernel_sizes)!=tuple:
            kernel_sizes = [kernel_sizes]
        if type(number_of_hidden_states_channels)!=list and type(number_of_hidden_states_channels)!=tuple:
            number_of_hidden_states_channels = [number_of_hidden_states_channels]
        number_of_layers = len(kernel_sizes);
        strides = to_list_of_certain_size(strides, number_of_layers)
        dilations = to_list_of_certain_size(dilations, number_of_layers)
        groups = to_list_of_certain_size(groups, number_of_layers)
        flag_deformable_convolution = to_list_of_certain_size(flag_deformable_convolution,number_of_layers)

        self.number_of_cross_connection_channels = number_of_cross_connection_channels;
        self.flag_use_cross_connection = flag_use_cross_connection

        #(*). Upsample Method For Low Layer Input:
        if flag_upsample_method=='deconvolution' or 'transpose' in flag_upsample_method:
            self.up = nn.ConvTranspose2d(number_of_lower_level_channels, number_of_lower_level_channels_after_upsample, kernel_size=2, stride=2)
        elif flag_upsample_method=='bilinear':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        elif 'pytorch_smart_shuffle' in flag_upsample_method:
            self.up = nn.PixelShuffle(upscale_factor=2)
        elif flag_upsample_method == 'my_smart_shuffle':
            self.up = Pixel_Shuffle_Block(number_of_lower_level_channels,
                                          number_of_lower_level_channels_after_upsample,
                                          upscale_factor=2,
                                          kernel_size=3, stride=1, bias=True, padding_type='zero',
                                          normalization_function=None,
                                          activation_function=activation_function)
        elif flag_upsample_method == 'my_smart_shuffle_stacked':
            self.up = Pixel_Shuffle_Block_Stacked(number_of_lower_level_channels, number_of_lower_level_channels_after_upsample,
                                                  upscale_factor=2, number_of_layers = 2,
                                                  kernel_size=3, stride=1, bias=True, padding_type='zero', normalization_function=None, activation_function=activation_function)
        elif flag_upsample_method == 'simple_shuffle':
            self.up = ShufflePixels(2)

        #(*). Fusion Strategy:
        #number_of_input_channels = number_of_lower_level_channels + number_of_cross_connection_channels
        self.conv_block_on_fused_inputs = Get_Memory_Unit(memory_unit_name = memory_unit_name,
                                                          input_size=number_of_lower_level_channels + number_of_cross_connection_channels,
                                                          hidden_sizes=number_of_hidden_states_channels,
                                                          n_layers=number_of_layers,
                                                          kernel_sizes=kernel_sizes,
                                                          strides=strides,
                                                          dilations=dilations,
                                                          groups=groups,
                                                          normalization_function=normalization_function,
                                                          flag_deformable_convolution=flag_deformable_convolution)


    def reset_hidden_states(self):
        self.conv_block_on_fused_inputs.reset_hidden_states()

    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.conv_block_on_fused_inputs.reset_or_detach_hidden_states(reset_flags_list)


    def hidden_states_to_device(self,device):
        self.conv_block_on_fused_inputs.hidden_states_to_device(device);


    def forward(self, input_cross_connection, input_low_layer,reset_flags_list):
        # (1). upsample input coming from lower layer:
        lower_level_upsampled = self.up(input_low_layer)
        if self.flag_use_cross_connection == False:
            ### No Cross Conection!: ###
            return self.conv_block_on_fused_inputs(lower_level_upsampled_padded)
        else:
            ### With Cross Connection!: ###
            # Correct For cross connection and upsample layer output size discrepency:   TODO: this below assumes a rectangular shape...not necessarily true!!!!!!
            offset = input_cross_connection.size()[2] - lower_level_upsampled.size()[2]  # offset SHOULD BE either 0 or 1 (for instance downsampling from 25 to 12 and the upscaling from 12 to 24)
            padding = [offset, 0, offset, 0]
            lower_level_upsampled_padded = F.pad(lower_level_upsampled, padding, 'reflect')
            # Fuse Information by concatenating and convolving:
            return self.conv_block_on_fused_inputs(torch.cat([input_cross_connection, lower_level_upsampled_padded], 1),reset_flags_list)






















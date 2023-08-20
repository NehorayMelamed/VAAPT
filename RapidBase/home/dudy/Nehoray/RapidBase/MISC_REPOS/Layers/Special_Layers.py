import RapidBase.Basic_Import_Libs
from RapidBase.Basic_Import_Libs import *







##############################################################################################################################################################################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
##############################################################################################################################################################################################################################




############################################################################################################################################################################################################################################################
### PCD Align + TSA Modules: ###
#### EDVR - Alignment, Deformable Convolution, Video Enhancement: ####

### Pyramidial, Cascading Alignment Module with Deformable Convolutions: ###
class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self,
                 number_of_feature_channels=16,
                 flag_deforrmable_convolution_type='v3', #'v1','v3', 'conditional', 'conditional_SFT', 'conditional_SFT2'
                 ):
        super(PCD_Align, self).__init__()
        #######################################
        #### L3: level 3, 1/4 spatial size: ###
        self.L3_offset_conv1 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(number_of_feature_channels, number_of_feature_channels, 3, 1, 1, bias=True)
        self.L3_offset_sequential = nn.Sequential(self.L3_offset_conv1, nn.LeakyReLU(), self.L3_offset_conv2, nn.LeakyReLU())

        ### Deformable: ###
        # self.L3_dcnpack = ConvOffset2d_v3(number_of_feature_channels, kernel_size=3, init_normal_stddev=0.01, flag_same_on_all_channels=False)
        self.L3_dcnpack = ConvOffset2d_v3_Groups_Conditional_SFT(number_of_input_channels=number_of_feature_channels,
                                               number_of_deformable_groups=1,
                                               flag_use_outside_conditional = False, #True-> deformable contingent also on outside "y_tensor", False-> only use input x (CLASSICAL)
                                               flag_base_convs_mix = 'y',
                                               flag_SFT_convs_mix = 'y',
                                               flag_shift = False,
                                               flag_scale = False,
                                               flag_add_y_to_output=False,
                                               kernel_size=5,
                                               init_normal_stddev=0.01,
                                               flag_same_on_all_channels=False,
                                               flag_automatic=True)
        #######################################


        ######################################
        ### L2: level 2, 1/2 spatial size: ###
        self.L2_offset_conv1 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(number_of_feature_channels, number_of_feature_channels, 3, 1, 1, bias=True)
        self.L2_offset_sequential = nn.Sequential(self.L2_offset_conv2, nn.LeakyReLU(), self.L2_offset_conv3, nn.LeakyReLU())
        ### Deformable: ###
        self.L2_dcnpack = ConvOffset2d_v3_Groups_Conditional_SFT(number_of_input_channels=number_of_feature_channels,
                                               number_of_deformable_groups=1,
                                               flag_use_outside_conditional = False, #True-> deformable contingent also on outside "y_tensor", False-> only use input x (CLASSICAL)
                                               flag_base_convs_mix = 'y',
                                               flag_SFT_convs_mix = 'y',
                                               flag_shift = False,
                                               flag_scale = False,
                                               flag_add_y_to_output=False,
                                               kernel_size=5,
                                               init_normal_stddev=0.01,
                                               flag_same_on_all_channels=False,
                                               flag_automatic=True)
        self.L2_concat_features_conv = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for fea
        ######################################


        ##########################################
        ### L1: level 1, original spatial size:###
        self.L1_offset_conv1 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(number_of_feature_channels, number_of_feature_channels, 3, 1, 1, bias=True)
        self.L1_offset_sequential = nn.Sequential(self.L1_offset_conv2, nn.LeakyReLU(), self.L1_offset_conv3, nn.LeakyReLU())
        ### Deformable: ###
        self.L1_dcnpack = ConvOffset2d_v3_Groups_Conditional_SFT(number_of_input_channels=number_of_feature_channels,
                                               number_of_deformable_groups=1,
                                               flag_use_outside_conditional = False, #True-> deformable contingent also on outside "y_tensor", False-> only use input x (CLASSICAL)
                                               flag_base_convs_mix = 'y',
                                               flag_SFT_convs_mix = 'y',
                                               flag_shift = False,
                                               flag_scale = False,
                                               flag_add_y_to_output=False,
                                               kernel_size=5,
                                               init_normal_stddev=0.01,
                                               flag_same_on_all_channels=False,
                                               flag_automatic=True)
        self.L1_concat_features_conv = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for fea
        ##########################################


        ######################
        ### Cascading DCN: ###
        self.cas_offset_conv1 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(number_of_feature_channels, number_of_feature_channels, 3, 1, 1, bias=True)
        ### Deformable: ###
        self.cas_dcnpack = ConvOffset2d_v3_Groups_Conditional_SFT(number_of_input_channels=number_of_feature_channels,
                                               number_of_deformable_groups=1,
                                               flag_use_outside_conditional = False, #True-> deformable contingent also on outside "y_tensor", False-> only use input x (CLASSICAL)
                                               flag_base_convs_mix = 'y',
                                               flag_SFT_convs_mix = 'y',
                                               flag_shift = False,
                                               flag_scale = False,
                                               flag_add_y_to_output=False,
                                               kernel_size=5,
                                               init_normal_stddev=0.01,
                                               flag_same_on_all_channels=False,
                                               flag_automatic=True)
        ######################

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, previous_frame_features, current_frame_features):
        '''align other neighboring frames to the reference frame in the feature level
        previous_frame_features, current_frame_features: [L1, L2, L3], each with [B,C,H,W] features
        '''
        ###########
        ### L3: ###
        #(1). Concat current and previous frame features (Pre-Extracted):
        L3_input = torch.cat([previous_frame_features[2], current_frame_features[2]], dim=1)
        #(2). Get L3 offset:
        #   (2.1). Initial Only L3_input conv to "convert to offset domain":
        L3_offset = self.L3_offset_sequential(L3_input)
        #   (2.2). No Offsets from below to Add/Concat - No Conv:
        1;
        #(3). Warp Previous Frame L3 According To Outside Features (L3_offset):
        L3_features_warped = self.lrelu(self.L3_dcnpack(previous_frame_features_L3, L3_offset)) #why is the leakyrelu afterwards?....for good measure?
        #(4). No Features To Cascade From Below - No Additional Conv:
        1;
        #(5). Upsample/Interpolate L3 offsets & features To Feed to L2: (by the way....why not directly convolve with sinc for almost "perfect" upsample?)
        L3_offset_upsampled = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False) #Upsample offsets
        L3_features_warped_upsampled = F.interpolate(L3_features_warped, scale_factor=2, mode='bilinear', align_corners=False) #Upsample features
        ############


        ###########
        ### L2: ###
        #(1). Concat current and previous frame features (Pre-Extracted):
        L2_input = torch.cat([previous_frame_features[1], current_frame_features[1]], dim=1)
        #(2). Get L2 offset:
        #   (2.1). Initial Only L2_input conv to "convert to offset domain":
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_input))
        #   (2.2). Add/Concat (indeed....why not add?) L3_offset to L2_offset and get final L2_offset estimate:
        L2_offset = torch.cat([L2_offset, L3_offset_upsampled*2], dim=1)
        L2_offset = self.L2_offset_sequential(L2_offset)
        #(3). Warp Previous Frame L2 According To Outside Features (L2_offset):
        L2_features_warped = self.L2_dcnpack(previous_frame_features[1], L2_offset)
        #(4). Conv Cascaded Features So Far:
        L2_features_warped = self.lrelu(self.L2_concat_features_conv(torch.cat([L2_features_warped, L3_features_warped_upsampled], dim=1)))
        #(5). Upsample Offsets & Features For Next Level:
        L2_offset_upsampled = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_features_warped_upsampled = F.interpolate(L2_features_warped, scale_factor=2, mode='bilinear', align_corners=False)
        ############


        ###########
        ### L1: ###
        #(1). Concat current and previous frame features (Pre-Extracted):
        L1_input = torch.cat([previous_frame_features[0], current_frame_features[0]], dim=1)
        #(2). Get L1 offset:
        #   (2.1). Initial Only L1_input conv to "convert to offset domain":
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_input))
        #   (2.2). Add/Concat (indeed....why not add?) L2_offset to L1_offset and get final L1_offset estimate:
        L1_offset = torch.cat([L1_offset, L2_offset_upsampled * 2], dim=1)
        L1_offset = self.L1_offset_sequential(L1_offset)
        #(3). Warp Previous Frame L1 According To Outside Features (L1_offset):
        L1_features_warped = self.L1_dcnpack(previous_frame_features[0], L1_offset)
        #(4). Conv Cascaded Features So Far:
        L1_features_warped = self.L1_concat_features_conv(torch.cat([L1_features_warped, L2_features_warped_upsampled], dim=1))
        #(5). No More Upsampling To Do - Reached Original Scale:
        1;
        ###########


        ############################################################################################
        ### Cascading -> Basically Take Final L1_features_warped and Make Final Warp Prediction: ###
        offset = torch.cat([L1_features_warped, current_frame_features[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_features_warped = self.lrelu(self.cas_dcnpack(L1_features_warped, offset))
        #############################################################################################

        return L1_features





### Pyramidial, Cascading Alignment Module with Deformable Convolutions: ###
class PCD_Extract_and_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self,
                 number_of_input_channels = 3,
                 number_of_feature_channels=16,
                 number_of_output_channels=3,
                 flag_deformable_convolution_type='v3', #'v1','v3', 'conditional', 'conditional_SFT', 'conditional_SFT2'
                 number_of_deformable_groups = 2,
                 ):
        super(PCD_Extract_and_Align, self).__init__()
        #TODO: as of now number_of_feature_channels is the same and pervasive throughout the module

        ################################
        ### Features Pre-Extraction: ###
        ################################
        self.L1_features_extractor = Sequential_Conv_Block_General(
                                               number_of_input_channels=number_of_input_channels,
                                               number_of_output_channels=[number_of_feature_channels,number_of_feature_channels,number_of_feature_channels],
                                               kernel_sizes=[[3, 3, 3], [3, 3, 3], [3]],
                                               strides=1,
                                               dilations=[[1, 2, 4], [1, 2, 4], [1]],
                                               groups=1,
                                               padding_type='reflect',
                                               normalization_function='none',
                                               activation_function='leakyrelu',
                                               mode='CNA',
                                               initialization_method=['xavier', 'xavier', 'xavier'],
                                               flag_dense=False,
                                               flag_resnet=True,
                                               flag_concat=False,
                                               stack_residual_scale = 1, ##############################################   --  1-K-1 --(*+)---
            ### FIRE/Super Block SFT Self-Interaction: ###   ------------|
            flag_SuperBlock_SFT=[False, False, False], flag_SuperBlock_SFT_use_outside_conditional=[True, True, True], flag_SuperBlock_SFT_same_on_all_channels=[False, False, False], flag_SuperBlock_SFT_base_convs_mix=['y', 'y', 'y'],  # 'x', 'y', 'xy'
            flag_SuperBlock_SFT_SFT_convs_mix=['y', 'y', 'y'],  # 'x', 'y', 'xy'
            flag_SuperBlock_SFT_add_y_to_output=[False, False, False], flag_SuperBlock_SFT_shift=[True, True, True], flag_SuperBlock_SFT_scale=[True, True, True], ### Deformable Convolution: ###
            flag_deformable_convolution=[False, False, False], flag_deformable_convolution_version=['v3', 'v3', 'v3'], flag_deformable_convolution_before_or_after_main_convolution=['before', 'before', 'before'],  # 'before' / 'after'
            flag_deformable_convolution_modulation=[False, False, False], flag_deformable_kernel_size=[5, 5, 5], flag_deformable_number_of_deformable_groups=[-1, -1, -1], flag_deformable_number_of_channels_in_group1=[-1, -1, -1], flag_deformable_same_on_all_channels=[True, True, True], ### Deformable SFT: ###
            flag_deformable_SFT_use_outside_conditional=[False, False, False], flag_deformable_SFT_same_on_all_channels=[False, False, False],  ### The interactive shift/scale convs by default spits-out x.shape[1] number_of_channels....but maybe to save memory and parameters we want the same interaction on all channels
            flag_deformable_SFT_base_convs_mix=['y', 'y', 'y'], flag_deformable_SFT_SFT_convs_mix=['y', 'y', 'y'], flag_deformable_SFT_shift=[False, False, False], flag_deformable_SFT_scale=[False, False, False], flag_deformable_SFT_add_y_to_output=[False, False, False], flag_deformable_for_each_sub_block_or_for_super_block=['super_block', 'super_block', 'super_block'],  # 'super_block' / 'sub_block'
            #####################################
            flag_single_cell_block_type=['simple', 'simple', 'simple'], flag_super_cell_block_type=['131_collapse', '131_collapse', '131_collapse'],
                                                       )







        ### TODO: currently my Input_Output_Sum doesn't account for RESOLUTION CHANGES!!!!.... i should correct it to be able to handle it....as of now i can't use it with strided convolution!#@$#
        self.AvgPooling = nn.AvgPool2d(2)
        self.L2_features_extractor = Sequential_Conv_Block_General(number_of_input_channels=number_of_feature_channels,
                                                                       number_of_output_channels=[number_of_feature_channels, number_of_feature_channels, number_of_feature_channels],
                                                                       kernel_sizes=[[3, 3], [3, 3], [3]],
                                                                       strides=[[1,1],[1,1],[1]],   #Initialy stride=2 to effectively downsample
                                                                       dilations=[[1, 2], [1, 2], [1]],
                                                                       groups=1,
                                                                       padding_type='reflect',
                                                                       normalization_function='none',
                                                                       activation_function='leakyrelu',
                                                                       mode='CNA',
                                                                       initialization_method=['xavier', 'xavier', 'xavier'],
                                                                       flag_dense=False,
                                                                       flag_resnet=True,
                                                                       flag_concat=False,
                                                                       stack_residual_scale=1, ##############################################   --  1-K-1 --(*+)---
                                                                   ### FIRE/Super Block SFT Self-Interaction: ###   ------------|
                                                                   flag_SuperBlock_SFT=[False, False, False], flag_SuperBlock_SFT_use_outside_conditional=[True, True, True], flag_SuperBlock_SFT_same_on_all_channels=[False, False, False], flag_SuperBlock_SFT_base_convs_mix=['y', 'y', 'y'],  # 'x', 'y', 'xy'
                                                                   flag_SuperBlock_SFT_SFT_convs_mix=['y', 'y', 'y'],  # 'x', 'y', 'xy'
                                                                   flag_SuperBlock_SFT_add_y_to_output=[False, False, False], flag_SuperBlock_SFT_shift=[True, True, True], flag_SuperBlock_SFT_scale=[True, True, True], ### Deformable Convolution: ###
                                                                   flag_deformable_convolution=[False, False, False], flag_deformable_convolution_version=['v3', 'v3', 'v3'], flag_deformable_convolution_before_or_after_main_convolution=['before', 'before', 'before'],  # 'before' / 'after'
                                                                   flag_deformable_convolution_modulation=[False, False, False], flag_deformable_kernel_size=[5, 5, 5], flag_deformable_number_of_deformable_groups=[-1, -1, -1], flag_deformable_number_of_channels_in_group1=[-1, -1, -1], flag_deformable_same_on_all_channels=[True, True, True], ### Deformable SFT: ###
                                                                   flag_deformable_SFT_use_outside_conditional=[False, False, False], flag_deformable_SFT_same_on_all_channels=[False, False, False],  ### The interactive shift/scale convs by default spits-out x.shape[1] number_of_channels....but maybe to save memory and parameters we want the same interaction on all channels
                                                                   flag_deformable_SFT_base_convs_mix=['y', 'y', 'y'], flag_deformable_SFT_SFT_convs_mix=['y', 'y', 'y'], flag_deformable_SFT_shift=[False, False, False], flag_deformable_SFT_scale=[False, False, False], flag_deformable_SFT_add_y_to_output=[False, False, False], flag_deformable_for_each_sub_block_or_for_super_block=['super_block', 'super_block', 'super_block'],  # 'super_block' / 'sub_block'
                                                                   #####################################
                                                                   flag_single_cell_block_type=['simple', 'simple', 'simple'], flag_super_cell_block_type=['131_collapse', '131_collapse', '131_collapse'],)
        self.L3_features_extractor = Sequential_Conv_Block_General(number_of_input_channels=number_of_feature_channels,
                                                                   number_of_output_channels=[number_of_feature_channels, number_of_feature_channels, number_of_feature_channels],
                                                                   kernel_sizes=[[3, 3], [3, 3], [3]],
                                                                   strides=[[1,1], [1, 1], [1]],  # Initialy stride=2 to effectively downsample
                                                                   dilations=[[1, 2], [1, 2], [1]],
                                                                   groups=1,
                                                                   padding_type='reflect',
                                                                   normalization_function='none',
                                                                   activation_function='leakyrelu',
                                                                   mode='CNA',
                                                                   initialization_method=['xavier', 'xavier', 'xavier'],
                                                                   flag_dense=False,
                                                                   flag_resnet=True,
                                                                   flag_concat=False,
                                                                   stack_residual_scale=1, ##############################################   --  1-K-1 --(*+)---
                                                                   ### FIRE/Super Block SFT Self-Interaction: ###   ------------|
                                                                   flag_SuperBlock_SFT=[False, False, False], flag_SuperBlock_SFT_use_outside_conditional=[True, True, True], flag_SuperBlock_SFT_same_on_all_channels=[False, False, False], flag_SuperBlock_SFT_base_convs_mix=['y', 'y', 'y'],  # 'x', 'y', 'xy'
                                                                   flag_SuperBlock_SFT_SFT_convs_mix=['y', 'y', 'y'],  # 'x', 'y', 'xy'
                                                                   flag_SuperBlock_SFT_add_y_to_output=[False, False, False], flag_SuperBlock_SFT_shift=[True, True, True], flag_SuperBlock_SFT_scale=[True, True, True], ### Deformable Convolution: ###
                                                                   flag_deformable_convolution=[False, False, False], flag_deformable_convolution_version=['v3', 'v3', 'v3'], flag_deformable_convolution_before_or_after_main_convolution=['before', 'before', 'before'],  # 'before' / 'after'
                                                                   flag_deformable_convolution_modulation=[False, False, False], flag_deformable_kernel_size=[5, 5, 5], flag_deformable_number_of_deformable_groups=[-1, -1, -1], flag_deformable_number_of_channels_in_group1=[-1, -1, -1], flag_deformable_same_on_all_channels=[True, True, True], ### Deformable SFT: ###
                                                                   flag_deformable_SFT_use_outside_conditional=[False, False, False], flag_deformable_SFT_same_on_all_channels=[False, False, False],  ### The interactive shift/scale convs by default spits-out x.shape[1] number_of_channels....but maybe to save memory and parameters we want the same interaction on all channels
                                                                   flag_deformable_SFT_base_convs_mix=['y', 'y', 'y'], flag_deformable_SFT_SFT_convs_mix=['y', 'y', 'y'], flag_deformable_SFT_shift=[False, False, False], flag_deformable_SFT_scale=[False, False, False], flag_deformable_SFT_add_y_to_output=[False, False, False], flag_deformable_for_each_sub_block_or_for_super_block=['super_block', 'super_block', 'super_block'],  # 'super_block' / 'sub_block'
                                                                   #####################################
                                                                   flag_single_cell_block_type=['simple', 'simple', 'simple'], flag_super_cell_block_type=['131_collapse', '131_collapse', '131_collapse'],)




        ##############################################################
        ###  Pyramidial, Cascading & Deformable Alignment Module:  ###
        ##############################################################
        #### L3: level 3, 1/4 spatial size: ###
        self.L3_offset_conv1 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(number_of_feature_channels, number_of_feature_channels, 3, 1, 1, bias=True)
        self.L3_offset_sequential = nn.Sequential(self.L3_offset_conv1, nn.LeakyReLU(), self.L3_offset_conv2, nn.LeakyReLU())
        ### Deformable: ###
        # self.L3_dcnpack = ConvOffset2d_v3(number_of_feature_channels, kernel_size=3, init_normal_stddev=0.01, flag_same_on_all_channels=False)
        self.L3_dcnpack = ConvOffset2d_v3_Groups_Conditional_SFT(number_of_input_channels=number_of_feature_channels,
                                               number_of_deformable_groups=number_of_deformable_groups, #TODO: in the original paper it's 8 and the convolution is defined per kernel weight so it's probably actually more....
                                               flag_use_outside_conditional = False, #True-> deformable contingent also on outside "y_tensor", False-> only use input x (CLASSICAL)
                                               flag_base_convs_mix = 'y',
                                               flag_SFT_convs_mix = 'y',
                                               flag_shift = False,
                                               flag_scale = False,
                                               flag_add_y_to_output=False,
                                               kernel_size=5,
                                               init_normal_stddev=0.01,
                                               flag_same_on_all_channels=False,
                                               flag_automatic=True)
        #######################################


        ######################################
        ### L2: level 2, 1/2 spatial size: ###
        self.L2_offset_conv1 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(number_of_feature_channels, number_of_feature_channels, 3, 1, 1, bias=True)
        self.L2_offset_sequential = nn.Sequential(self.L2_offset_conv2, nn.LeakyReLU(), self.L2_offset_conv3, nn.LeakyReLU())
        ### Deformable: ###
        self.L2_dcnpack = ConvOffset2d_v3_Groups_Conditional_SFT(number_of_input_channels=number_of_feature_channels,
                                               number_of_deformable_groups=number_of_deformable_groups,
                                               flag_use_outside_conditional = False, #True-> deformable contingent also on outside "y_tensor", False-> only use input x (CLASSICAL)
                                               flag_base_convs_mix = 'y',
                                               flag_SFT_convs_mix = 'y',
                                               flag_shift = False,
                                               flag_scale = False,
                                               flag_add_y_to_output=False,
                                               kernel_size=5,
                                               init_normal_stddev=0.01,
                                               flag_same_on_all_channels=False,
                                               flag_automatic=True)
        self.L2_concat_features_conv = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for fea
        ######################################


        ##########################################
        ### L1: level 1, original spatial size:###
        self.L1_offset_conv1 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(number_of_feature_channels, number_of_feature_channels, 3, 1, 1, bias=True)
        self.L1_offset_sequential = nn.Sequential(self.L1_offset_conv2, nn.LeakyReLU(), self.L1_offset_conv3, nn.LeakyReLU())
        ### Deformable: ###
        self.L1_dcnpack = ConvOffset2d_v3_Groups_Conditional_SFT(number_of_input_channels=number_of_feature_channels,
                                               number_of_deformable_groups=number_of_deformable_groups,
                                               flag_use_outside_conditional = False, #True-> deformable contingent also on outside "y_tensor", False-> only use input x (CLASSICAL)
                                               flag_base_convs_mix = 'y',
                                               flag_SFT_convs_mix = 'y',
                                               flag_shift = False,
                                               flag_scale = False,
                                               flag_add_y_to_output=False,
                                               kernel_size=5,
                                               init_normal_stddev=0.01,
                                               flag_same_on_all_channels=False,
                                               flag_automatic=True)
        self.L1_concat_features_conv = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for fea
        ###########################################################################################################################################


        ######################
        ### Cascading DCN: ###
        ######################
        self.cas_offset_conv1 = nn.Conv2d(number_of_feature_channels * 2, number_of_feature_channels, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(number_of_feature_channels, number_of_feature_channels, 3, 1, 1, bias=True)
        ### Deformable: ###
        self.cas_dcnpack = ConvOffset2d_v3_Groups_Conditional_SFT(number_of_input_channels=number_of_feature_channels,
                                               number_of_deformable_groups=number_of_deformable_groups,
                                               flag_use_outside_conditional = False, #True-> deformable contingent also on outside "y_tensor", False-> only use input x (CLASSICAL)
                                               flag_base_convs_mix = 'y',
                                               flag_SFT_convs_mix = 'y',
                                               flag_shift = False,
                                               flag_scale = False,
                                               flag_add_y_to_output=False,
                                               kernel_size=5,
                                               init_normal_stddev=0.01,
                                               flag_same_on_all_channels=False,
                                               flag_automatic=True)


        #########################################################################
        ### Final Projection Of Features To Predetermined Number Of Channels: ###
        #########################################################################
        self.final_feature_projection = Pad_Conv2d(number_of_feature_channels, number_of_output_channels, kernel_size=1, stride=1, dilation=1, groups=1)


        ### Activation Function: ###
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, previous_frame_features, current_frame_features):
        '''align other neighboring frames to the reference frame in the feature level
        previous_frame_features, current_frame_features: [L1, L2, L3], each with [B,C,H,W] features
        '''
        ### TODO: after making sure everything works replace this "full" code with Pre-extraction and self.pcd_align instead of putting pcd_align logical explicitely here....
        ##############################################################################################################################
        #### Reshape Frame Features: ####
        x_tot = torch.cat([current_frame_features.unsqueeze(1), previous_frame_features.unsqueeze(1)], dim=1)
        B, N, C, H, W = x_tot.size()  # N video frames
        x_frames_to_batches = x_tot.view(-1,C,H,W)
        # x_center = x_tot[:, self.center, :, :, :].contiguous()

        #### Extract Pyramidial Features: ####
        #TODO: right now each pyramid has it's own convolutions....if i'm thinking about optical flow maybe it can be possible to use the same kernels at each pyramid level???
        #(*). L1:
        L1_fea = self.L1_features_extractor(x_frames_to_batches)
        #(*). L2
        L1_fea_downsampled = self.AvgPooling(L1_fea)
        L2_fea = self.L2_features_extractor(L1_fea_downsampled)
        #(*). L3
        L2_fea_downsampled  = self.AvgPooling(L2_fea)
        L3_fea = self.L3_features_extractor(L2_fea_downsampled)

        #### Reshape Frame Features: ####
        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### Put the pyramidial features into a list: ####
        current_frame_pyramid_features_list = [L1_fea[:, 0, :, :, :].clone(),
                                               L2_fea[:, 0, :, :, :].clone(),
                                               L3_fea[:, 0, :, :, :].clone()]
        previous_frame_pyramid_features_list = [L1_fea[:, 1, :, :, :].clone(),
                                                 L2_fea[:, 1, :, :, :].clone(),
                                                 L3_fea[:, 1, :, :, :].clone()]
        ###############################################################################################################################




        # ########################################################################################################################################################################
        # #### PCD-Align: ####
        # ref_fea_l = [L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(), L3_fea[:, self.center, :, :, :].clone()]
        # aligned_fea = []
        # for i in range(N):
        #     nbr_fea_l = [L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(), L3_fea[:, i, :, :, :].clone()]
        #     aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        # aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        # ########################################################################################################################################################################




        ###########
        ### L3: ###
        #(1). Concat current and previous frame features (Pre-Extracted):
        L3_input = torch.cat([previous_frame_pyramid_features_list[2], current_frame_pyramid_features_list[2]], dim=1)
        #(2). Get L3 offset:
        #   (2.1). Initial Only L3_input conv to "convert to offset domain":
        L3_offset = self.L3_offset_sequential(L3_input)
        #   (2.2). No Offsets from below to Add/Concat - No Conv:
        1;
        #(3). Warp Previous Frame L3 According To Outside Features (L3_offset):
        # print(L3_offset.shape)
        # print(previous_frame_pyramid_features_list[2].shape)
        L3_features_warped = self.lrelu(self.L3_dcnpack(previous_frame_pyramid_features_list[2], L3_offset)) #why is the leakyrelu afterwards?....for good measure?
        #(4). No Features To Cascade From Below - No Additional Conv:
        1;
        #(5). Upsample/Interpolate L3 offsets & features To Feed to L2: (by the way....why not directly convolve with sinc for almost "perfect" upsample?)
        L3_offset_upsampled = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False) #Upsample offsets
        L3_features_warped_upsampled = F.interpolate(L3_features_warped, scale_factor=2, mode='bilinear', align_corners=False) #Upsample features
        ############


        ###########
        ### L2: ###
        #(1). Concat current and previous frame features (Pre-Extracted):
        L2_input = torch.cat([previous_frame_pyramid_features_list[1], current_frame_pyramid_features_list[1]], dim=1)
        #(2). Get L2 offset:
        #   (2.1). Initial Only L2_input conv to "convert to offset domain":
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_input))
        #   (2.2). Add/Concat (indeed....why not add?) L3_offset to L2_offset and get final L2_offset estimate:
        L2_offset = torch.cat([L2_offset, L3_offset_upsampled*2], dim=1)
        L2_offset = self.L2_offset_sequential(L2_offset)
        #(3). Warp Previous Frame L2 According To Outside Features (L2_offset):
        L2_features_warped = self.L2_dcnpack(previous_frame_pyramid_features_list[1], L2_offset)
        #(4). Conv Cascaded Features So Far:
        L2_features_warped = self.lrelu(self.L2_concat_features_conv(torch.cat([L2_features_warped, L3_features_warped_upsampled], dim=1)))
        #(5). Upsample Offsets & Features For Next Level:
        L2_offset_upsampled = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_features_warped_upsampled = F.interpolate(L2_features_warped, scale_factor=2, mode='bilinear', align_corners=False)
        ############


        ###########
        ### L1: ###
        #(1). Concat current and previous frame features (Pre-Extracted):
        L1_input = torch.cat([previous_frame_pyramid_features_list[0], current_frame_pyramid_features_list[0]], dim=1)
        #(2). Get L1 offset:
        #   (2.1). Initial Only L1_input conv to "convert to offset domain":
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_input))
        #   (2.2). Add/Concat (indeed....why not add?) L2_offset to L1_offset and get final L1_offset estimate:
        L1_offset = torch.cat([L1_offset, L2_offset_upsampled * 2], dim=1)
        L1_offset = self.L1_offset_sequential(L1_offset)
        #(3). Warp Previous Frame L1 According To Outside Features (L1_offset):
        L1_features_warped = self.L1_dcnpack(previous_frame_pyramid_features_list[0], L1_offset)
        #(4). Conv Cascaded Features So Far:
        L1_features_warped = self.L1_concat_features_conv(torch.cat([L1_features_warped, L2_features_warped_upsampled], dim=1))
        #(5). No More Upsampling To Do - Reached Original Scale:
        1;
        ###########


        ############################################################################################
        ### Cascading -> Basically Take Final L1_features_warped and Make Final Warp Prediction: ###
        offset = torch.cat([L1_features_warped, current_frame_pyramid_features_list[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_features_warped = self.lrelu(self.cas_dcnpack(L1_features_warped, offset))
        #############################################################################################


        ### Project Features Into Wanted Number Of Channels: ###
        L1_features_warped = self.final_feature_projection(L1_features_warped)

        return L1_features_warped





### TSA- temporal spatial attention module: ###
class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=64, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()
        self.center_reference_frame = center

        ### temporal attention (before fusion conv): ###
        self.TemporalAttention1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.TemporalAttention2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        ### fusion conv: using 1x1 to save parameters and computation: ###
        self.feature_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        ### Spatial Attention (after fusion conv): ###
        self.SpatialAttention1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.SpatialAttention2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.SpatialAttention3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.SpatialAttention4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.SpatialAttention5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.SpatialAttention_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.SpatialAttention_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.SpatialAttention_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.SpatialAttention_Add1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.SpatialAttention_Add2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features_aligned):
        ### Get Input Features Shape (TODO: probably change this to fit my IIR framework): ##
        B, N, C, H, W = features_aligned.size()  # N video frames

        #### Feature Space -> Temporal-Attention Space: ###
        #(1). Get features in appropriate shapes:
        reference_features_aligned = features_aligned[:, self.center_reference_frame, :, :, :].clone()  #TODO: why to .clone()?
        frames_features_as_batch_examples = features_aligned.view(-1, C, H, W)
        #(2). Features to "temporal attention space":
        reference_features_TA_space = self.TemporalAttention2(reference_features_aligned)
        features_TA_space = self.TemporalAttention1(frames_features_as_batch_examples).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        ### Get Temporal-Attention/Correlation Maps: ###
        #TODO: maybe simply output the pixelwise_correlation_normalized part instead of multiplying it and reshaping it to [B,N*C,H,W] and continuing the logic....they could maybe prove to be great conditionals
        pixelwise_correlation_list = []
        for i in range(N):
            current_features_TA_space = features_TA_space[:, i, :, :, :]
            current_pixelwise_correlation = torch.sum(current_features_TA_space * reference_features_TA_space, 1).unsqueeze(1)  # B, 1, H, W
            pixelwise_correlation_list.append(current_pixelwise_correlation)
        pixelwise_correlation_normalized = torch.sigmoid(torch.cat(pixelwise_correlation_list, dim=1))  # B, N, H, W
        pixelwise_correlation_normalized = pixelwise_correlation_normalized.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)

        ### Aligned Features Weighted By Correlation-To-Reference Maps: ###
        features_aligned_Temporal_weighted = features_aligned.view(B, -1, H, W) * pixelwise_correlation_normalized

        #### Aligned Features Projection: ###
        features_aligned_Temporal_weighted_projection = self.lrelu(self.feature_fusion(features_aligned_Temporal_weighted))

        ### Spatial Attention: ###
        #(1). L1->L2:
        SpatioTemporal_Attention_L1 = self.lrelu(self.SpatialAttention1(features_aligned_Temporal_weighted)) #TODO: why use features_aligned instead of features_aligned_projection????
        att_max = self.maxpool(SpatioTemporal_Attention_L1)
        att_avg = self.avgpool(SpatioTemporal_Attention_L1)
        SpatioTemporal_Attention = self.lrelu(self.SpatialAttention2(torch.cat([att_max, att_avg], dim=1)))
        #(2). L2->L3:
        SpatioTemporal_Attention_lower_level = self.lrelu(self.SpatialAttention_L1(SpatioTemporal_Attention))
        att_max = self.maxpool(SpatioTemporal_Attention_lower_level)
        att_avg = self.avgpool(SpatioTemporal_Attention_lower_level)
        SpatioTemporal_Attention_lower_level = self.lrelu(self.SpatialAttention_L2(torch.cat([att_max, att_avg], dim=1)))


        ### Up-Cascading: ###
        #(1). L3->L2 Upsample:
        SpatioTemporal_Attention_lower_level = self.lrelu(self.SpatialAttention_L3(SpatioTemporal_Attention_lower_level))
        SpatioTemporal_Attention_lower_level = F.interpolate(SpatioTemporal_Attention_lower_level, scale_factor=2, mode='bilinear', align_corners=False)
        #(2). L3+L2 Cascading:
        SpatioTemporal_Attention = self.lrelu(self.SpatialAttention3(SpatioTemporal_Attention))
        SpatioTemporal_Attention = SpatioTemporal_Attention + SpatioTemporal_Attention_lower_level
        #(3). L2->L1 Upsample:
        SpatioTemporal_Attention = self.lrelu(self.SpatialAttention4(SpatioTemporal_Attention))
        SpatioTemporal_Attention = F.interpolate(SpatioTemporal_Attention, scale_factor=2, mode='bilinear', align_corners=False)
        #(4). L1 Spatial Attention Scale&Add Maps:
        SpatioTemporal_Attention = self.SpatialAttention5(SpatioTemporal_Attention)
        SpatioTemporal_Attention_Shift = self.SpatialAttention_Add2(self.lrelu(self.SpatialAttention_Add1(SpatioTemporal_Attention)))
        SpatioTemporal_Attention_Scale = torch.sigmoid(SpatioTemporal_Attention)

        ### Rescale And Shift Fused Features: ###
        features_aligned_SpatioTemporal_weighted_projection = features_aligned_Temporal_weighted_projection * SpatioTemporal_Attention_Scale * 2 + SpatioTemporal_Attention_Shift
        return features_aligned_SpatioTemporal_weighted_projection
############################################################################################################################################################################################################################################################















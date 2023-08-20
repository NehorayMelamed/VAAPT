


#ESRGAN:
import ESRGAN_dataset
import ESRGAN_Visualizers
import ESRGAN_Optimizers
import ESRGAN_Losses
import ESRGAN_deep_utils
import ESRGAN_utils
import ESRGAN_Models
import ESRGAN_basic_Blocks_and_Layers
import ESRGAN_OPT
from ESRGAN_utils import *
from ESRGAN_deep_utils import *
from ESRGAN_basic_Blocks_and_Layers import *
from ESRGAN_Models import *
from ESRGAN_dataset import *
from ESRGAN_Losses import *
from ESRGAN_Optimizers import *
from ESRGAN_Visualizers import *
from ESRGAN_dataset import *
from ESRGAN_OPT import *

import Memory_Networks
from Memory_Networks import *
import Denoising_Layers_And_Networks
from Denoising_Layers_And_Networks import *

from AnyNet.models.submodules import post_3dconvs,feature_extraction_conv





def preconv2d(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bn=True):
    ### Probably Change to my own conv_block: ###
    if bn:
        return nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False)) ### Why No Bias?????
    else:
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False)) ### Why No Bias?????



def batch_relu_conv3d(in_planes, out_planes, kernel_size=3, stride=1, pad=1, bn3d=True):
    if bn3d:
        return nn.Sequential(
            nn.BatchNorm3d(in_planes),
            nn.ReLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False)) ### Why No Bias?????
    else:
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))   ### Why No Bias?????


### Cost Volume 3D Convs Post-Processing Layers: ###
#TODO: understand how to make this more efficient...maybe depthwise convs or something else....
def post_3dconvs(layers, channels):
    net = [batch_relu_conv3d(1, channels)]
    net += [batch_relu_conv3d(channels, channels) for _ in range(layers)]
    net += [batch_relu_conv3d(channels, 1)]
    return nn.Sequential(*net)




### U-Net Upsampling Block: ###
class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        if is_deconv:
            self.up = nn.Sequential(
                nn.BatchNorm2d(in_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)  #ConvTranspose2d probably isn't the wisest choice (checkerboard)
            )
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            in_size = int(in_size * 1.5)  #the 1.5 factor is because the lower level has twice the number of channels as the upper layer so C_lower_level+C_upper_level=C_lower_level+0.5*C_upper_level=1.5*C_lower_level


        self.conv = nn.Sequential(
            preconv2d(in_size, out_size, 3, 1, 1),
            preconv2d(out_size, out_size, 3, 1, 1),
        )


    def forward(self, input_cross_connection, input_lower_level):
        outputs2 = self.up(input_lower_level)
        B,C,H,W = input_cross_connection.shape
        buttom, right = H%2, W%2  #there's got to be a prettier way
        outputs2 = F.pad(outputs2, (0, -right, 0, -buttom))
        return self.conv(torch.cat([input_cross_connection, outputs2], 1))




### Feature Extractor: ###
class feature_extraction_conv(nn.Module):
    def __init__(self, number_of_FE_channels_base,  nblock=2):
        super(feature_extraction_conv, self).__init__()

        ### Initial Strided_Conv (DownSampling) Block: ###
        self.number_of_FE_channels_base = number_of_FE_channels_base
        downsample_conv = [nn.Conv2d(3,  number_of_FE_channels_base, 3, 1, 1), # 512x256
                           preconv2d(number_of_FE_channels_base, number_of_FE_channels_base, kernel_size=3, stride=2, pad=1)]
        downsample_conv = nn.Sequential(*downsample_conv)
        ### Make DownSampling Sequential Conv Blocks With 2x Channels: ###
        inC = number_of_FE_channels_base
        outC = 2*number_of_FE_channels_base
        block0 = self._make_DownSampling_SequentialConvBlocks(inC, outC, nblock)
        self.block0 = nn.Sequential(downsample_conv, block0)


        ### Make DownSampling Sequential Conv Blocks With Geometrically Increasing Number Of Channels: ###
        number_of_block_channels_base = 2*number_of_FE_channels_base
        self.blocks = []
        for i in range(2):
            self.blocks.append(self._make_DownSampling_SequentialConvBlocks(inC=(2**i)*number_of_block_channels_base, outC=(2**(i+1))*number_of_block_channels_base, nblock=nblock))
        self.blocks = nn.ModuleList(self.blocks)


        ### Make UpSampling Blocks: ###
        self.upblocks = []
        for i in reversed(range(2)):
            self.upblocks.append(unetUp(number_of_block_channels_base*2**(i+1), number_of_block_channels_base*2**i, False))
        self.upblocks = nn.ModuleList(self.upblocks)


        ### Weight-Initialize Layers: ###
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        ### Initial DownSampling Block: ###
        Features_Per_Scale_List = [self.block0(x)]  #however remember that the "finest" scale here is still not in the original resolution.....there are 3 downsamples here!@#%$^% that's a factor of 8 in resolution

        ### Second DownSampling Blocks: ###
        for i in range(2):
            Features_Per_Scale_List.append(self.blocks[i](Features_Per_Scale_List[-1]))

        ### Reverse Outputs Order As Preperation For Upsampling Blocks which go from coarse to fine (LIFO, 0=Coarse, 2=Fine): ###
        Features_Per_Scale_List = list(reversed(Features_Per_Scale_List))

        ### Upsampling Blocks: ###
        for i in range(1,3):
            Features_Per_Scale_List[i] = self.upblocks[i-1](Features_Per_Scale_List[i], Features_Per_Scale_List[i-1])  #We only "Refine" with the UNET structure >1 Feature Scales

        return Features_Per_Scale_List


    ### Basic DownSampling Block: ###
    def _make_DownSampling_SequentialConvBlocks(self, inC, outC, nblock):
        ### MaxPool2d(2) -> Sequential_Basic_Blocks: ###
        #TODO: try strided instead of MaxPool?...or maybe not maxpool every layer?
        model = []
        model.append(nn.MaxPool2d(2, 2))
        for i in range(nblock):
            model.append(preconv2d(inC, outC, 3, 1, 1))
            inC = outC
        return nn.Sequential(*model)













class AnyNet(nn.Module):
    def __init__(self, args):
        super(AnyNet, self).__init__()

        self.number_of_input_channels = args.init_channels
        self.max_disparity_per_scale_list = args.maxdisplist
        self.SPN_number_of_base_channels = args.spn_init_channels
        self.Feature_Extractor_number_of_blocks = args.nblocks
        self.number_of_post_processing_3d_layers = args.layers_3d
        self.number_of_3d_post_processing_channels = args.channels_3d
        self.growth_rate = args.growth_rate
        self.flag_use_SPN = args.with_spn

        ### Feature Extractor: ###
        self.feature_extraction = feature_extraction_conv(self.number_of_input_channels, self.Feature_Extractor_number_of_blocks)


        ### Get SPN Module If Wanted: ###
        if self.flag_use_SPN:
            ### Get CUDA SPN Module: ###
            try:
                from .spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
            except:
                print('Cannot load spn model')
                sys.exit()
            self.spn_layer = GateRecurrent2dnoind(True,False)
            ### Post SPN Refinement Module: ###
            spnC = self.SPN_number_of_base_channels
            self.SPN_refinement_layers_list = [nn.Sequential(
                nn.Conv2d(3, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*3, 3, 1, 1, bias=False),
            )]
            self.SPN_refinement_layers_list += [nn.Conv2d(1,spnC,3,1,1,bias=False)]
            self.SPN_refinement_layers_list += [nn.Conv2d(spnC,1,3,1,1,bias=False)]
            self.SPN_refinement_layers_list = nn.ModuleList(self.SPN_refinement_layers_list)
        else:
            self.SPN_refinement_layers_list = None
        ###


        ### Cost Volume Post-Processing: ###
        self.Cost_volume_post_processing_layers_list = []
        for i in range(3):
            net3d = post_3dconvs(self.number_of_post_processing_3d_layers, self.number_of_3d_post_processing_channels*self.growth_rate[i])
            self.Cost_volume_post_processing_layers_list.append(net3d)
        self.Cost_volume_post_processing_layers_list = nn.ModuleList(self.Cost_volume_post_processing_layers_list)


        ### Weight-Initialize Different Layers: ###
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()




    def forward(self, left, right, feature_extraction_list, cost_volume_list, cost_volume_post_processing_list, softmax_list, disparity_regression_from_cost_volume, disparity_map_upsampling_list, SPN_refinement_list):
        left_image_shape = left.size()
        B,C,H,W = left_image_shape

        # ### TicToc Tracker: ###
        # feature_extraction_list = [[],[],[]]
        # cost_volume_list = [[],[],[]]
        # cost_volume_post_processing_list = [[],[],[]]
        # softmax_list = [[],[],[]]
        # disparity_regression_from_cost_volume = [[],[],[]]
        # disparity_map_upsampling_list = [[],[],[]]
        # SPN_refinement_list = [[],[],[]]

        ### Extract Features From Right & Left: ###
        tic()
        Left_Features_List = self.feature_extraction(left)
        Right_Features_List = self.feature_extraction(right)
        Outputs_per_scale_list = []
        feature_extraction_list.append(get_toc('Feature Extraction'))

        ### Loop Over Features Scales: ###
        for scale_index in range(len(Left_Features_List)):
            ### Build Cost-Volume: ###
            if scale_index == 0:
                tic()
                Cost_Volume = self._build_volume_2d(Left_Features_List[scale_index], Right_Features_List[scale_index], self.max_disparity_per_scale_list[scale_index], stride=1)
                cost_volume_list[scale_index].append(get_toc('x4 scale Cost Volume'))
            else:
                ### Build cost volume again but with features warped according to previous layer: ###
                #(*). Unlike something like Riesz transform the coarsest level "waits" for all other scales and the "fine" layers don't enjoy that high level of non-linearity...maybe we can gain line-buffers, speed and accuracy by changing strategy
                #     to something like shared kernels across scales (maybe focusing on x axis) or something like that, or maybe simple refinement with context from all scales from fine to coarse but with changing "attention" as far as number of kernels
                #     from coarse to fine at each iteration - check with hardware what's better (maybe even use dilated/strided convolution as a substitute for pyramid decomposition with nearest neighbor upsampling when wanting to concatenate with stride=1 maps)
                tic()
                coarser_level_disparity_upsampled = F.upsample(Outputs_per_scale_list[scale_index - 1], (Left_Features_List[scale_index].size(2), Left_Features_List[scale_index].size(3)), mode='bilinear') * (Left_Features_List[scale_index].size(2) / left_image_shape[2])
                Cost_Volume = self._build_volume_2d3(Left_Features_List[scale_index], Right_Features_List[scale_index], self.max_disparity_per_scale_list[scale_index], coarser_level_disparity_upsampled, Disparity_Strides_To_Explore=1)
                cost_volume_list[scale_index].append(get_toc(str(2**(scale_index+2)) + 'scale previous disparity upsampling + Cost Volume'))

            ### Post Process Cost Volume For This Scale! (maybe only refine some scales???): ###
            tic()
            Cost_Volume = torch.unsqueeze(Cost_Volume, 1) #unsqueeze it in to -> [B,1,D,H,W] and the 3D filters will work on [D,H,W]???
            Cost_Volume = self.Cost_volume_post_processing_layers_list[scale_index](Cost_Volume)
            Cost_Volume = Cost_Volume.squeeze(1)
            cost_volume_post_processing_list[scale_index].append(get_toc(str(2 ** (scale_index + 2)) + ' scale Cost Volume Post Processing'))

            ### Regress Disparity From Cost Volume: ###
            if scale_index == 0:
                ### Get normalized "probability" of cost: ###
                tic()
                disparity_softmax_probability = F.softmax(-Cost_Volume, dim=1)
                softmax_list[scale_index].append(get_toc(str(2 ** (scale_index + 2)) + ' scale Softmax(-cost_volume)'))
                ### Get Min Cost Arg Expectation Value: ###
                tic()
                pred_low_res = disparity_regression_from_cost_layer(0, self.max_disparity_per_scale_list[0])(disparity_softmax_probability)
                disparity_regression_from_cost_volume[scale_index].append(get_toc(str(2 ** (scale_index + 2)) + ' scale Disparity Regression from cost volume'))
                ### Scale and Upsample Predicted Disparity: ###
                tic()
                pred_low_res = pred_low_res * H / pred_low_res.size(2)
                disp_upsampled = F.upsample(pred_low_res, (H, W), mode='bilinear')
                disparity_map_upsampling_list[scale_index].append(get_toc(str(2 ** (scale_index + 2)) + ' scale Disparity Map Upsampling'))
                ### Append Current Prediction to output: ###
                Outputs_per_scale_list.append(disp_upsampled)
            else:
                ### Get normalized "probability" of cost: ###
                tic()
                disparity_softmax_probability = F.softmax(-Cost_Volume, dim=1)
                softmax_list[scale_index].append(get_toc(str(2**(scale_index+2)) + ' scale Softmax(-cost_volume)'))
                ### Get Min Cost Arg Expectation Value: ###
                tic()
                pred_low_res = disparity_regression_from_cost_layer(-self.max_disparity_per_scale_list[scale_index]+1, self.max_disparity_per_scale_list[scale_index], stride=1)(disparity_softmax_probability)
                disparity_regression_from_cost_volume[scale_index].append(get_toc(str(2**(scale_index+2)) + ' scale Disparity Regression from cost volume'))
                ### Scale and Upsample Predicted Disparity RESIDUAL: ###
                tic()
                pred_low_res = pred_low_res * H / pred_low_res.size(2)
                disp_upsampled = F.upsample(pred_low_res, (H, W), mode='bilinear')
                disparity_map_upsampling_list[scale_index].append(get_toc(str(2**(scale_index+2)) + ' scale Disparity Map Upsampling'))
                ### Add Residual To Previous Layer Predicted Disparity Map: ###
                current_scale_disprity = disp_upsampled + Outputs_per_scale_list[scale_index-1]
                Outputs_per_scale_list.append(current_scale_disprity)



        ### SPN Refinement At the End: ###
        tic()
        if self.SPN_refinement_layers_list:
            ### Preprocessing left image to get weights (G1,G2,G3)
            spn_out = self.SPN_refinement_layers_list[0](nn.functional.upsample(left, (H//4, W//4), mode='bilinear'))  #first element refines left image.... i think that eventhough we activate functiona.upsample we actually downsample here!!@!#$@
            G1, G2, G3 = spn_out[:,:self.SPN_number_of_base_channels,:,:], spn_out[:,self.SPN_number_of_base_channels:self.SPN_number_of_base_channels*2,:,:], spn_out[:,self.SPN_number_of_base_channels*2:,:,:]  #remember of to slice to equal amounts...maybe .slice?O
            sum_abs = G1.abs() + G2.abs() + G3.abs()
            G1 = torch.div(G1, sum_abs + 1e-8)
            G2 = torch.div(G2, sum_abs + 1e-8)
            G3 = torch.div(G3, sum_abs + 1e-8)
            pred_flow = nn.functional.upsample(Outputs_per_scale_list[-1], (H//4, W//4), mode='bilinear')
            refine_flow = self.spn_layer(self.SPN_refinement_layers_list[1](pred_flow), G1, G2, G3)  #second refines disparity, then it goes through the SPN_layer (CUDA) and then the third element refines that SPN_layer
            refine_flow = self.SPN_refinement_layers_list[2](refine_flow)
            Outputs_per_scale_list.append(nn.functional.upsample(refine_flow, (H , W), mode='bilinear'))  #finally upsample and add that output to the outputs list....TODO: why the upsample?...without SPN we predict lower resolution??.....
        SPN_refinement_list.append(get_toc(str(2 ** (scale_index + 2)) + ' SPN refinement'))

        return Outputs_per_scale_list








    ### Warp Input Image According To Optical-Flow/Disparity: ###
    #(*). can this be done more efficiently because it's only x-axis???
    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """

        ### Build MeshGrid (Redundent!!!...everytime it build the meshgrid again???....it's very expensive!!!: ###
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid)

        ### Substract Disparity Map From Grid To Get Final Mapping Grid: ###
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        ### Scale Grid to between [-1,1]: ###
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        ### Permute Grid [B,2,H,W]->[B,H,W,2]: ###
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output



    ### Build Cost Volume With Lowest Scale Features: ###
    def _build_volume_2d(self, Left_Features, Right_Features, maxdisp, stride=1):
        assert maxdisp % stride == 0  # Assume maxdisp is multiple of stride - OH COOL...i can choose the stride by which i shift the features!!!...
        Cost_Volume = torch.zeros(Left_Features.size()[0], maxdisp//stride, Left_Features.size()[2], Left_Features.size()[3]).cuda()  #that's kind of weird...[cost]=[B,D,H,W] ...what about the C channels? -> okay they are all used to calculate the L1 together...maybe a better way?
        ### Loop over the different shifts/strides and calculate cost: ###
        for i in range(0, maxdisp, stride):
            if i > 0:
                ### What's going on here?...why the alternation with: from [0,current_stride]->|Left_Features[0:current_stride]|, from [current_stride,end]->|Left_Featutres[current_stride:END]-Right_Features[0,END-current_stride]
                ### Okay got it-> you've got to fill the invalid regions with something so you fill it up with the left_Features because it's in the same domain i guess.....maybe i can do something better like reflection?.... ###
                ### TODO: maybe i can somehow learn to guess indices to save time or use something like an IIR or nonlocal network??? ###
                Cost_Volume[:, i//stride, :, :i] = Left_Features[:, :, :, :i].abs().sum(1)
                Cost_Volume[:, i//stride, :, i:] = torch.norm(Left_Features[:, :, :, i:] - Right_Features[:, :, :, :-i], 1, 1)
            else:
                Cost_Volume[:, 0, :, :] = torch.norm(Left_Features[:, :, :, :] - Right_Features[:, :, :, :], 1, 1)

        return Cost_Volume.contiguous()



    ### Refine Upper Levels Cost Volume By Guessing Disparity Residual: ###
    def _build_volume_2d3(self, Left_Features, Right_Features, Max_Disparity_To_Explore, Disparity_Map, Disparity_Strides_To_Explore=1):
        Left_Features_Shape = Left_Features.shape
        B,C,H,W = Left_Features_Shape

        ### Reshape Disparity Map As Base To Cost Volume Computation [B,1,H,W] -> [B,D,1,H,W] -> [B*D,1,H,W]
        Disparity_Map_Volume_C2B = Disparity_Map[:,None,:,:,:].repeat(1, Max_Disparity_To_Explore*2-1, 1, 1, 1).view(-1,1,H, W)

        ### Reshape Features As Base To Cost Volume Computation [B,C,H,W] -> [B,D,C,H,W] -> [B*D,C,H,W] : ###
        Left_Features_Volume_C2B = Left_Features[:, None, :, :, :].repeat(1, Max_Disparity_To_Explore * 2 - 1, 1, 1, 1).view(-1, C, H, W)
        Right_Features_Volume_C2B = Right_Features[:, None, :, :, :].repeat(1, Max_Disparity_To_Explore * 2 - 1, 1, 1, 1).view(-1, C, H, W)

        ### Get Discrete Shifts Tensor As Base To Be Explored When Building Cost Volume: ###
        temp_strides_array = np.tile(np.array(range(-Max_Disparity_To_Explore + 1, Max_Disparity_To_Explore)), B) * Disparity_Strides_To_Explore
        Disparity_Discrete_Shifts_Tensor_Volume = Variable(torch.Tensor(np.reshape(temp_strides_array, [len(temp_strides_array), 1, 1, 1])).cuda(), requires_grad=False)  #Default disparities in cost volume

        ### Get Fractional Disparity Map Volume For Cost Calculation: ###
        Disparity_Map_Volume_C2B = Disparity_Map_Volume_C2B - Disparity_Discrete_Shifts_Tensor_Volume #get the residual disparity from the default ones in the cost volume

        ### Get (fractionally) Shifted Right Features Volume For Cost Calculation: ###
        Right_Features_Volume_C2B_Warped = self.warp(Right_Features_Volume_C2B, Disparity_Map_Volume_C2B)

        ### Calculate Cost Volume By L1 On Shifted Features: ###
        Cost_Volume = torch.norm(Left_Features_Volume_C2B - Right_Features_Volume_C2B_Warped, 1, 1)

        ### Reshape Cost Volume By Putting All Shifts In Channels Index [B*D,C,H,W]->[B,C*D,H,W]: ###
        Cost_Volume = Cost_Volume.view(B, -1, H, W)
        return Cost_Volume.contiguous()




class disparity_regression_from_cost_layer(nn.Module):
    def __init__(self, start, end, stride=1):
        super(disparity_regression_from_cost_layer, self).__init__()
        strides_array = np.array(range(start * stride, end * stride, stride))
        strides_array_reshaped = np.reshape(strides_array, [1,len(strides_array),1,1])
        self.disparity_strides_tensor = Variable(torch.Tensor(strides_array_reshaped).cuda(), requires_grad=False)

    def forward(self, disparity_softmax_probability):
        B,C,H,W = disparity_softmax_probability.shape
        disparity = self.disparity_strides_tensor.repeat(B, 1, H, W)
        out = torch.sum(disparity * disparity_softmax_probability, 1, keepdim=True) #Pseudo-Expectation Value: Sum(x*p(x))=Sum(disparity*P(disparity)),   P(disparity)=Softmax(-cost)
        return out



# ### amount of time spent per second: ~0.013 seconds: ###
# tic()
# FPS = 30
# for i in arange(FPS):
#     bla = disparity_regression_from_cost_layer(1,12,1)
#     bla = disparity_regression_from_cost_layer(1,3,1)
#     bla = disparity_regression_from_cost_layer(1,3,1)
# toc()
#


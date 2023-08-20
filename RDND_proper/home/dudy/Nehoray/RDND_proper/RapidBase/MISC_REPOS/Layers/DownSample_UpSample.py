import RapidBase.Basic_Import_Libs
from RapidBase.Basic_Import_Libs import *

from torch.nn.modules.utils import _pair, _quadruple


# #(5). Layers:
# import RapidBase.MISC_REPOS.Layers.Activations
# import RapidBase.MISC_REPOS.Layers.Basic_Layers
# import RapidBase.MISC_REPOS.Layers.Conv_Blocks
# # import RapidBase.MISC_REPOS.Layers.DownSample_UpSample
# import RapidBase.MISC_REPOS.Layers.Memory_and_RNN
# import RapidBase.MISC_REPOS.Layers.Refinement_Modules
# import RapidBase.MISC_REPOS.Layers.Special_Layers
# import RapidBase.MISC_REPOS.Layers.Unet_UpDown_Blocks
# import RapidBase.MISC_REPOS.Layers.Warp_Layers
# import RapidBase.MISC_REPOS.Layers.Wrappers
from RapidBase.MISC_REPOS.Layers.Activations import *
from RapidBase.MISC_REPOS.Layers.Basic_Layers import *
from RapidBase.MISC_REPOS.Layers.Conv_Blocks import *
# # from RapidBase.MISC_REPOS.Layers.DownSample_UpSample import *
from RapidBase.MISC_REPOS.Layers.Memory_and_RNN import *
# from RapidBase.MISC_REPOS.Layers.Refinement_Modules import *
from RapidBase.MISC_REPOS.Layers.Special_Layers import *
# from RapidBase.MISC_REPOS.Layers.Unet_UpDown_Blocks import *
from RapidBase.Utils.Registration.Warp_Layers import *
from RapidBase.MISC_REPOS.Layers.Wrappers import *



############################################################################################################################################################################################################################################################
### Pooling and Pyramid Decomposition: ###

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

# input_tensor = torch.Tensor(randn(1,3,100,100))
# layer = MedianPool2d(kernel_size=2, stride=2, padding=0, same=True)
# output_tensor = layer(input_tensor)
# print(output_tensor.shape)


class StridedPool2d(nn.Module):
    def __init__(self, stride=1):
        super(StridedPool2d, self).__init__()
        self.stride = stride

    def forward(self, x):
        x_strided = x[:,:,0:-1:self.stride,0:-1:self.stride]
        return x_strided

# ### Use Example: ###
# input_tensor = torch.Tensor(randn(1,3,100,100))
# layer = StridedPool2d(stride=2)
# output_tensor = layer(input_tensor)
# print(output_tensor.shape)

class MinPool2d(nn.Module):
    def __init__(self, kernel_size=1, stride=1):
        super(MinPool2d, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.layer = nn.MaxPool2d(kernel_size, stride=stride)

    def forward(self, x):
        x_strided = -self.layer(-x)
        return x_strided

# ### Use Example: ###
# input_tensor = torch.Tensor(randn(1,3,100,100))
# layer = MinPool2d(2,2)
# output_tensor = layer(input_tensor)
# print(output_tensor.shape)


### Detail preserving pooling: ####
class pospowbias(nn.Module):
    def __init__(self):
        super(pospowbias, self).__init__()
        self.Lambda = nn.Parameter(torch.zeros(1))
        self.Alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.Alpha + x**torch.exp(self.Lambda)  #raising to the power of exp(Lambda) instead of simply to the power of Lambda because we need the power to be positive and putting it in an exp insures that

# class DPP_learned(nn.Module):
#     def __init__(self):
#         super(DPP_learned, self).__init__():
#     """ Median pool (usable as median filter when stride=1) module.
#
#     Args:
#         self.pospowbias = pospowbias()
#
#     def forward(self, I):
#         It   = F.upsample(F.avg_pool2d(I, 2), scale_factor=2, mode='nearest')
#         It = F.pad(It, [0, I.shape[3] - It.shape[3], 0, I.shape[2] - It.shape[2]])
#         x   = ((I-It)**2)+1e-3
#         xn = F.upsample(F.avg_pool2d(x, 2), scale_factor=2, mode='nearest')
#         xn = F.pad(xn, [0, x.shape[3] - xn.shape[3], 0, x.shape[2] - xn.shape[2]])
#         w  = self.pospowbias(x/xn)
#         kp = F.avg_pool2d(w, 2)
#         Iw = F.avg_pool2d(I*w, 2)
#         return Iw/kp


class DPP(nn.Module):
    def __init__(self):
        super(DPP, self).__init__()

    def forward(self, I):
        It   = F.upsample(F.avg_pool2d(I, 2), scale_factor=2, mode='nearest')
        It = F.pad(It,[0,I.shape[3]-It.shape[3],0,I.shape[2]-It.shape[2]])
        x   = ((I-It)**2)+1e-3
        xn = F.upsample(F.avg_pool2d(x, 2), scale_factor=2, mode='nearest')
        xn = F.pad(xn, [0, x.shape[3] - xn.shape[3], 0, x.shape[2] - xn.shape[2]])
        w  = 1 + (x/xn)
        kp = F.avg_pool2d(w, 2)
        Iw = F.avg_pool2d(I*w, 2)
        return Iw/kp



class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x



### Pyramid Decomposition: ###
#TODO: add DPPP
class Pyramid_Decomposition(nn.Module):
    def __init__(self, number_of_scales, flag_downsample_strategy=0):
        super(Pyramid_Decomposition, self).__init__()
        self.downsample_blocks_list = []
        self.number_of_scales = number_of_scales
        for i in arange(number_of_scales):
            if flag_downsample_strategy == 0 or ('avg' in str.lower(flag_downsample_strategy)):
                self.downsample_blocks_list.append(nn.AvgPool2d(2, 2))
            elif flag_downsample_strategy == 1 or ('median' in str.lower(flag_downsample_strategy)):
                self.downsample_blocks_list.append(MedianPool2d(kernel_size=2, stride=2, padding=0, same=False))
            elif flag_downsample_strategy == 2 or ('max' in str.lower(flag_downsample_strategy)):
                self.downsample_blocks_list.append(nn.MaxPool2d(2, 2))
            elif flag_downsample_strategy == 2 or ('stride' in str.lower(flag_downsample_strategy)):
                self.downsample_blocks_list.append(StridedPool2d(2))
            elif flag_downsample_strategy == 2 or ('min' in str.lower(flag_downsample_strategy)):
                self.downsample_blocks_list.append(MinPool2d(2, 2))
            elif flag_downsample_strategy == 2 or ('none' or 'no' in str.lower(flag_downsample_strategy)):
                self.downsample_blocks_list.append(Identity_Layer())

    def forward(self, input_tensor):
        input_tensor_pyramid_list = [input_tensor]
        input_tensor_current_scale = input_tensor
        for i in arange(self.number_of_scales - 1):
            input_tensor_current_scale = self.downsample_blocks_list[i](input_tensor_current_scale)
            input_tensor_pyramid_list.append(input_tensor_current_scale)
        return input_tensor_pyramid_list


# ### Use Example:###
# input_tensor = torch.Tensor(randn(1,3,100,100))
# input_tensor = read_image_default_torch()
# number_of_scales = 3
# layer = Pyramid_Decomposition(number_of_scales=number_of_scales)
# output_tensor_list = layer(input_tensor)
# for i in arange(number_of_scales):
#     figure(i)
#     imshow_torch(output_tensor_list[i][0])
############################################################################################################################################################################################################################################################














############################################################################################################################################################################################################################################################
### Upsample Blocks: ###
def Pixel_Shuffle_Block(number_of_input_channels, number_of_output_channels, upscale_factor=2, kernel_size=3, stride=1,
                        bias=True,
                        padding_type='zero', normalization_function=None, activation_function='leakyrelu'):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """

    # Instead of an explicit Upsample/ConvTranspose layer i do a trick with nn.PixelShuffle and hope for the best in terms of effectively implementing super pixel conv:
    conv_layer = Conv_Block(number_of_input_channels, number_of_output_channels * (upscale_factor ** 2), kernel_size,
                            stride, bias=bias,
                            padding_type=padding_type, normalization_function=None, activation_function=None, initialization_method='dirac')
    pixel_shuffle = ShufflePixels(2)# Notice - now the output is of size (height*upsample_factor, width*upsample_factor)

    # After the Implicit Upsampling By Convolution and nn.PixelShuffle i use Normalization and Activation:
    normalization_layer = Normalization_Function(normalization_function,
                                                 number_of_output_channels) if normalization_function else None
    activation_layer = Activation_Function(activation_function) if activation_function else None
    return Pile_Modules_On_Top_Of_Each_Other(conv_layer, pixel_shuffle, normalization_layer, activation_layer)



def Pixel_Shuffle_Block_Stacked(number_of_input_channels, number_of_output_channels, upscale_factor=2, number_of_layers=2, kernel_size=3, stride=1,
                        bias=True,
                        padding_type='zero', normalization_function=None, activation_function='leakyrelu'):

    # Instead of an explicit Upsample/ConvTranspose layer i do a trick with nn.PixelShuffle and hope for the best in terms of effectively implementing super pixel conv:
    conv_layer = Sequential_Conv_Block(number_of_input_channels,
                          number_of_output_channels=[number_of_output_channels * (upscale_factor ** 2)] * number_of_layers,
                          kernel_sizes=kernel_size, strides=1, dilations=1, groups=1, padding_type='reflect',
                          normalization_function=None, activation_function='leakyrelu', mode='CNA',
                          initialization_method='dirac',
                          flag_resnet=False,
                          flag_concat=False)
    pixel_shuffle = ShufflePixels(2)# Notice - now the output is of size (height*upsample_factor, width*upsample_factor)

    return Pile_Modules_On_Top_Of_Each_Other(conv_layer, pixel_shuffle)



# class Interolation_Layer(nn.Module):
#     # Initialize this with a module
#     def __init__(self, downscale_factor=2):
#         super(UnshufflePixels, self).__init__()
#         self.downscale_factor = downscale_factor;
#
#     # Elementwise sum the output of a submodule to its input
#     def forward(self, x):
#         return UnshufflePixels_Function(x,self.downscale_factor)



class UnshufflePixels(nn.Module):
    # Initialize this with a module
    def __init__(self, downscale_factor=2):
        super(UnshufflePixels, self).__init__()
        self.downscale_factor = downscale_factor;

    # Elementwise sum the output of a submodule to its input
    def forward(self, x):
        return UnshufflePixels_Function(x,self.downscale_factor)


class DownSample_Simple(nn.Module):
    def __init__(self, downscale_factor=2, flag_3D=False):
        super(DownSample_Simple, self).__init__()
        self.downscale_factor = downscale_factor;
        self.flag_3D = flag_3D
    # Elementwise sum the output of a submodule to its input
    def forward(self, x):
        if self.flag_3D:
            return x[:,:,0::self.downscale_factor,0::self.downscale_factor,0::self.downscale_factor]
        else:
            return x[:,:,0::self.downscale_factor,0::self.downscale_factor]


class DownSample_Bilinear(nn.Module):
    def __init__(self, downscale_factor=2, flag_3D=False):
        super(DownSample_Bilinear, self).__init__()
        self.downscale_factor = downscale_factor;
        self.flag_3D = flag_3D
    # Elementwise sum the output of a submodule to its input
    def forward(self, x):
        if self.flag_3D==False:
            return F.interpolate(x, (int(x.shape[2] / self.downscale_factor), int(x.shape[3] / self.downscale_factor)), mode='bilinear')
        else:
            return F.interpolate(x, (int(x.shape[2] / self.downscale_factor), int(x.shape[3] / self.downscale_factor), int(x.shape[4] / self.downscale_factor)), mode='trilinear')


class DownSample_Projection_Strided(nn.Module):
    def __init__(self, output_number_of_channels=3, downscale_factor=2, flag_3D=False):
        super(DownSample_Projection_Strided, self).__init__()
        self.downscale_factor = downscale_factor;
        self.output_number_of_channels = output_number_of_channels
        self.flag_3D = flag_3D
        if self.flag_3D:
            self.conv_block = nn.Sequential(nn.BatchNorm3d(self.output_number_of_channels),nn.ReLU(inplace=False), nn.Conv3d(self.output_number_of_channels,self.output_number_of_channels,1,2))
        else:
            self.conv_block = nn.Sequential(nn.BatchNorm2d(self.output_number_of_channels),nn.ReLU(inplace=False), nn.Conv2d(self.output_number_of_channels,self.output_number_of_channels,1,2))

    # Elementwise sum the output of a submodule to its input
    def forward(self, x):
        return self.conv_block(x)


class DownSample_Projection_Strided_SameChannels(nn.Module):
    def __init__(self, downscale_factor=2, flag_3D=False):
        super(DownSample_Projection_Strided_SameChannels, self).__init__()
        self.downscale_factor = downscale_factor;
        self.flag_3D = flag_3D
        self.conv_block = None

    # Elementwise sum the output of a submodule to its input
    def forward(self, x):
        if self.conv_block is None:
            self.output_number_of_channels = x.shape[1]
            if self.flag_3D:
                self.conv_block = nn.Sequential(nn.BatchNorm3d(self.output_number_of_channels), nn.ReLU(inplace=False), nn.Conv3d(self.output_number_of_channels, self.output_number_of_channels, 1, self.downscale_factor))
            else:
                self.conv_block = nn.Sequential(nn.BatchNorm2d(self.output_number_of_channels), nn.ReLU(inplace=False), nn.Conv2d(self.output_number_of_channels, self.output_number_of_channels, 1, self.downscale_factor))
        return self.conv_block(x)



def UnshufflePixels_Function(input_tensor, r):
    batch_size, c, h, w = input_tensor.shape
    output_tensor = torch.zeros(batch_size, r ** 2 * c, h // r, w // r).to(input_tensor.device)
    counter = 0;
    for i in arange(r):
        for j in arange(r):
            # print(str(r*i+j) + ': ' + str(r*i+j+c))
            output_tensor[:, c*counter:c*(counter+1), :, :] = input_tensor[:, :, i::r, j::r][:, :, 0:h // r, 0:w // r]
            counter += 1;
            # output_tensor[:, r * i + j:r * i + j + c, :, :] = input_tensor[:, :, i::r, j::r][:, :, 0:h // r, 0:w // r]  #the h//r and w//r is to avoid problems with non whole multiplication

    # out_channel = c * (r ** 2)  # out_h = h // r
    # out_w = w // r
    # fm_view = input_tensor.contiguous().view(batch_size, c, out_h, r, out_w, r)
    # fm_prime = fm_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(batch_size, out_channel, out_h, out_w)
    #  # for i in arange(int(fm_prime.shape[1]/c)):
    #     figure()
    #     imshow_torch(fm_prime[0,3*i:3*(i+1),:,:])
    return output_tensor





class ShufflePixels(nn.Module):
    # Initialize this with a module
    def __init__(self, upscale_factor=2):
        super(ShufflePixels, self).__init__()
        self.upscale_factor = upscale_factor;

    # Elementwise sum the output of a submodule to its input
    def forward(self, x):
        return ShufflePixels_Function(x,self.upscale_factor)

def ShufflePixels_Function(input_tensor, r):
    batch_size, c, h, w = input_tensor.shape
    output_number_of_channels = int(c/r**2)
    output_tensor = torch.zeros(batch_size, int(c/r**2), h * r, w * r).to(input_tensor.device)
    counter = 0;
    for i in arange(r):
        for j in arange(r):
            output_tensor[:, :, i::r, j::r] = input_tensor[:, output_number_of_channels*counter:output_number_of_channels*(counter+1), :, :]
            # output_tensor[:, r * i + j:r * i + j + c, :, :] = input_tensor[:, :, i::r, j::r][:, :, 0:h // r, 0:w // r]
            counter += 1

    # out_channel = c * (r ** 2)  # out_h = h // r
    # out_w = w // r
    # fm_view = input_tensor.contiguous().view(batch_size, c, out_h, r, out_w, r)
    # fm_prime = fm_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(batch_size, out_channel, out_h, out_w)
    #  # for i in arange(int(fm_prime.shape[1]/c)):
    #     figure()
    #     imshow_torch(fm_prime[0,3*i:3*(i+1),:,:])
    return output_tensor




# ### Testing Shuffle Unshuffle: ###
# input_tensor = read_image_torch('C:/Users\dkarl\PycharmProjects\dudykarl\Imagenet/n01496331/n01496331_490.jpg')
# input_tensor_unshuffled = UnshufflePixels(2)(input_tensor)
# for i in arange(2**2):
#     figure(i)
#     imshow_torch(input_tensor_unshuffled[:,i*3:(i+1)*3,:,:]/255)
#
# # output_tensor_shuffled = nn.PixelShuffle(2)(input_tensor_unshuffled)
# output_tensor_shuffled = ShufflePixels(2)(input_tensor_unshuffled)
# imshow_torch(output_tensor_shuffled/255)



# nn.Upsample -> Conv_Block Layer.
def Up_Conv_Block(number_of_input_channels, number_of_output_channels, upscale_factor=2, kernel_size=3, stride=1,
                  bias=True,
                  padding_type='zero', normalization_function=None, activation_function='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample_layer = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv_layer = Conv_Block(number_of_input_channels, number_of_output_channels, kernel_size, stride, bias=bias,
                            padding_type=padding_type, normalization_function=normalization_function,
                            activation_function=activation_function)
    return Pile_Modules_On_Top_Of_Each_Other(upsample_layer, conv_layer)
############################################################################################################################################################################################################################################################












#################################################################################################################################################################################################################################
### CARN block: ###
def init_weights(modules):
    pass

class CARN_BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(CARN_BasicBlock, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
                                  nn.ReLU(inplace=True))

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class CARN_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CARN_ResidualBlock, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1), )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class CARN_EResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(CARN_EResidualBlock, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, 1, 1, 0), )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class CARN_UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, group=1):
        super(CARN_UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _CARN_UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _CARN_UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _CARN_UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _CARN_UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _CARN_UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super(_CARN_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group),
                            nn.ReLU(inplace=True)]
                modules += [ShufflePixels(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group),
                        nn.ReLU(inplace=True)]
            modules += [ShufflePixels(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class CARN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(CARN_Block, self).__init__()

        self.b1 = CARN_ResidualBlock(64, 64)
        self.b2 = CARN_ResidualBlock(64, 64)
        self.b3 = CARN_ResidualBlock(64, 64)
        self.c1 = CARN_BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = CARN_BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = CARN_BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class CARN_Net(nn.Module):
    def __init__(self, **kwargs):
        super(CARN_Net, self).__init__()

        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        group = kwargs.get("group", 1)


        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = CARN_Block(64, 64)
        self.b2 = CARN_Block(64, 64)
        self.b3 = CARN_Block(64, 64)
        self.c1 = CARN_BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = CARN_BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = CARN_BasicBlock(64 * 4, 64, 1, 1, 0)

        self.upsample = CARN_UpsampleBlock(64, scale=scale, multi_scale=multi_scale, group=group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x, scale):
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=scale)

        out = self.exit(out)

        return out
#################################################################################################################################################################################################################################















#DownSample Strategy:
def Get_DownSample_Block(flag_downsample_strategy, flag_3D=False):
    if flag_downsample_strategy == 'maxpool':
        if flag_3D:
            downsample_block = nn.MaxPool3d(2,2)
        else:
            downsample_block = nn.MaxPool2d(2,2)
    elif flag_downsample_strategy == 'unshuffle':
        #TODO: still doesn't support 3D
        downsample_block = UnshufflePixels(2) #less spatial extent more channels!!!!
    elif flag_downsample_strategy == 'simple_downsample':
        downsample_block = DownSample_Simple(2,flag_3D)  #TODO: generalize downsample_simple to 3D as well
    elif flag_downsample_strategy == 'bilinear':
        downsample_block = DownSample_Bilinear(2,flag_3D) #TODO: generalize downsample_bilinear to 3D as well
    elif flag_downsample_strategy == 'DPP_learned':
        #TODO: still doesn't support 3D
        downsample_block = DPP_learned()
    elif flag_downsample_strategy == 'DPP':
        #TODO: still doesn't support 3D
        downsample_block = DPP()
    elif flag_downsample_strategy == 'avgpool':
        if flag_3D:
            downsample_block = nn.AvgPool3d(2,2)
        else:
            downsample_block = nn.AvgPool2d(2,2)
    elif flag_downsample_strategy == 'strided_conv_block':
        downsample_block = DownSample_Projection_Strided_SameChannels(2) #TODO: generalize downsample_projection_strided_samechannels to 3D as well

    return downsample_block





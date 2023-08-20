import functools

import torch.nn as nn
import torch.nn.functional as F

from SemanticSegmentation_Utils import get_upsampling_weight
from SemanticSegmentation_Losses import cross_entropy2d





####################################################################################################################################################################################
#### Functions Applications Methods: ####
#TODO: add parallel_apply
def apply_advanced(parent_module, function):
    """Applies ``function`` recursively to every submodule (as returned by ``.children()``).
       Similar to module.apply() but also returns the parent object of every module being
       traversed.

       Parameters
       ----------
       parent_module : nn.Module
           Module object representing the root of the computation graph.

       function : function closure
           Function with signature (child_module (nn.Module), child_name (str), parent_module (nn.Module)
    """

    for child_name, child_module in parent_module.named_children():
        function(child_module, child_name, parent_module)

        apply_advanced(child_module, function)



def apply_advanced(parent_module, function):
    """Applies ``function`` recursively to every submodule (as returned by ``.children()``).
       Similar to module.apply() but also returns the parent object of every module being
       traversed.

       Parameters
       ----------
       parent_module : nn.Module
           Module object representing the root of the computation graph.

       function : function closure
           Function with signature (child_module (nn.Module), child_name (str), parent_module (nn.Module)
    """

    for child_name, child_module in parent_module.named_children():
        function(child_module, child_name, parent_module)

        apply_advanced(child_module, function)
####################################################################################################################################################################################









####################################################################################################################################################################################
########## Basic, Repetitive Conv Blocks: ###############
#TODO: add instance norm?
class Conv2D_BatchNorm(nn.Module):
    def __init__(
        self,
        number_of_input_channels,
        number_of_output_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        flag_use_BN=True):
        super(Conv2D_BatchNorm, self).__init__()

        conv_module = nn.Conv2d(int(number_of_input_channels),
                                int(number_of_output_channels),
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=bias,
                                dilation=dilation)

        if flag_use_BN:
            self.conv_BN_block = nn.Sequential(conv_module, nn.BatchNorm2d(int(number_of_output_channels)))
        else:
            self.conv_BN_block = nn.Sequential(conv_BN_block)

    def forward(self, inputs):
        outputs = self.conv_BN_block(inputs)
        return outputs


class Conv2D_GroupNorm(nn.Module):
    def __init__(
        self,
        number_of_input_channels,
        number_of_output_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        number_of_groups=16):
        super(Conv2D_GroupNorm, self).__init__()

        conv_module = nn.Conv2d(int(number_of_input_channels),
                                int(number_of_output_channels),
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                bias   =bias,
                                dilation=dilation,)

        self.conv_GN_block = nn.Sequential(conv_module,
                                     nn.GroupNorm(number_of_groups, int(number_of_output_channels)))

    def forward(self, inputs):
        outputs = self.conv_GN_block(inputs)
        return outputs


class Conv2D_Transpose_BatchNorm(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels, kernel_size, stride, padding, bias=True):
        super(Conv2D_Transpose_BatchNorm, self).__init__()

        self.conv_transpose_block = nn.Sequential(
            nn.ConvTranspose2d(
                int(number_of_input_channels),
                int(number_of_output_channels),
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(int(number_of_output_channels)),
        )

    def forward(self, inputs):
        outputs = self.conv_transpose_block(inputs)
        return outputs


class Conv2D_BatchNorm_ReLU(nn.Module):
    def __init__(
        self,
        number_of_input_channels,
        number_of_output_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        flag_use_BN=True,
    ):
        super(Conv2D_BatchNorm_ReLU, self).__init__()

        conv_module = nn.Conv2d(int(number_of_input_channels),
                             int(number_of_output_channels),
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation,)

        if flag_use_BN:
            self.conv_BN_Relu_block = nn.Sequential(conv_module,
                                          nn.BatchNorm2d(int(number_of_output_channels)),
                                          nn.ReLU(inplace=True))
        else:
            self.conv_BN_Relu_block = nn.Sequential(conv_module, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.conv_BN_Relu_block(inputs)
        return outputs


class Conv2D_GroupNorm_ReLU(nn.Module):
    def __init__(
        self,
        number_of_input_channels,
        number_of_output_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        number_of_groups=16,
    ):
        super(Conv2D_GroupNorm_ReLU, self).__init__()

        conv_module = nn.Conv2d(int(number_of_input_channels),
                             int(number_of_output_channels),
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation,)

        self.conv_GN_Relu = nn.Sequential(conv_module,
                                      nn.GroupNorm(number_of_groups, int(number_of_output_channels)),
                                      nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.conv_GN_Relu(inputs)
        return outputs



class Conv2D_Transpose_BatchNorm_ReLU(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels, kernel_size, stride, padding, bias=True):
        super(Conv2D_Transpose_BatchNorm_ReLU, self).__init__()

        self.conv_transpose_BN_Relu_block = nn.Sequential(
            nn.ConvTranspose2d(
                int(number_of_input_channels),
                int(number_of_output_channels),
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(int(number_of_output_channels)),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.conv_transpose_BN_Relu_block(inputs)
        return outputs
####################################################################################################################################################################################






####################################################################################################################################################################################
#############   Residual Blocks:  ###############
class Residual_Block(nn.Module):
    #TODO: i should change the name to ResBlock_33 and generalize the residual block to other possible blocks
    expansion = 1

    def __init__(self, number_of_input_channels, number_of_output_channels, stride=1, DownSample_Block=None):
        super(Residual_Block, self).__init__()

        self.conv_bn_relu_1 = Conv2D_BatchNorm_ReLU(number_of_input_channels, number_of_output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_bn_2 = Conv2D_BatchNorm(number_of_output_channels, number_of_output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.DownSample_Block = DownSample_Block
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv_bn_relu_1(x)
        out = self.conv_bn_2(out)

        if self.DownSample_Block is not None: #i think i should generalize this to "projection" block (moreover...we're using a conv block before...why would we need downsampling...if anything we will need upsampling)
            residual = self.DownSample_Block(x)

        out += residual
        out = self.relu(out)
        return out


class Residual_Block_BottleNeck(nn.Module):
    #TODO: change name to something like resblock_131
    expansion = 4

    def __init__(self, number_of_input_channels, number_of_output_channels, stride=1, DownSample_Block=None):
        #final output number of channels is number_of_output_channels*4...so i think i should change input variables names...
        #moreover...it seems the number_of_output_channels should be equal to number_of_input_channels/4 because we add a variable with number_of_output_channels*4 channels to a vairable with number_of_input_channels variable
        #that is, unless we use a downsample block...so it seems either number_of_input_channels=number_of_output_channels*4 or number_of_input_channels=number_of_outptu_channels (if there's a downsample block)
        super(Residual_Block_BottleNeck, self).__init__()
        self.conv_bn_1 = Conv2D_BatchNorm(number_of_input_channels, number_of_output_channels, kernel_size=1, bias=False)
        self.conv_bn_2 = Conv2D_BatchNorm(number_of_output_channels, number_of_output_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.conv_bn_3 = Conv2D_BatchNorm(number_of_output_channels, number_of_output_channels * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.DownSample_Block = DownSample_Block
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv_bn_1(x)
        out = self.conv_bn_2(out)
        out = self.conv_bn_3(out)

        if self.DownSample_Block is not None:
            residual = self.DownSample_Block(x)

        out += residual
        out = self.relu(out)

        return out



class Chained_Residual_Pooling(nn.Module):
    def __init__(self, number_of_output_channels, input_shape):
        #How can number_of_output_channels not be equal to input_shape[1]?
        super(Chained_Residual_Pooling, self).__init__()

        self.chained_residual_pooling = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(input_shape[1], number_of_output_channels, kernel_size=3), #i don't see how you can add input and output without having Conv2d with padding=1!?!?!??!?!
        )

    def forward(self, x):
        input = x
        x = self.chained_residual_pooling(x)
        return x + input
####################################################################################################################################################################################






####################################################################################################################################################################################
### Special Layers To Check: ###
#TODO: add deformable convolution
def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    "3x3 convolution with padding"
    #Basic Structure to be used for other specialized convolution layers because it has a calculation of effective kernel size and needed padding to maintain constant number of channels

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)




class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) / 2
        pad1 = (kernel_size[1] - 1) / 2
        # kernel size had better be odd number so as to avoid alignment error (!!!!!)
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out
####################################################################################################################################################################################





####################################################################################################################################################################################
class MultiResolution_FeatureFusion_Block(nn.Module):
    #Very similiar to the other Fusion Blocks in that it upsamples the lower resolution, convs it and ADDs it to the high resolution....
    # what's different here is that it's more general and can Upsample BOTH high resolution and low resolution input features
    def __init__(self, number_of_output_channels, upsample_factor_high, upsample_factor_low, high_shape, low_shape):
        super(MultiResolution_FeatureFusion_Block, self).__init__()

        self.upsample_factor_high = upsample_factor_high
        self.upsample_factor_low = upsample_factor_low
        number_of_input_channels_high = high_shape[1];
        number_of_input_channels_low = low_shape[1];
        self.conv_high = nn.Conv2d(number_of_input_channels_high, number_of_output_channels, kernel_size=3)
        if low_shape is not None:
            self.conv_low = nn.Conv2d(number_of_input_channels_low, number_of_output_channels, kernel_size=3)

    def forward(self, x_high, x_low):
        #Both inputs are upsampled by a predetermined factor...when i this used?
        #after upsampling we add the two results together
        high_upsampled = F.upsample(self.conv_high(x_high), scale_factor=self.upsample_factor_high, mode="bilinear")

        if x_low is None:
            return high_upsampled

        low_upsampled = F.upsample(self.conv_low(x_low), scale_factor=self.upsample_factor_low, mode="bilinear")

        return low_upsampled + high_upsampled
####################################################################################################################################################################################






####################################################################################################################################################################################
###########  Simple Utils:  ##############
def get_interp_size(input, s_factor=1, z_factor=1):  # for caffe
    ori_h, ori_w = input.shape[2:]

    # shrink (s_factor >= 1)
    ori_h = (ori_h - 1) / s_factor + 1
    ori_w = (ori_w - 1) / s_factor + 1

    # zoom (z_factor >= 1)
    ori_h = ori_h + (ori_h - 1) * (z_factor - 1)
    ori_w = ori_w + (ori_w - 1) * (z_factor - 1)

    resize_shape = (int(ori_h), int(ori_w))
    return resize_shape


def interp(input, output_size, mode="bilinear"):
    n, c, ih, iw = input.shape
    oh, ow = output_size

    # normalize to [-1, 1]
    h = torch.arange(0, oh, dtype=torch.float, device='cuda' if input.is_cuda else 'cpu') / (oh - 1) * 2 - 1
    w = torch.arange(0, ow, dtype=torch.float, device='cuda' if input.is_cuda else 'cpu') / (ow - 1) * 2 - 1

    grid = torch.zeros(oh, ow, 2, dtype=torch.float, device='cuda' if input.is_cuda else 'cpu')
    grid[:, :, 0] = w.unsqueeze(0).repeat(oh, 1)
    grid[:, :, 1] = h.unsqueeze(0).repeat(ow, 1).transpose(0, 1)
    grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)  # grid.shape: [n, oh, ow, 2]

    return F.grid_sample(input, grid, mode=mode)


def get_upsampling_weight(number_of_input_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((number_of_input_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(number_of_input_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()
####################################################################################################################################################################################








######################################################################################################################################################################################################################################################################################################################################################################################################################
############### Fully Convolutional Networks: #################
# FCN32s
class fcn32s(nn.Module):
    def __init__(self, number_of_classes=21, flag_learned_billinear=False):
        super(fcn32s, self).__init__()
        self.flag_learned_billinear = flag_learned_billinear
        self.number_of_classes = number_of_classes
        self.loss = functools.partial(cross_entropy2d, size_average=False)

        #TODO: this, again, is a repetitive block so i should define it as a layer
        #Conv3x3->Conv3x3(increasing channels)->MaxPool(downsampling)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        #Conv3x3->Conv3x3(increasing channels)->MaxPool(downsampling)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        #Conv3x3->Conv3x3(increasing channels)->MaxPool(downsampling)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        #Conv3x3->Conv3x3(increasing channels)->MaxPool(downsampling)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        #Conv3x3->Conv3x3->Conv3x3(Constant number of channels)->MaxPool(downsampling)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        #Classifier Head: Conv7x7(increase number of channels)->Conv1x1(constant number of channels)->Conv1x1 (segmentation map for each class)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.number_of_classes, 1),
        )

        if self.flag_learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        #Upsample Segmentation Map (which has low resolution) to original image size:
        out = F.upsample(score, x.size()[2:])

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]: ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


class fcn16s(nn.Module):
    def __init__(self, number_of_classes=21, flag_learned_billinear=False):
        super(fcn16s, self).__init__()
        self.flag_learned_billinear = flag_learned_billinear
        self.number_of_classes = number_of_classes
        self.loss = functools.partial(cross_entropy2d,
                                      size_average=False)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.number_of_classes, 1),
        )

        self.score_pool4 = nn.Conv2d(512, self.number_of_classes, 1)

        # TODO: Add support for learned upsampling
        if self.flag_learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score_from_conv5 = self.classifier(conv5)
        score_from_conv4 = self.score_pool4(conv4)

        score_from_conv5 = F.upsample(score_from_conv5, score_from_conv4.size()[2:])
        score_from_conv5 += score_from_conv4
        out = F.upsample(score, x.size()[2:])

        return out

    #Initialize Conv Blocks with VGG16 parameters:
    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        #conv_block, l1,l2 etc' are pointers and so this is a way to infuse the above FCN network parameters with vgg16 parameters
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]: ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print(idx, l1, l2)
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


# FCN 8s - including Feature Pyramid Fusion Strategy at the end (NOT Feature Pyramid Network)
class fcn8s(nn.Module):
    def __init__(self, number_of_classes=21, flag_learned_billinear=True):
        super(fcn8s, self).__init__()
        self.flag_learned_billinear = flag_learned_billinear
        self.number_of_classes = number_of_classes
        self.loss = functools.partial(cross_entropy2d,
                                      size_average=False)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.number_of_classes, 1),
        )

        self.score_pool4 = nn.Conv2d(512, self.number_of_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.number_of_classes, 1)

        if self.flag_learned_billinear: #instead of simple bilinear upsampling interpolation use deconvolution (conv transpose) to implement it
            self.upscore2 = nn.ConvTranspose2d(self.number_of_classes, self.number_of_classes, 4,
                                               stride=2, bias=False)
            self.upscore4 = nn.ConvTranspose2d(self.number_of_classes, self.number_of_classes, 4,
                                               stride=2, bias=False)
            self.upscore8 = nn.ConvTranspose2d(self.number_of_classes, self.number_of_classes, 16,
                                               stride=8, bias=False)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(get_upsampling_weight(m.in_channels,
                                                          m.out_channels,
                                                          m.kernel_size[0]))

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        if self.flag_learned_billinear:
            upscore2 = self.upscore2(score)
            score_pool4c = self.score_pool4(conv4)[:, :, 5:5 + upscore2.size()[2],
                           5:5 + upscore2.size()[3]]
            upscore_pool4 = self.upscore4(upscore2 + score_pool4c)

            score_pool3c = self.score_pool3(conv3)[:, :, 9:9 + upscore_pool4.size()[2],
                           9:9 + upscore_pool4.size()[3]]

            out = self.upscore8(score_pool3c + upscore_pool4)[:, :, 31:31 + x.size()[2],
                  31:31 + x.size()[3]]
            return out.contiguous()


        else:
            score_pool4 = self.score_pool4(conv4)
            score_pool3 = self.score_pool3(conv3)
            score = F.upsample(score, score_pool4.size()[2:])
            score += score_pool4
            score = F.upsample(score, score_pool3.size()[2:])
            score += score_pool3
            out = F.upsample(score, x.size()[2:])

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]: ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]
######################################################################################################################################################################################################################################################################################################################################################################################################################










######################################################################################################################################################################################################################################################################################################################################################################################################################
############## FRRN - Full Resolution Residual Networks: ###############
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from ptsemseg.models.utils import *
from ptsemseg.loss import bootstrapped_cross_entropy2d



class FRRN_FullResolution_ResidualBlock(nn.Module):
    """
    Full Resolution Residual Unit for FRRN (FRRN=full resolution residual network???)
    """
    #This is Basically a Fusion Block...i need to rename this
    def __init__(self,
                 number_of_previous_channels,
                 number_of_output_channels,
                 upsample_factor,
                 flag_group_norm=False,
                 number_of_groups=None):
        super(FRRN_FullResolution_ResidualBlock, self).__init__()
        self.upsample_factor = upsample_factor
        self.number_of_previous_channels = number_of_previous_channels
        self.number_of_output_channels = number_of_output_channels
        self.flag_group_norm = flag_group_norm
        self.number_of_groups = number_of_groups


        if self.flag_group_norm:
            #(*). Note - notice the conv2d has number_of_previous_channels+32 input channels....why the +32????
            self.conv1 = Conv2D_GroupNorm_ReLU(number_of_previous_channels + 32, number_of_output_channels, kernel_size=3,
                                                stride=1, padding=1, bias=False, number_of_groups=self.number_of_groups)
            self.conv2 = Conv2D_GroupNorm_ReLU(number_of_output_channels, number_of_output_channels, kernel_size=3,
                                                stride=1, padding=1, bias=False, number_of_groups=self.number_of_groups)

        else:
            self.conv1 = Conv2D_BatchNorm_ReLU(number_of_previous_channels + 32, number_of_output_channels, kernel_size=3, stride=1, padding=1, bias=False,)
            self.conv2 = Conv2D_BatchNorm_ReLU(number_of_output_channels, number_of_output_channels, kernel_size=3, stride=1, padding=1, bias=False,)

        self.conv_res = nn.Conv2d(number_of_output_channels, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, x1_low_resolution, x2_high_resolution):
        #z is downsampled & concatenated with y,
        #conved, upsampled, connected residually with original input z:

        #DownSample then Conv:
        x2_high_resolution_downsampled = nn.MaxPool2d(self.upsample_factor, self.upsample_factor)(x2_high_resolution)
        x1_and_x2downsampled_concatenated = torch.cat([x1_low_resolution, x2_high_resolution_downsampled], dim=1)
        x1_and_x2downsampled_concatenated_conv = self.conv1(x1_and_x2downsampled_concatenated)
        x1_and_x2downsampled_concatenated_conv = self.conv2(x1_and_x2downsampled_concatenated_conv)

        upsample_size = torch.Size([_s * self.upsample_factor for _s in x1_and_x2downsampled_concatenated_conv.shape[-2:]])
        #Conv low resolution --> Upsample --> Add to z
        x1_and_x2downsampled_concatenated_conv = self.conv_res(x1_and_x2downsampled_concatenated_conv)
        x1_and_x2downsampled_concatenated_conv_upsampled = F.upsample(x1_and_x2downsampled_concatenated_conv, size=upsample_size, mode="nearest")
        z_prime = x2_high_resolution + x1_and_x2downsampled_concatenated_conv_upsampled

        return x1_and_x2downsampled_concatenated_conv, z_prime




class FRRN_ResidualBlock(nn.Module):
    """
    Residual Unit for FRRN
    """
    #TODO: change name to Conv2D_GroupNorm_Residual_Block or something
    def __init__(self,
                 number_of_constant_channels,
                 kernel_size=3,
                 strides=1,
                 flag_group_norm=False,
                 number_of_groups=None):
        super(FRRN_ResidualBlock, self).__init__()
        self.flag_group_norm = flag_group_norm
        self.number_of_groups = number_of_groups

        if self.flag_group_norm:
            self.conv1 = Conv2D_GroupNorm_ReLU(
                number_of_constant_channels, number_of_constant_channels, kernel_size=kernel_size,
               stride=strides, padding=1, bias=False,number_of_groups=self.number_of_groups)
            self.conv2 = Conv2D_GroupNorm(
                number_of_constant_channels, number_of_constant_channels, kernel_size=kernel_size,
                stride=strides, padding=1, bias=False,number_of_groups=self.number_of_groups)

        else:
            self.conv1 = Conv2D_BatchNorm_ReLU(
                number_of_constant_channels, number_of_constant_channels, kernel_size=kernel_size, stride=strides, padding=1, bias=False,)
            self.conv2 = Conv2D_BatchNorm(
                number_of_constant_channels, number_of_constant_channels, kernel_size=kernel_size, stride=strides, padding=1, bias=False,)

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming



# each spec is as (n_blocks, channels, scale)
frrn_specs_dic = {
    "A": {
        "encoder": [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16]],
        "decoder": [[2, 192, 8], [2, 192, 4], [2, 48, 2]],
    },
    "B": {
        "encoder": [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16], [2, 384, 32]],
        "decoder": [[2, 192, 16], [2, 192, 8], [2, 192, 4], [2, 48, 2]],
    },
}


import SemanticSegmentation_Utils
from SemanticSegmentation_Utils import *

class FRRN_Network(nn.Module):
    """
    Full Resolution Residual Networks for Semantic Segmentation
    URL: https://arxiv.org/abs/1611.08323
    References:
    1) Original Author's code: https://github.com/TobyPDE/FRRN
    2) TF implementation by @kiwonjoon: https://github.com/hiwonjoon/tf-frrn
    """

    def __init__(self,
                 number_of_classes=21,
                 model_type=None,
                 flag_group_norm=False,
                 number_of_groups=16):
        super(FRRN_Network, self).__init__()
        self.number_of_classes = number_of_classes
        self.model_type = model_type
        self.flag_group_norm = flag_group_norm
        self.number_of_groups = number_of_groups

        ####################
        if self.flag_group_norm:
            self.conv1 = Conv2D_GroupNorm_ReLU(3, 48, 5, 1, 2)
        else:
            self.conv1 = Conv2D_BatchNorm_ReLU(3, 48, 5, 1, 2)
        ####################


        ####################
        # Get Initial Input Conv Blocks and Final Output Conv Blocks:
        self.initial_input_conv_units = []
        self.final_output_conv_units = []
        for i in range(3):
            self.initial_input_conv_units.append(FRRN_ResidualBlock(number_of_constant_channels=48, #48 is clearly a hyper parameter
                                                             kernel_size=3,
                                                             strides=1,
                                                             flag_group_norm=self.flag_group_norm,
                                                             number_of_groups=self.number_of_groups))
            self.final_output_conv_units.append(FRRN_ResidualBlock(number_of_constant_channels=48,
                                                               kernel_size=3,
                                                               strides=1,
                                                               flag_group_norm=self.flag_group_norm,
                                                               number_of_groups=self.number_of_groups))
        self.initial_input_conv_units = nn.ModuleList(self.initial_input_conv_units)
        self.final_output_conv_units = nn.ModuleList(self.final_output_conv_units)
        ####################


        ####################
        #Simple Convolution As Initial Processing To The Branch Which Goes Into The "Pyramidial" Decomposition and Reconstruction:
        self.split_conv = nn.Conv2d(48, 32, kernel_size=1, padding=0, stride=1, bias=False)
        ####################


        ####################################################################################################################################################################################
        #Get "Pyramidial" Encoding And Decoding Blocks:
        # each spec is as (n_blocks, channels, scale)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]["encoder"]
        self.decoder_frru_specs = frrn_specs_dic[self.model_type]["decoder"]

        #(*). Encoding
        number_of_previous_channels = 48
        self.encoding_frrus = {}
        for current_number_of_blocks, current_number_of_channels, current_upscale_factor in self.encoder_frru_specs:
            for current_block_index in range(current_number_of_blocks):
                key = "_".join(map(str, ["encoding_frru", current_number_of_blocks, current_number_of_channels, current_upscale_factor, current_block_index]))
                setattr(self, key, FRRN_FullResolution_ResidualBlock(number_of_previous_channels=number_of_previous_channels,
                                        number_of_output_channels=current_number_of_channels,
                                        upsample_factor=current_upscale_factor,
                                        flag_group_norm=self.flag_group_norm,
                                        number_of_groups=self.number_of_groups),)
            number_of_previous_channels = current_number_of_channels

        #(*). Decoding
        self.decoding_frrus = {}
        for current_number_of_blocks, current_number_of_channels, current_upscale_factor in self.decoder_frru_specs:
            # pass through decoding FRRUs
            for current_block_index in range(current_number_of_blocks):
                key = "_".join(map(str, ["decoding_frru", current_number_of_blocks, current_number_of_channels, current_upscale_factor, current_block_index]))
                setattr(self, key, FRRN_FullResolution_ResidualBlock(number_of_previous_channels=number_of_previous_channels,
                                        number_of_output_channels=current_number_of_channels,
                                        upsample_factor=current_upscale_factor,
                                        flag_group_norm=self.flag_group_norm,
                                        number_of_groups=self.number_of_groups),)
            number_of_previous_channels = current_number_of_channels
        ####################################################################################################################################################################################


        ####################
        # Simple Convolution As Final Processing To The Branch Which Goes Into The "Pyramidial" Decomposition and Reconstruction:
        self.merge_conv = nn.Conv2d(number_of_previous_channels + 32, 48, kernel_size=1, padding=0, stride=1, bias=False)
        ####################

        ####################
        # Simple Convolution To Get Number Of Channels Be Equal To Number Of Classes:
        self.classif_conv = nn.Conv2d(48, self.number_of_classes, kernel_size=1, padding=0, stride=1, bias=True)
        ####################


    def forward(self, x):

        ####################
        # pass to initial conv block
        x = self.conv1(x)
        # pass through residual units
        for i in range(3):
            x = self.initial_input_conv_units[i](x)
        ####################


        ####################
        #(*). Divide Stream:
        y = x
        z = self.split_conv(x)
        ####################


        ############################################################
        #### Encoding -> Decoding: ####
        number_of_previous_channels = 48
        #(1). Encoding
        # loop dynamics: the outer loop downsampled y everytime whilst z remains the same size to allow the next pyramid feature folding feature y some high resolution context
        for current_number_of_blocks, current_number_of_channels, current_upscale_factor in self.encoder_frru_specs:
            # maxpool bigger feature map:  (TODO: add possibility to choose between max_pool2d & strided convolution)
            y_pooled = F.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            # pass through encoding FRRUs:
            for current_block_index in range(current_number_of_blocks):
                key = "_".join(map(str, ["encoding_frru", current_number_of_blocks, current_number_of_channels, current_upscale_factor, current_block_index]))
                y, z = getattr(self, key)(y_pooled, z) #Output Here is the input to the next outer loop F.max_pool2d() operation...so at each iteration of the outer loop y is downsampled by a factor of 2 creating effectively a pyramid:
            number_of_previous_channels = current_number_of_channels

        #(2). Decoding
        # loop dynamics: the outer loop upsamples y everytime whilst z remains the same size to allow the next pyramid unfolding feature y some high resolution context
        for current_number_of_blocks, current_number_of_channels, current_upscale_factor in self.decoder_frru_specs:
            # bilinear upsample smaller feature map   (TODO: add posibility to choose between upsample and pixel_shuffle)
            upsample_size = torch.Size([_s * 2 for _s in y.size()[-2:]])
            y_upsampled = F.upsample(y, size=upsample_size, mode="bilinear", align_corners=True)
            # pass through decoding FRRUs
            for current_block_index in range(current_number_of_blocks):
                key = "_".join(map(str, ["decoding_frru", current_number_of_blocks, current_number_of_channels, current_upscale_factor, current_block_index]))
                y, z = getattr(self, key)(y_upsampled, z)
            number_of_previous_channels = current_number_of_channels
        ######################################################################


        ####################
        # Merge Streams:
        x = torch.cat([F.upsample(y, scale_factor=2, mode="bilinear", align_corners=True), z], dim=1) #Notice: align_corners=True
        x = self.merge_conv(x)
        ####################


        ####################
        # pass through residual units
        for i in range(3):
            x = self.final_output_conv_units[i](x)
        ####################


        ####################
        # final 1x1 conv to get classification
        x = self.classif_conv(x)
        ####################

        return x
######################################################################################################################################################################################################################################################################################################################################################################################################################









######################################################################################################################################################################################################################################################################################################################################################################################################################
################### ICNET: ############################
import torch
import numpy as np
import torch.nn as nn

from math import ceil
from torch.autograd import Variable

import SemanticSegmentation_caffe_pb2

class Cascade_HighResLowRes_Feature_Fusion_Layer(nn.Module):
    def __init__(
        self, number_of_classes, low_in_channels, high_in_channels, number_of_output_channels, flag_use_BN=True):
        super(Cascade_HighResLowRes_Feature_Fusion_Layer, self).__init__()

        bias = not flag_use_BN

        self.low_dilated_conv_bn = Conv2D_BatchNorm(
            low_in_channels,
            number_of_output_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            bias=bias,
            dilation=2,
            flag_use_BN=flag_use_BN)
        self.low_classifier_conv = nn.Conv2d(
            int(low_in_channels),
            int(number_of_classes), #this is the final segmentation maps so we have 1 map per class
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
            dilation=1)  # Train only
        self.high_proj_conv_bn = Conv2D_BatchNorm(
            high_in_channels,
            number_of_output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            flag_use_BN=flag_use_BN)

    def forward(self, x_low, x_high):
        #Upsample low resolution input to same size as high resolution one (factor of 2)
        x_low_upsampled = F.interpolate(x_low, size=get_interp_size(x_low, z_factor=2), mode="bilinear", align_corners=True)

        #Get classification maps only from low resolution input:
        low_resolution_branch_classification_maps = self.low_classifier_conv(x_low_upsampled)

        #Fuse low resolution upsampled input with high resolution input to get fused information feature map:
        low_fm = self.low_dilated_conv_bn(x_low_upsampled)
        high_fm = self.high_proj_conv_bn(x_high)
        high_fused_fm = F.relu(low_fm + high_fm, inplace=True)

        return high_fused_fm, low_resolution_branch_classification_maps




class Cascade_HighResLowRes_Feature_Fusion_Layer(nn.Module):
    def __init__(
        self, number_of_classes, low_in_channels, high_in_channels, number_of_output_channels, flag_use_BN=True):
        super(Cascade_HighResLowRes_Feature_Fusion_Layer, self).__init__()

        bias = not flag_use_BN

        self.low_dilated_conv_bn = Conv2D_BatchNorm(
            low_in_channels,
            number_of_output_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            bias=bias,
            dilation=2,
            flag_use_BN=flag_use_BN)
        self.low_classifier_conv = nn.Conv2d(
            int(low_in_channels),
            int(number_of_classes), #this is the final segmentation maps so we have 1 map per class
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
            dilation=1)  # Train only
        self.high_proj_conv_bn = Conv2D_BatchNorm(
            high_in_channels,
            number_of_output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            flag_use_BN=flag_use_BN)

    def forward(self, x_low, x_high):
        #Upsample low resolution input to same size as high resolution one (factor of 2)
        x_low_upsampled = F.interpolate(x_low, size=get_interp_size(x_low, z_factor=2), mode="bilinear", align_corners=True)

        #Get classification maps only from low resolution input:
        low_resolution_branch_classification_maps = self.low_classifier_conv(x_low_upsampled)

        #Fuse low resolution upsampled input with high resolution input to get fused information feature map:
        low_fm = self.low_dilated_conv_bn(x_low_upsampled)
        high_fm = self.high_proj_conv_bn(x_high)
        high_fused_fm = F.relu(low_fm + high_fm, inplace=True)

        return high_fused_fm, low_resolution_branch_classification_maps






icnet_specs = {
    "cityscapes": {
        "number_of_classes": 19,
        "input_size": (1025, 2049),
        "number_of_residual_blocks_per_certain_amount_of_channels": [3, 4, 6, 3],
    }
}
class ICNet(nn.Module):

    """
    Image Cascade Network
    URL: https://arxiv.org/abs/1704.08545
    References:
    1) Original Author's code: https://github.com/hszhao/ICNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/ICNet-tensorflow
    """

    def __init__(
        self,
        number_of_classes=19,
        number_of_residual_blocks_per_certain_amount_of_channels=[3, 4, 6, 3],
        input_size=(1025, 2049),
        version=None,
        flag_use_BN=True):
        super(ICNet, self).__init__()

        bias = not flag_use_BN

        #If specs exist use those, if not use function defaults:
        self.number_of_residual_blocks_per_certain_amount_of_channels = (icnet_specs[version]["number_of_residual_blocks_per_certain_amount_of_channels"] if version is not None else number_of_residual_blocks_per_certain_amount_of_channels)
        self.number_of_classes = (icnet_specs[version]["number_of_classes"] if version is not None else number_of_classes)
        self.input_size = (icnet_specs[version]["input_size"] if version is not None else input_size)


        ################################################################################################################################
        #(*). Encoder
        self.initial_conv_bn_relu_downsample = Conv2D_BatchNorm_ReLU(
            number_of_input_channels=3,
            number_of_output_channels=32,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=bias,
            flag_use_BN=flag_use_BN)
        self.initial_conv_bn_relu_2 = Conv2D_BatchNorm_ReLU(
            number_of_input_channels=32,
            number_of_output_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=bias,
            flag_use_BN=flag_use_BN)
        self.initial_conv_bn_relu_3 = Conv2D_BatchNorm_ReLU(
            number_of_input_channels=32,
            number_of_output_channels=64,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=bias,
            flag_use_BN=flag_use_BN)
        ################################################################################################################################################################



        ################################################################################################################################################################
        ### Vanilla Residual Blocks: ###
        self.intermediate_residual_block_1 = PSP_Multiple_Residual_Blocks(number_of_blocks=self.number_of_residual_blocks_per_certain_amount_of_channels[0],
                                             number_of_input_channels=64,
                                             number_of_intermediate_channels=32,
                                             number_of_output_channels=128,
                                             stride=1,
                                             dilation=1,
                                             flag_use_BN=flag_use_BN)
        self.intermediate_residual_block_2_downsample = PSP_Multiple_Residual_Blocks(number_of_blocks=self.number_of_residual_blocks_per_certain_amount_of_channels[1],
                                                  number_of_input_channels=128,
                                                  number_of_intermediate_channels=64,
                                                  number_of_output_channels=256,
                                                  stride=2,
                                                  dilation=1,
                                                  include_range="conv",
                                                  flag_use_BN=flag_use_BN)
        self.intermediate_residual_block_3_downsample = PSP_Multiple_Residual_Blocks(number_of_blocks=self.number_of_residual_blocks_per_certain_amount_of_channels[1],
                                                      number_of_input_channels=128,
                                                      number_of_intermediate_channels=64,
                                                      number_of_output_channels=256,
                                                      stride=2,
                                                      dilation=1,
                                                      include_range="identity",
                                                      flag_use_BN=flag_use_BN)
        ################################################################################################################################



        ################################################################################################################################
        ### Dilated Residual Blocks: ####
        self.intermediate_residual_block_4_dilation2 = PSP_Multiple_Residual_Blocks(number_of_blocks=self.number_of_residual_blocks_per_certain_amount_of_channels[2],
                                             number_of_input_channels=256,
                                             number_of_intermediate_channels=128,
                                             number_of_output_channels=512,
                                             stride=1,
                                             dilation=2,
                                             flag_use_BN=flag_use_BN)
        self.intermediate_residual_block_5_dilation4 = PSP_Multiple_Residual_Blocks(number_of_blocks=self.number_of_residual_blocks_per_certain_amount_of_channels[3],
                                             number_of_input_channels=512,
                                             number_of_intermediate_channels=256,
                                             number_of_output_channels=1024,
                                             stride=1,
                                             dilation=4,
                                             flag_use_BN=flag_use_BN)
        ################################################################################################################################


        ################################################################################################################################
        # Pyramid Pooling Module
        self.pyramid_pooling = Pyramid_Poooling_MultiPoolSizesFusion(number_of_input_channels=1024,
                                                pool_sizes=[6, 3, 2, 1],
                                                model_name="icnet",
                                                fusion_mode="sum",
                                                flag_use_BN=flag_use_BN)
        ################################################################################################################################


        ################################################################################################################################
        # Final conv layer with kernel 1 in sub4 branch
        self.final_4thLayer_conv = Conv2D_BatchNorm_ReLU(number_of_input_channels=1024,number_of_output_channels=256,kernel_size=1,padding=0,stride=1,bias=bias,flag_use_BN=flag_use_BN)

        # High-resolution (sub1) branch
        self.second_input_branch_conv_bn_relu_1_downsample = Conv2D_BatchNorm_ReLU(number_of_input_channels=3,number_of_output_channels=32,kernel_size=3,padding=1,stride=2,bias=bias,flag_use_BN=flag_use_BN)
        self.second_input_branch_conv_bn_relu_2_downsample = Conv2D_BatchNorm_ReLU(number_of_input_channels=32,number_of_output_channels=32,kernel_size=3,padding=1,stride=2,bias=bias,flag_use_BN=flag_use_BN)
        self.second_input_branch_conv_bn_relu_3_downsample = Conv2D_BatchNorm_ReLU(number_of_input_channels=32,number_of_output_channels=64,kernel_size=3,padding=1,stride=2,bias=bias,flag_use_BN=flag_use_BN)
        self.classification = nn.Conv2d(128, self.number_of_classes, 1, 1, 0)

        # Cascade Feature Fusion Units
        self.final_cascade_feature_fusion_sub24 = Cascade_HighResLowRes_Feature_Fusion_Layer(self.number_of_classes, 256, 256, 128, flag_use_BN=flag_use_BN)
        self.final_cascade_feature_fusion_sub12 = Cascade_HighResLowRes_Feature_Fusion_Layer(self.number_of_classes, 128, 64, 128, flag_use_BN=flag_use_BN)
        ################################################################################################################################


        ################################################################################################################################
        # Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d
        ################################################################################################################################



    def forward(self, x):
        h, w = x.shape[2:]

        ### Pyramidial/Multi-Scale Analysis Of Input: ###
        # H, W -> H/2, W/2
        x_sub2 = F.interpolate(x, size=get_interp_size(x, s_factor=2), mode='bilinear', align_corners=True)
        # H/2, W/2 -> H/4, W/4
        x_sub2 = self.initial_conv_bn_relu_downsample(x_sub2)
        x_sub2 = self.initial_conv_bn_relu_2(x_sub2)
        x_sub2 = self.initial_conv_bn_relu_3(x_sub2)
        # H/4, W/4 -> H/8, W/8
        x_sub2 = F.max_pool2d(x_sub2, 3, 2, 1)
        # H/8, W/8 -> H/16, W/16
        x_sub2 = self.intermediate_residual_block_1(x_sub2)
        x_sub2 = self.intermediate_residual_block_2_downsample(x_sub2)
        # H/16, W/16 -> H/32, W/32
        x_sub4 = F.interpolate(x_sub2, size=get_interp_size(x_sub2, s_factor=2), mode='bilinear', align_corners=True)
        x_sub4 = self.intermediate_residual_block_3_downsample(x_sub4)
        x_sub4 = self.intermediate_residual_block_4_dilation2(x_sub4)
        x_sub4 = self.intermediate_residual_block_5_dilation4(x_sub4)
        x_sub4 = self.pyramid_pooling(x_sub4) #multiple pooling kernels analysis layer
        x_sub4 = self.final_4thLayer_conv(x_sub4)


        ### Pyramidial/Multi-Scale Analysis Of Input Again (but simply with less convolutions?.....): ###
        x_sub1 = self.second_input_branch_conv_bn_relu_1_downsample(x)
        x_sub1 = self.second_input_branch_conv_bn_relu_2_downsample(x_sub1)
        x_sub1 = self.second_input_branch_conv_bn_relu_3_downsample(x_sub1)

        ### Take The Difference Scales And Fuse The Information In Them: ###
        x_sub24, sub4_cls = self.final_cascade_feature_fusion_sub24(x_sub4, x_sub2)
        x_sub12, sub24_cls = self.final_cascade_feature_fusion_sub12(x_sub24, x_sub1)

        #Upsample TO Original Size
        x_sub12 = F.interpolate(x_sub12, size=get_interp_size(x_sub12, z_factor=2), mode="bilinear", align_corners=True)
        sub124_cls = self.classification(x_sub12)


        if self.training:
            return (sub124_cls, sub24_cls, sub4_cls)
        else:  # eval mode
            sub124_cls = F.interpolate(sub124_cls,
                                       size=get_interp_size(sub124_cls, z_factor=4),
                                       mode="bilinear",
                                       align_corners=True)  # Test only
            return sub124_cls



    ##########################
    #Loading Model Weights From Caffe:
    def load_pretrained_model(self, model_path):
        """
        Load weights from caffemodel w/o caffe dependency
        and plug them in corresponding modules
        """
        # My eyes and my heart both hurt when writing this method

        # Only care about layer_types that have trainable parameters
        ltypes = [
            "BNData",
            "ConvolutionData",
            "HoleConvolutionData",
            "Convolution",
        ]  # Convolution type for conv3_sub1_proj

        def _get_layer_params(layer, ltype):

            if ltype == "BNData":
                gamma = np.array(layer.blobs[0].data)
                beta = np.array(layer.blobs[1].data)
                mean = np.array(layer.blobs[2].data)
                var = np.array(layer.blobs[3].data)
                return [mean, var, gamma, beta]

            elif ltype in ["ConvolutionData", "HoleConvolutionData", "Convolution"]:
                is_bias = layer.convolution_param.bias_term
                weights = np.array(layer.blobs[0].data)
                bias = []
                if is_bias:
                    bias = np.array(layer.blobs[1].data)
                return [weights, bias]

            elif ltype == "InnerProduct":
                raise Exception(
                    "Fully connected layers {}, not supported".format(ltype)
                )

            else:
                raise Exception("Unkown layer type {}".format(ltype))

        net = SemanticSegmentation_caffe_pb2.NetParameter()
        with open(model_path, "rb") as model_file:
            net.MergeFromString(model_file.read())

        # dict formatted as ->  key:<layer_name> :: value:<layer_type>
        layer_types = {}
        # dict formatted as ->  key:<layer_name> :: value:[<list_of_params>]
        layer_params = {}

        for l in net.layer:
            lname = l.name
            ltype = l.type
            lbottom = l.bottom
            ltop = l.top
            if ltype in ltypes:
                print("Processing layer {} | {}, {}".format(lname, lbottom, ltop))
                layer_types[lname] = ltype
                layer_params[lname] = _get_layer_params(l, ltype)
            # if len(l.blobs) > 0:
            #    print(lname, ltype, lbottom, ltop, len(l.blobs))

        # Set affine=False for all batchnorm modules
        def _no_affine_bn(module=None):
            if isinstance(module, nn.BatchNorm2d):
                module.affine = False

            if len([m for m in module.children()]) > 0:
                for child in module.children():
                    _no_affine_bn(child)

        # _no_affine_bn(self)

        def _transfer_conv(layer_name, module):
            weights, bias = layer_params[layer_name]
            w_shape = np.array(module.weight.size())

            print(
                "CONV {}: Original {} and trans weights {}".format(
                    layer_name, w_shape, weights.shape
                )
            )

            module.weight.data.copy_(torch.from_numpy(weights).view_as(module.weight))

            if len(bias) != 0:
                b_shape = np.array(module.bias.size())
                print(
                    "CONV {}: Original {} and trans bias {}".format(
                        layer_name, b_shape, bias.shape
                    )
                )
                module.bias.data.copy_(torch.from_numpy(bias).view_as(module.bias))

        def _transfer_bn(conv_layer_name, bn_module):
            mean, var, gamma, beta = layer_params[conv_layer_name + "/bn"]
            print(
                "BN {}: Original {} and trans weights {}".format(
                    conv_layer_name, bn_module.running_mean.size(), mean.shape
                )
            )
            bn_module.running_mean.copy_(
                torch.from_numpy(mean).view_as(bn_module.running_mean)
            )
            bn_module.running_var.copy_(
                torch.from_numpy(var).view_as(bn_module.running_var)
            )
            bn_module.weight.data.copy_(
                torch.from_numpy(gamma).view_as(bn_module.weight)
            )
            bn_module.bias.data.copy_(torch.from_numpy(beta).view_as(bn_module.bias))

        def _transfer_conv_bn(conv_layer_name, mother_module):
            conv_module = mother_module[0]
            _transfer_conv(conv_layer_name, conv_module)

            if conv_layer_name + "/bn" in layer_params.keys():
                bn_module = mother_module[1]
                _transfer_bn(conv_layer_name, bn_module)

        def _transfer_residual(block_name, block):
            block_module, n_layers = block[0], block[1]
            prefix = block_name[:5]

            if ("bottleneck" in block_name) or (
                "identity" not in block_name
            ):  # Conv block
                bottleneck = block_module.layers[0]
                bottleneck_conv_bn_dic = {
                    prefix + "_1_1x1_reduce": bottleneck.cbr1.cbr_unit,
                    prefix + "_1_3x3": bottleneck.cbr2.cbr_unit,
                    prefix + "_1_1x1_proj": bottleneck.cb4.cb_unit,
                    prefix + "_1_1x1_increase": bottleneck.cb3.cb_unit,
                }

                for k, v in bottleneck_conv_bn_dic.items():
                    _transfer_conv_bn(k, v)

            if ("identity" in block_name) or (
                "bottleneck" not in block_name
            ):  # Identity blocks
                base_idx = 2 if "identity" in block_name else 1

                for layer_idx in range(2, n_layers + 1):
                    residual_layer = block_module.layers[layer_idx - base_idx]
                    residual_conv_bn_dic = {
                        "_".join(
                            map(str, [prefix, layer_idx, "1x1_reduce"])
                        ): residual_layer.cbr1.cbr_unit,
                        "_".join(
                            map(str, [prefix, layer_idx, "3x3"])
                        ): residual_layer.cbr2.cbr_unit,
                        "_".join(
                            map(str, [prefix, layer_idx, "1x1_increase"])
                        ): residual_layer.cb3.cb_unit,
                    }

                    for k, v in residual_conv_bn_dic.items():
                        _transfer_conv_bn(k, v)

        convbn_layer_mapping = {
            "conv1_1_3x3_s2": self.initial_conv_bn_relu_downsample.cbr_unit,
            "conv1_2_3x3": self.initial_conv_bn_relu_2.cbr_unit,
            "conv1_3_3x3": self.initial_conv_bn_relu_3.cbr_unit,
            "conv1_sub1": self.second_input_branch_conv_bn_relu_1_downsample.cbr_unit,
            "conv2_sub1": self.second_input_branch_conv_bn_relu_2_downsample.cbr_unit,
            "conv3_sub1": self.second_input_branch_conv_bn_relu_3_downsample.cbr_unit,
            # 'conv5_3_pool6_conv': self.pyramid_pooling.paths[0].cbr_unit,
            # 'conv5_3_pool3_conv': self.pyramid_pooling.paths[1].cbr_unit,
            # 'conv5_3_pool2_conv': self.pyramid_pooling.paths[2].cbr_unit,
            # 'conv5_3_pool1_conv': self.pyramid_pooling.paths[3].cbr_unit,
            "final_4thLayer_conv": self.final_4thLayer_conv.cbr_unit,
            "conv_sub4": self.final_cascade_feature_fusion_sub24.low_dilated_conv_bn.cb_unit,
            "conv3_1_sub2_proj": self.final_cascade_feature_fusion_sub24.high_proj_conv_bn.cb_unit,
            "conv_sub2": self.final_cascade_feature_fusion_sub12.low_dilated_conv_bn.cb_unit,
            "conv3_sub1_proj": self.final_cascade_feature_fusion_sub12.high_proj_conv_bn.cb_unit,
        }

        residual_layers = {
            "conv2": [self.intermediate_residual_block_1, self.number_of_residual_blocks_per_certain_amount_of_channels[0]],
            "conv3_bottleneck": [self.intermediate_residual_block_2_downsample, self.number_of_residual_blocks_per_certain_amount_of_channels[1]],
            "conv3_identity": [self.intermediate_residual_block_3_downsample, self.number_of_residual_blocks_per_certain_amount_of_channels[1]],
            "conv4": [self.intermediate_residual_block_4_dilation2, self.number_of_residual_blocks_per_certain_amount_of_channels[2]],
            "conv5": [self.intermediate_residual_block_5_dilation4, self.number_of_residual_blocks_per_certain_amount_of_channels[3]],
        }

        # Transfer weights for all non-residual conv+bn layers
        for k, v in convbn_layer_mapping.items():
            _transfer_conv_bn(k, v)

        # Transfer weights for final non-bn conv layer
        _transfer_conv("conv6_cls", self.classification)
        _transfer_conv("conv6_sub4", self.final_cascade_feature_fusion_sub24.low_classifier_conv)
        _transfer_conv("conv6_sub2", self.final_cascade_feature_fusion_sub12.low_classifier_conv)

        # Transfer weights for all residual layers
        for k, v in residual_layers.items():
            _transfer_residual(k, v)
    ##########################


    ##########################
    #The Segmentation Maps Prediction Function: This Uses the .forward() fuction
    def tile_predict(self, input_image_batch, include_flip_mode=True):
        """
        Predict by takin overlapping tiles from the image.
        Strides are adaptively computed from the input_image_batch shape
        and input size
        :param input_image_batch: torch.Tensor with shape [N, C, H, W] in BGR format
        :param side: int with side length of model input
        :param number_of_classes: int with number of classes in seg output.
        """

        expected_model_input_height, expected_model_input_width = self.input_size
        number_of_classes = self.number_of_classes
        input_batch_size, input_number_of_channels, input_image_height, input_image_width = input_image_batch.shape
        # n = int(max(input_image_height,input_image_width) / float(side) + 1)
        number_of_height_cuts_and_predictions = int(input_image_height / float(expected_model_input_height) + 1)
        number_of_width_cuts_and_predictions = int(input_image_width / float(expected_model_input_width) + 1)
        stride_x = (input_image_height - expected_model_input_height) / float(number_of_height_cuts_and_predictions)
        stride_y = (input_image_width - expected_model_input_width) / float(number_of_width_cuts_and_predictions)
        
        
        #####################################
        #height indices list of [start_index,stop_index] lists:
        x_ends = [
            [int(i * stride_x), int(i * stride_x) + expected_model_input_height] for i in range(number_of_height_cuts_and_predictions + 1)
        ]
        #width indices list of [start_index,stop_index] lists:
        y_ends = [
            [int(i * stride_y), int(i * stride_y) + expected_model_input_width] for i in range(number_of_width_cuts_and_predictions + 1)
        ]
        #####################################
        
        
        ######################################
        #Initial Prediction Maps and prediction counts 
        pred = np.zeros([input_batch_size, number_of_classes, input_image_height, input_image_width])
        count = np.zeros([input_image_height, input_image_width])
        ######################################
        
        
        ######################################
        #Cut Part of the image each time, predict on that crop of the image, and continue to the next crop until entire image is predicted:
        slice_count = 0
        for start_index_height, stop_index_height in x_ends:
            for start_index_width, stop_index_width in y_ends:
                slice_count += 1
                
                #Get Current Crop Location:
                current_image_crop = input_image_batch[:, :, start_index_height:stop_index_height, start_index_width:stop_index_width]
                if include_flip_mode:
                    current_image_crop_flipped = torch.from_numpy( np.copy(current_image_crop.cpu().numpy()[:, :, :, ::-1]) ).float()
                
                flag_is_model_on_cuda = next(self.parameters()).is_cuda
                
                #Make Current Image Crops Variables:
                current_image_crop_as_variable = Variable(current_image_crop, volatile=True)
                if include_flip_mode:
                    current_image_crop_flipped_variable = Variable(current_image_crop_flipped, volatile=True)
                
                #If Model is on CUDA than load current crops to gpus:
                if flag_is_model_on_cuda:
                    current_image_crop_as_variable = current_image_crop_as_variable.cuda()
                    if include_flip_mode:
                        current_image_crop_flipped_variable = current_image_crop_flipped_variable.cuda()

                #Take Current Crop --> Pass it to the above defined .forward() function to predict segmentation logits --> use softmax to predict probabilities
                psub1 = F.softmax(self.forward(current_image_crop_as_variable), dim=1).data.cpu().numpy()
                if include_flip_mode:
                    psub2 = F.softmax(self.forward(current_image_crop_flipped_variable), dim=1).data.cpu().numpy()
                    psub = (psub1 + psub2[:, :, :, ::-1]) / 2.0
                else:
                    psub = psub1

                #Put current crop probabilities maps into large image prediction maps:
                pred[:, :, start_index_height:stop_index_height, start_index_width:stop_index_width] = psub
                count[start_index_height:stop_index_height, start_index_width:stop_index_width] += 1.0

        score = (pred / count[None, None, ...]).astype(np.float32)
        return score / np.expand_dims(score.sum(axis=1), axis=1)


# For Testing Purposes only
if __name__ == "__main__":
    cd = 0
    import os
    from torch.autograd import Variable
    import matplotlib.pyplot as plt
    import scipy.misc as scipy_misceleneous
    from ptsemseg.loader.cityscapes_loader import cityscapesLoader as cityscapes_loader

    ICNet_generator = ICNet(version="cityscapes", flag_use_BN=False)

    # Just need to do this one time
    caffemodel_dir_path = "PATH_TO_ICNET_DIR/evaluation/model"
    ICNet_generator.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, "icnet_cityscapes_train_30k.caffemodel"))
    # ICNet_generator.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'icnet_cityscapes_train_30k_bnnomerge.caffemodel'))
    # ICNet_generator.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'icnet_cityscapes_trainval_90k.caffemodel'))
    # ICNet_generator.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'icnet_cityscapes_trainval_90k_bnnomerge.caffemodel'))

    # ICNet_generator.load_state_dict(torch.load('ic.pth'))

    ICNet_generator.float()
    ICNet_generator.cuda(cd)
    ICNet_generator.eval()

    dataset_root_dir = "PATH_TO_CITYSCAPES_DIR"
    cityscapes_dataset_dataloader = cityscapes_loader(root=dataset_root_dir)
    img = scipy_misceleneous.imread(os.path.join(dataset_root_dir, "leftImg8bit/demoVideo/stuttgart_00/stuttgart_00_000000_000010_leftImg8bit.png"))
    scipy_misceleneous.imsave("test_input.png", img)
    orig_size = img.shape[:-1]
    img = scipy_misceleneous.imresize(img, ICNet_generator.input_size)  # uint8 with RGB mode
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float64)
    img -= np.array([123.68, 116.779, 103.939])[:, None, None]
    img = np.copy(img[::-1, :, :])
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)

    out = ICNet_generator.tile_predict(img)
    pred = np.argmax(out, axis=1)[0]
    pred = pred.astype(np.float32)
    pred = scipy_misceleneous.imresize(pred, orig_size, "nearest", mode="F")  # float32 with F mode
    decoded = cityscapes_dataset_dataloader.decode_segmap(pred)
    scipy_misceleneous.imsave("test_output.png", decoded)
    # scipy_misceleneous.imsave('test_output.png', pred)

    checkpoints_dir_path = "checkpoints"
    if not os.path.exists(checkpoints_dir_path):
        os.mkdir(checkpoints_dir_path)
    ICNet_generator = torch.nn.DataParallel(ICNet_generator, device_ids=range(torch.cuda.device_count()))
    state = {"model_state": ICNet_generator.state_dict()}
    torch.save(state, os.path.join(checkpoints_dir_path, "icnet_cityscapes_train_30k.pth"))
    # torch.save(state, os.path.join(checkpoints_dir_path, "icnetBN_cityscapes_train_30k.pth"))
    # torch.save(state, os.path.join(checkpoints_dir_path, "icnet_cityscapes_trainval_90k.pth"))
    # torch.save(state, os.path.join(checkpoints_dir_path, "icnetBN_cityscapes_trainval_90k.pth"))
    print("Output Shape {} \t Input Shape {}".format(out.shape, img.shape))
######################################################################################################################################################################################################################################################################################################################################################################################################################








######################################################################################################################################################################################################################################################################################################################################################################################################################
#######################   LinkNet:   #########################
import torch.nn as nn



class LinkNet_Up(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels):
        super(LinkNet_Up, self).__init__()

        # B, 2C, H, W -> B, C/2, H, W
        self.conv_bn_relu_1 = Conv2D_BatchNorm_ReLU(number_of_input_channels, number_of_output_channels / 2, kernel_size=1, stride=1, padding=1)

        # B, C/2, H, W -> B, C/2, H, W
        self.deconv_bn_relu_2 = Conv2D_Transpose_BatchNorm_ReLU(number_of_output_channels / 2, number_of_output_channels / 2, kernel_size=3, stride=2, padding=0)

        # B, C/2, H, W -> B, C, H, W
        self.conv_bn_relu_3 = Conv2D_BatchNorm_ReLU(number_of_output_channels / 2, number_of_output_channels, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_bn_relu_1(x)
        x = self.deconv_bn_relu_2(x)
        x = self.conv_bn_relu_3(x)
        return x



class linknet(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        number_of_classes=21,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(linknet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.layers = [2, 2, 2, 2]  # Currently hardcoded for ResNet-18

        filters = [64, 128, 256, 512]
        filters = [x / self.feature_scale for x in filters]

        self.inplanes = filters[0]

        # Encoder
        self.convbnrelu1 = conv2DBatchNormRelu(
            in_channels=3, k_size=7, n_filters=64, padding=3, stride=2, bias=False
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = residualBlock
        self.encoder1 = self._make_layer(block, filters[0], self.layers[0])
        self.encoder2 = self._make_layer(block, filters[1], self.layers[1], stride=2)
        self.encoder3 = self._make_layer(block, filters[2], self.layers[2], stride=2)
        self.encoder4 = self._make_layer(block, filters[3], self.layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        # Decoder
        self.decoder4 = LinkNet_Up(filters[3], filters[2])
        self.decoder4 = LinkNet_Up(filters[2], filters[1])
        self.decoder4 = LinkNet_Up(filters[1], filters[0])
        self.decoder4 = LinkNet_Up(filters[0], filters[0])

        # Final Classifier
        self.finaldeconvbnrelu1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32 / feature_scale, 3, 2, 1),
            nn.BatchNorm2d(32 / feature_scale),
            nn.ReLU(inplace=True),
        )
        self.finalconvbnrelu2 = Conv2D_BatchNorm_ReLU(
            in_channels=32 / feature_scale,
            k_size=3,
            n_filters=32 / feature_scale,
            padding=1,
            stride=1,
        )
        self.finalconv3 = nn.Conv2d(32 / feature_scale, number_of_classes, 2, 2, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.convbnrelu1(x)
        x = self.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4)
        d4 += e3
        d3 = self.decoder3(d4)
        d3 += e2
        d2 = self.decoder2(d3)
        d2 += e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconvbnrelu1(d1)
        f2 = self.finalconvbnrelu2(f1)
        f3 = self.finalconv3(f2)

        return f3
######################################################################################################################################################################################################################################################################################################################################################################################################################








######################################################################################################################################################################################################################################################################################################################################################################################################################
################## PSP-Net (Pyramid Scene Parsing Net): ####################
import torch
import numpy as np
import torch.nn as nn

from math import ceil
from torch.autograd import Variable

import SemanticSegmentation_caffe_pb2


class Pyramid_Poooling_MultiPoolSizesFusion(nn.Module):
    #Change Name of layer to something like: grouped_multiple_pooling_kernels_conv_layer
    def __init__(
        self,
        number_of_input_channels,
        pool_sizes,
        model_name="pspnet",
        fusion_mode="cat",
        flag_use_BN=True,):
        super(Pyramid_Poooling_MultiPoolSizesFusion, self).__init__()

        bias = not flag_use_BN

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(
                Conv2D_BatchNorm_ReLU(
                    number_of_input_channels,
                    int(number_of_input_channels / len(pool_sizes)),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=bias,
                    flag_use_BN=flag_use_BN,
                )
            )

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    def forward(self, x):
        input_images_height, input_images_width = x.shape[2:]

        #Get pooling kernel_sizes and strides for the different slices (the number of which is distated by the number of pool_sizes):
        if self.training or self.model_name != "icnet":  # general settings or pspnet
            pooling_kernel_sizes_list = []
            pooling_strides_list = []
            for current_pool_size in self.pool_sizes:
                # TODO: i think this is wrong because the pool sizes with which BOTH icnet and pspnet are called are small [3,4,5,6...] and so converting pool size to image_size/pool_size is wrong
                pooling_kernel_sizes_list.append((int(input_images_height / current_pool_size), int(input_images_width / current_pool_size)))
                pooling_strides_list.append((int(input_images_height / current_pool_size), int(input_images_width / current_pool_size)))
        else:  # eval mode and icnet: pre-trained for 1025 x 2049
            pooling_kernel_sizes_list = [(8, 15), (13, 25), (17, 33), (33, 65)] #(*). note the pooling sizes are different for the two axis
            pooling_strides_list = [(5, 10), (10, 20), (16, 32), (33, 65)]


        #Loop over the different pool sizes, pass input x through an average_pooling layer, then through a convolution (module) and then upsample to final size (the original x size):
        if self.fusion_mode == 'cat':  # pspnet: concat (including x)
            output_slices = [x]
        else:  # icnet: element_wise sum (including x)
            pp_sum = x

        for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
            current_output = F.avg_pool2d(x, pooling_kernel_sizes_list[i], stride=pooling_strides_list[i], padding=0)
            # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
            if self.model_name != "icnet":
                current_output = module(current_output)
                current_output = F.interpolate(current_output, size=(input_images_height, input_images_width), mode="bilinear", align_corners=True)

            if self.fusion_mode == 'cat': #pspnet: concat (including x)
                output_slices.append(current_output)
            else: #icnet: element_wise sum (including x)
                pp_sum = pp_sum + current_output


        #Return:
        if self.fusion_mode == 'cat':  # pspnet: concat (including x)
            return torch.cat(output_slices, dim=1)
        else:  # icnet: element_wise sum (including x)
            return pp_sum






class PSP_BottleNeck(nn.Module):
    #TODO: rename to something like residual_block_131 or something
    #Residual block with 131 convolution on the main branch and 1x1 convolution on the residual branch....(haven't this been implemented above?)
    def __init__(self, number_of_input_channels, number_of_intermediate_channels, number_of_output_channels, stride, dilation=1, flag_use_BN=True):
        super(PSP_BottleNeck, self).__init__()

        bias = not flag_use_BN

        #Initial 1x1 Convolution:
        self.cbr1 = Conv2D_BatchNorm_ReLU(
            number_of_input_channels,
            number_of_intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            flag_use_BN=flag_use_BN)

        #Intermediate 3x3 Convolution (whether dialeted or not)
        if dilation > 1:
            #Dilated
            self.cbr2 = Conv2D_BatchNorm_ReLU(
                number_of_intermediate_channels,
                number_of_intermediate_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                bias=bias,
                dilation=dilation,
                flag_use_BN=flag_use_BN)
        else:
            #Non Dilated
            self.cbr2 = Conv2D_BatchNorm_ReLU(
                number_of_intermediate_channels,
                number_of_intermediate_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias,
                dilation=1,
                flag_use_BN=flag_use_BN)

        #Final 1x1 Convolution:
        self.cb3 = Conv2D_BatchNorm(
            number_of_intermediate_channels,
            number_of_output_channels,
            1,
            stride=1,
            padding=0,
            bias=bias,
            flag_use_BN=flag_use_BN)

        #Single 1x1 Convolution for the residual branch:
        self.cb4 = Conv2D_BatchNorm(
            number_of_input_channels,
            number_of_output_channels,
            1,
            stride=stride,
            padding=0,
            bias=bias,
            flag_use_BN=flag_use_BN)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)
        return F.relu(conv + residual, inplace=True)



class PSP_BottleNeck_IdentityResidual(nn.Module):
    #(*****). DIFFERENCE BETWEEN THIS AND REGULAR RESIDUAL BOTTLENECK: The difference between this and the above PSP_BottleNeck is that the residual branch doesn't get it's own convolution (thus the name identity)
    def __init__(self, number_of_input_channels, number_of_intermediate_channels, stride, dilation=1, flag_use_BN=True):
        super(PSP_BottleNeck_IdentityResidual, self).__init__()

        bias = not flag_use_BN

        #Initial 1x1 Convolution
        self.cbr1 = Conv2D_BatchNorm_ReLU(
            number_of_input_channels,
            number_of_intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            flag_use_BN=flag_use_BN)

        #Intermediate 3x3 Convolution:
        if dilation > 1:
            self.cbr2 = Conv2D_BatchNorm_ReLU(
                number_of_intermediate_channels,
                number_of_intermediate_channels,
                3,
                stride=1,
                padding=dilation,
                bias=bias,
                dilation=dilation,
                flag_use_BN=flag_use_BN)
        else:
            self.cbr2 = Conv2D_BatchNorm_ReLU(
                number_of_intermediate_channels,
                number_of_intermediate_channels,
                3,
                stride=1,
                padding=1,
                bias=bias,
                dilation=1,
                flag_use_BN=flag_use_BN)


        #Final 1x1 Convolution:
        self.cb3 = Conv2D_BatchNorm(
            number_of_intermediate_channels,
            number_of_input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            flag_use_BN=flag_use_BN)

    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x + residual, inplace=True)


class PSP_Multiple_Residual_Blocks(nn.Module):
    def __init__(
        self,
        number_of_blocks,
        number_of_input_channels,
        number_of_intermediate_channels,
        number_of_output_channels,
        stride,
        dilation=1,
        include_range="all",
        flag_use_BN=True):
        super(PSP_Multiple_Residual_Blocks, self).__init__()

        if dilation > 1:
            stride = 1

        # PSP_Multiple_Residual_Blocks = convBlockPSP + identityBlockPSPs
        #(*). Note: i believe the equal equation is: PSP_Multiple_Residual_Blocks = PSP_BottleNeck(PSP_BottleNeck_IdentityResidual(x)) if include_range=='all'
        layers = []
        if include_range in ["all", "conv"]:
            layers.append(PSP_BottleNeck(number_of_input_channels,number_of_intermediate_channels,number_of_output_channels,stride,dilation,flag_use_BN=flag_use_BN))
        if include_range in ["all", "identity"]:
            for i in range(number_of_blocks - 1):
                #(*). Note: PSP_BottleNeck_IdentityResidual, being a residual block, resturns a batch with output number of channels being equal the input number of channels... so each blocks here returns a batch with number_of_output_channels number of channels
                layers.append(PSP_BottleNeck_IdentityResidual(number_of_output_channels, number_of_intermediate_channels, stride, dilation, flag_use_BN=flag_use_BN))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)






pspnet_specs = {
    "pascal": {
        "number_of_classes": 21,
        "input_size": (473, 473),
        "number_of_residual_blocks_per_certain_amount_of_channels": [3, 4, 23, 3],
    },
    "cityscapes": {
        "number_of_classes": 19,
        "input_size": (713, 713),
        "number_of_residual_blocks_per_certain_amount_of_channels": [3, 4, 23, 3],
    },
    "ade20k": {
        "number_of_classes": 150,
        "input_size": (473, 473),
        "number_of_residual_blocks_per_certain_amount_of_channels": [3, 4, 6, 3],
    },
}


class PSPNET(nn.Module):

    """
    Pyramid Scene Parsing Network
    URL: https://arxiv.org/abs/1612.01105
    References:
    1) Original Author's code: https://github.com/hszhao/PSPNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/PSPNet-tensorflow
    Visualization:
    http://dgschwend.github.io/netscope/#/gist/6bfb59e6a3cfcb4e2bb8d47f827c2928
    """

    def __init__(
        self,
        number_of_classes=21,
        number_of_residual_blocks_per_certain_amount_of_channels=[3, 4, 23, 3],
        input_size=(473, 473),
        version=None):

        super(PSPNET, self).__init__()

        self.number_of_residual_blocks_per_certain_amount_of_channels = (
            pspnet_specs[version]["number_of_residual_blocks_per_certain_amount_of_channels"]
            if version is not None
            else number_of_residual_blocks_per_certain_amount_of_channels)
        self.number_of_classes = (pspnet_specs[version]["number_of_classes"] if version is not None else number_of_classes)
        self.input_size = (pspnet_specs[version]["input_size"] if version is not None else input_size)

        # Encoder
        self.convbnrelu1_1 = Conv2D_BatchNorm_ReLU(number_of_input_channels=3,
                                                   number_of_output_channels=64,
                                                   kernel_size=3,
                                                   padding=1,
                                                   stride=2,
                                                   bias=False)
        self.convbnrelu1_2 = Conv2D_BatchNorm_ReLU(number_of_input_channels=64,
                                                   kernel_size=3,
                                                   number_of_output_channels=64,
                                                   padding=1,
                                                   stride=1,
                                                   bias=False)
        self.convbnrelu1_3 = Conv2D_BatchNorm_ReLU(number_of_input_channels=64,
                                                   number_of_output_channels=128,
                                                   kernel_size=3,
                                                   padding=1,
                                                   stride=1,
                                                   bias=False)

        # Vanilla Residual Blocks
        self.res_block2 = PSP_Multiple_Residual_Blocks(number_of_blocks=self.number_of_residual_blocks_per_certain_amount_of_channels[0],
                                             number_of_input_channels=128,
                                             number_of_intermediate_channels=64,
                                             number_of_output_channels=256,
                                             stride=1,
                                             dilation=1)
        self.res_block3 = PSP_Multiple_Residual_Blocks(number_of_blocks=self.number_of_residual_blocks_per_certain_amount_of_channels[1],
                                             number_of_input_channels=256,
                                             number_of_intermediate_channels=128,
                                             number_of_output_channels=512,
                                             stride=2,
                                             dilation=1)

        # Dilated Residual Blocks
        self.res_block4 = PSP_Multiple_Residual_Blocks(number_of_blocks=self.number_of_residual_blocks_per_certain_amount_of_channels[2],
                                             number_of_input_channels=512,
                                             number_of_intermediate_channels=256,
                                             number_of_output_channels=1024,
                                             stride=1,
                                             dilation=2)
        self.res_block5 = PSP_Multiple_Residual_Blocks(number_of_blocks=self.number_of_residual_blocks_per_certain_amount_of_channels[3],
                                             number_of_input_channels=1024,
                                             number_of_intermediate_channels=512,
                                             number_of_output_channels=2048,
                                             stride=1,
                                             dilation=4)

        # Pyramid Pooling Module
        self.pyramid_pooling = Pyramid_Poooling_MultiPoolSizesFusion(number_of_input_channels=2048,
                                                pool_sizes=[6, 3, 2, 1],
                                                model_name='PSPNET',
                                                fusion_mode='cat',
                                                flag_use_BN=True)
        import tensorflow as tf

        # Final conv layers
        self.cbr_final = Conv2D_BatchNorm_ReLU(number_of_input_channels=4096, number_of_output_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.classification = nn.Conv2d(512, self.number_of_classes, 1, 1, 0)

        # Auxiliary layers for training
        self.convbnrelu4_aux = Conv2D_BatchNorm_ReLU(number_of_input_channels=1024, kernel_size=3, number_of_output_channels=256, padding=1, stride=1, bias=False)
        self.aux_cls = nn.Conv2d(256, self.number_of_classes, kernel_size=1, stride=1, padding=0)

        # Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d

    def forward(self, x):
        inp_shape = x.shape[2:]

        # H, W -> H/2, W/2
        x = self.convbnrelu1_1(x)
        x = self.convbnrelu1_2(x)
        x = self.convbnrelu1_3(x)

        # H/2, W/2 -> H/4, W/4
        x = F.max_pool2d(x, 3, 2, 1)

        # H/4, W/4 -> H/8, W/8
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # Auxiliary layers for training
        x_aux = self.convbnrelu4_aux(x)
        x_aux = self.dropout(x_aux)
        x_aux = self.aux_cls(x_aux)

        x = self.res_block5(x)

        x = self.pyramid_pooling(x)

        x = self.cbr_final(x)
        x = self.dropout(x)

        x = self.classification(x)
        x = F.upsample(x, size=inp_shape, mode="bilinear")

        if self.training:
            return x_aux, x
        else:  # eval mode
            return x


    def load_pretrained_model(self, model_path):
        """
        Load weights from caffemodel w/o caffe dependency
        and plug them in corresponding modules
        """
        # My eyes and my heart both hurt when writing this method

        # Only care about layer_types that have trainable parameters
        ltypes = ["BNData", "ConvolutionData", "HoleConvolutionData"]

        def _get_layer_params(layer, ltype):

            if ltype == "BNData":
                gamma = np.array(layer.blobs[0].data)
                beta = np.array(layer.blobs[1].data)
                mean = np.array(layer.blobs[2].data)
                var = np.array(layer.blobs[3].data)
                return [mean, var, gamma, beta]

            elif ltype in ["ConvolutionData", "HoleConvolutionData"]:
                is_bias = layer.convolution_param.bias_term
                weights = np.array(layer.blobs[0].data)
                bias = []
                if is_bias:
                    bias = np.array(layer.blobs[1].data)
                return [weights, bias]

            elif ltype == "InnerProduct":
                raise Exception(
                    "Fully connected layers {}, not supported".format(ltype)
                )

            else:
                raise Exception("Unkown layer type {}".format(ltype))

        net = SemanticSegmentation_caffe_pb2.NetParameter()
        with open(model_path, "rb") as model_file:
            net.MergeFromString(model_file.read())

        # dict formatted as ->  key:<layer_name> :: value:<layer_type>
        layer_types = {}
        # dict formatted as ->  key:<layer_name> :: value:[<list_of_params>]
        layer_params = {}

        for l in net.layer:
            lname = l.name
            ltype = l.type
            if ltype in ltypes:
                print("Processing layer {}".format(lname))
                layer_types[lname] = ltype
                layer_params[lname] = _get_layer_params(l, ltype)

        # Set affine=False for all batchnorm modules
        def _no_affine_bn(module=None):
            if isinstance(module, nn.BatchNorm2d):
                module.affine = False

            if len([scipy_misceleneous for scipy_misceleneous in module.children()]) > 0:
                for child in module.children():
                    _no_affine_bn(child)

        # _no_affine_bn(self)

        def _transfer_conv(layer_name, module):
            weights, bias = layer_params[layer_name]
            w_shape = np.array(module.weight.size())

            print(
                "CONV {}: Original {} and trans weights {}".format(
                    layer_name, w_shape, weights.shape
                )
            )

            module.weight.data.copy_(torch.from_numpy(weights).view_as(module.weight))

            if len(bias) != 0:
                b_shape = np.array(module.bias.size())
                print(
                    "CONV {}: Original {} and trans bias {}".format(
                        layer_name, b_shape, bias.shape
                    )
                )
                module.bias.data.copy_(torch.from_numpy(bias).view_as(module.bias))

        def _transfer_conv_bn(conv_layer_name, mother_module):
            conv_module = mother_module[0]
            bn_module = mother_module[1]

            _transfer_conv(conv_layer_name, conv_module)

            mean, var, gamma, beta = layer_params[conv_layer_name + "/bn"]
            print(
                "BN {}: Original {} and trans weights {}".format(
                    conv_layer_name, bn_module.running_mean.size(), mean.shape
                )
            )
            bn_module.running_mean.copy_(
                torch.from_numpy(mean).view_as(bn_module.running_mean)
            )
            bn_module.running_var.copy_(
                torch.from_numpy(var).view_as(bn_module.running_var)
            )
            bn_module.weight.data.copy_(
                torch.from_numpy(gamma).view_as(bn_module.weight)
            )
            bn_module.bias.data.copy_(torch.from_numpy(beta).view_as(bn_module.bias))

        def _transfer_residual(prefix, block):
            block_module, n_layers = block[0], block[1]

            bottleneck = block_module.layers[0]
            bottleneck_conv_bn_dic = {
                prefix + "_1_1x1_reduce": bottleneck.cbr1.cbr_unit,
                prefix + "_1_3x3": bottleneck.cbr2.cbr_unit,
                prefix + "_1_1x1_proj": bottleneck.cb4.cb_unit,
                prefix + "_1_1x1_increase": bottleneck.cb3.cb_unit,
            }

            for k, v in bottleneck_conv_bn_dic.items():
                _transfer_conv_bn(k, v)

            for layer_idx in range(2, n_layers + 1):
                residual_layer = block_module.layers[layer_idx - 1]
                residual_conv_bn_dic = {
                    "_".join(
                        map(str, [prefix, layer_idx, "1x1_reduce"])
                    ): residual_layer.cbr1.cbr_unit,
                    "_".join(
                        map(str, [prefix, layer_idx, "3x3"])
                    ): residual_layer.cbr2.cbr_unit,
                    "_".join(
                        map(str, [prefix, layer_idx, "1x1_increase"])
                    ): residual_layer.cb3.cb_unit,
                }

                for k, v in residual_conv_bn_dic.items():
                    _transfer_conv_bn(k, v)

        convbn_layer_mapping = {
            "conv1_1_3x3_s2": self.convbnrelu1_1.cbr_unit,
            "conv1_2_3x3": self.convbnrelu1_2.cbr_unit,
            "conv1_3_3x3": self.convbnrelu1_3.cbr_unit,
            "conv5_3_pool6_conv": self.pyramid_pooling.paths[0].cbr_unit,
            "conv5_3_pool3_conv": self.pyramid_pooling.paths[1].cbr_unit,
            "conv5_3_pool2_conv": self.pyramid_pooling.paths[2].cbr_unit,
            "conv5_3_pool1_conv": self.pyramid_pooling.paths[3].cbr_unit,
            "conv5_4": self.cbr_final.cbr_unit,
            "conv4_" + str(self.number_of_residual_blocks_per_certain_amount_of_channels[2] + 1): self.convbnrelu4_aux.cbr_unit,
        }  # Auxiliary layers for training

        residual_layers = {
            "conv2": [self.res_block2, self.number_of_residual_blocks_per_certain_amount_of_channels[0]],
            "conv3": [self.res_block3, self.number_of_residual_blocks_per_certain_amount_of_channels[1]],
            "conv4": [self.res_block4, self.number_of_residual_blocks_per_certain_amount_of_channels[2]],
            "conv5": [self.res_block5, self.number_of_residual_blocks_per_certain_amount_of_channels[3]],
        }

        # Transfer weights for all non-residual conv+bn layers
        for k, v in convbn_layer_mapping.items():
            _transfer_conv_bn(k, v)

        # Transfer weights for final non-bn conv layer
        _transfer_conv("conv6", self.classification)
        _transfer_conv("conv6_1", self.aux_cls)

        # Transfer weights for all residual layers
        for k, v in residual_layers.items():
            _transfer_residual(k, v)


    def tile_predict(self, input_image_batch, include_flip_mode=True):
        """
        Predict by takin overlapping tiles from the image.
        Strides are adaptively computed from the input_image_batch shape
        and input size
        :param input_image_batch: torch.Tensor with shape [N, input_number_of_channels, H, W] in BGR format
        :param side: int with side length of model input
        :param number_of_classes: int with number of classes in seg output.
        """

        expected_model_input_height, expected_model_input_width = self.input_size
        number_of_classes = self.number_of_classes
        input_batch_size, input_number_of_channels, input_image_height, input_image_width = input_image_batch.shape
        # n = int(max(input_image_height,input_image_width) / float(side) + 1)
        number_of_height_cuts_and_predictions = int(input_image_height / float(expected_model_input_height) + 1)
        number_of_width_cuts_and_predictions = int(input_image_width / float(expected_model_input_width) + 1)
        stride_x = (input_image_height - expected_model_input_height) / float(number_of_height_cuts_and_predictions)
        stride_y = (input_image_width - expected_model_input_width) / float(number_of_width_cuts_and_predictions)

        x_ends = [
            [int(i * stride_x), int(i * stride_x) + expected_model_input_height] for i in range(number_of_height_cuts_and_predictions + 1)
        ]
        y_ends = [
            [int(i * stride_y), int(i * stride_y) + expected_model_input_width] for i in range(number_of_width_cuts_and_predictions + 1)
        ]

        pred = np.zeros([input_batch_size, number_of_classes, input_image_height, input_image_width])
        count = np.zeros([input_image_height, input_image_width])

        slice_count = 0
        for start_index_height, stop_index_height in x_ends:
            for start_index_width, stop_index_width in y_ends:
                slice_count += 1

                current_image_crop = input_image_batch[:, :, start_index_height:stop_index_height, start_index_width:stop_index_width]
                if include_flip_mode:
                    current_image_crop_flipped = torch.from_numpy(
                        np.copy(current_image_crop.cpu().numpy()[:, :, :, ::-1])
                    ).float()

                flag_is_model_on_cuda = next(self.parameters()).is_cuda

                current_image_crop_as_variable = Variable(current_image_crop, volatile=True)
                if include_flip_mode:
                    current_image_crop_flipped_variable = Variable(current_image_crop_flipped, volatile=True)

                if flag_is_model_on_cuda:
                    current_image_crop_as_variable = current_image_crop_as_variable.cuda()
                    if include_flip_mode:
                        current_image_crop_flipped_variable = current_image_crop_flipped_variable.cuda()

                psub1 = F.softmax(self.forward(current_image_crop_as_variable), dim=1).data.cpu().numpy()
                if include_flip_mode:
                    psub2 = F.softmax(self.forward(current_image_crop_flipped_variable), dim=1).data.cpu().numpy()
                    psub = (psub1 + psub2[:, :, :, ::-1]) / 2.0
                else:
                    psub = psub1

                pred[:, :, start_index_height:stop_index_height, start_index_width:stop_index_width] = psub
                count[start_index_height:stop_index_height, start_index_width:stop_index_width] += 1.0

        score = (pred / count[None, None, ...]).astype(np.float32)
        return score / np.expand_dims(score.sum(axis=1), axis=1)


# For Testing Purposes only
if __name__ == "__main__":
    cd = 0
    import os
    from torch.autograd import Variable
    import matplotlib.pyplot as plt
    import scipy.misc as scipy_misceleneous
    from ptsemseg.loader.cityscapes_loader import cityscapesLoader as cityscapes_loader

    psp = PSPNET(version="cityscapes")

    # Just need to do this one time
    caffemodel_dir_path = "PATH_TO_PSPNET_DIR/evaluation/model"
    psp.load_pretrained_model(
        model_path=os.path.join(caffemodel_dir_path, "pspnet101_cityscapes.caffemodel")
    )
    # psp.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'pspnet50_ADE20K.caffemodel'))
    # psp.load_pretrained_model(model_path=os.path.join(caffemodel_dir_path, 'pspnet101_VOC2012.caffemodel'))

    # psp.load_state_dict(torch.load('psp.pth'))

    psp.float()
    psp.cuda(cd)
    psp.eval()

    dataset_root_dir = "PATH_TO_CITYSCAPES_DIR"
    cityscapes_dataset_dataloader = cityscapes_loader(root=dataset_root_dir)
    img = scipy_misceleneous.imread(
        os.path.join(
            dataset_root_dir,
            "leftImg8bit/demoVideo/stuttgart_00/stuttgart_00_000000_000010_leftImg8bit.png",
        )
    )
    scipy_misceleneous.imsave("cropped.png", img)
    orig_size = img.shape[:-1]
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float64)
    img -= np.array([123.68, 116.779, 103.939])[:, None, None]
    img = np.copy(img[::-1, :, :])
    img = torch.from_numpy(img).float()  # convert to torch tensor
    img = img.unsqueeze(0)

    out = psp.tile_predict(img)
    pred = np.argmax(out, axis=1)[0]
    decoded = cityscapes_dataset_dataloader.decode_segmap(pred)
    scipy_misceleneous.imsave("cityscapes_sttutgart_tiled.png", decoded)
    # scipy_misceleneous.imsave('cityscapes_sttutgart_tiled.png', pred)

    checkpoints_dir_path = "checkpoints"
    if not os.path.exists(checkpoints_dir_path):
        os.mkdir(checkpoints_dir_path)
    psp = torch.nn.DataParallel(
        psp, device_ids=range(torch.cuda.device_count())
    )  # append `module.`
    state = {"model_state": psp.state_dict()}
    torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_101_cityscapes.pth"))
    # torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_50_ade20k.pth"))
    # torch.save(state, os.path.join(checkpoints_dir_path, "pspnet_101_pascalvoc.pth"))
    print("Output Shape {} \t Input Shape {}".format(out.shape, img.shape))
######################################################################################################################################################################################################################################################################################################################################################################################################################
#
#
#
#
#
#
#
######################################################################################################################################################################################################################################################################################################################################################################################################################
#########################  SEGNET: #########################
import torch.nn as nn


class SegNet_Down_MaxPooling_2Filters(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels):
        super(SegNet_Down_MaxPooling_2Filters, self).__init__()
        self.conv1 = Conv2D_BatchNorm_ReLU(number_of_input_channels, number_of_output_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2D_BatchNorm_ReLU(number_of_output_channels, number_of_output_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegNet_Down_MaxPooling_3Filters(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels):
        super(SegNet_Down_MaxPooling_3Filters, self).__init__()
        self.conv1 = Conv2D_BatchNorm_ReLU(number_of_input_channels, number_of_output_channels, 3, 1, 1)
        self.conv2 = Conv2D_BatchNorm_ReLU(number_of_output_channels, number_of_output_channels, 3, 1, 1)
        self.conv3 = Conv2D_BatchNorm_ReLU(number_of_output_channels, number_of_output_channels, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegNet_Up_Unpooling_2Filters(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels):
        super(SegNet_Up_Unpooling_2Filters, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2) #UnPooling layer...
        self.conv1 = Conv2D_BatchNorm_ReLU(number_of_input_channels, number_of_input_channels, 3, 1, 1)
        self.conv2 = Conv2D_BatchNorm_ReLU(number_of_input_channels, number_of_output_channels, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class SegNet_Up_Unpooling_3Filters(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels):
        super(SegNet_Up_Unpooling_3Filters, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = Conv2D_BatchNorm_ReLU(number_of_input_channels, number_of_input_channels, 3, 1, 1)
        self.conv2 = Conv2D_BatchNorm_ReLU(number_of_input_channels, number_of_input_channels, 3, 1, 1)
        self.conv3 = Conv2D_BatchNorm_ReLU(number_of_input_channels, number_of_output_channels, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs



class SEGNET(nn.Module):
    #Architecture Shtik - old architecture (based on vgg19 for christ sake....) which is a basic encoder-decoder scheme except that the decoder blocks are unconventional today in the to do upsampling they use UnPool....
    #By the way, maybe it's time to talk about ways of upsampling:
    #(1). Upsample/Interpolate
    #(2). PixelShuffle
    #(3). UnPool
    def __init__(self, number_of_classes=21, in_channels=3, is_unpooling=True):
        super(SEGNET, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = SEGNET_Down_MaxPooling_2Filters(self.in_channels, 64)
        self.down2 = SEGNET_Down_MaxPooling_2Filters(64, 128)
        self.down3 = SEGNET_Down_MaxPooling_3Filters(128, 256)
        self.down4 = SEGNET_Down_MaxPooling_3Filters(256, 512)
        self.down5 = SEGNET_Down_MaxPooling_3Filters(512, 512)

        self.up5 = SEGNET_Up_Unpooling_3Filters(512, 512)
        self.up4 = SEGNET_Up_Unpooling_3Filters(512, 256)
        self.up3 = SEGNET_Up_Unpooling_3Filters(256, 128)
        self.up2 = SEGNET_Up_Unpooling_2Filters(128, 64)
        self.up1 = SEGNET_Up_Unpooling_2Filters(64, number_of_classes)

    def forward(self, inputs):
        #(*). Simple Encoder:
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        #(*). Simple Decoder:
        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
######################################################################################################################################################################################################################################################################################################################################################################################################################
#
#
#
#
#
#
#
#
######################################################################################################################################################################################################################################################################################################################################################################################################################
#########################   UNET:   ########################
import torch.nn as nn

from ptsemseg.models.utils import *


class UNET_Conv2D(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels, flag_use_BN):
        super(UNET_Conv2D, self).__init__()

        if flag_use_BN:
            self.conv1 = nn.Sequential(
                nn.Conv2d(number_of_input_channels, number_of_output_channels, 3, 1, 0),
                nn.BatchNorm2d(number_of_output_channels),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(number_of_output_channels, number_of_output_channels, 3, 1, 0),
                nn.BatchNorm2d(number_of_output_channels),
                nn.ReLU(),
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(number_of_input_channels, number_of_output_channels, 3, 1, 0), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(number_of_output_channels, number_of_output_channels, 3, 1, 0), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNET_Down(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels, flag_use_BN):
        super(UNET_Down, self).__init__()
        self.conv1 = UNET_Conv2D(number_of_input_channels, number_of_output_channels, flag_use_BN)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
    def forward(self, input):
        output_before_maxpool = self.conv1(inputs)
        output_after_maxpool = self.maxpool1(conv1)
        return output_before_maxpool, output_after_maxpool


class UNET_Up(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels, flag_use_Deconvolution_or_simple_upsampling):
        super(UNET_Up, self).__init__()
        if flag_use_Deconvolution_or_simple_upsampling:
            self.up = nn.ConvTranspose2d(number_of_input_channels, number_of_output_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_block_on_fused_inputs = UNET_Conv2D(number_of_input_channels, number_of_output_channels, False)


    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv_block_on_fused_inputs(torch.cat([outputs1, outputs2], 1))




class UNET(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        number_of_classes=21,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True):
        super(UNET, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        number_of_channels_per_scale_vec = [64, 128, 256, 512, 1024]
        number_of_channels_per_scale_vec = [int(x / self.feature_scale) for x in number_of_channels_per_scale_vec]

        # DownSampling 1-4:
        self.down1 = UNET_Down(self.in_channels, number_of_channels_per_scale_vec[0], self.is_batchnorm)
        self.down2 = UNET_Down(number_of_channels_per_scale_vec[0], number_of_channels_per_scale_vec[1], self.is_batchnorm)
        self.down3 = UNET_Down(number_of_channels_per_scale_vec[1], number_of_channels_per_scale_vec[2], self.is_batchnorm)
        self.down4 = UNET_Down(number_of_channels_per_scale_vec[2], number_of_channels_per_scale_vec[3], self.is_batchnorm)

        # Final Conv (WithOut DownSampling):
        self.center = UNET_Conv2D(number_of_channels_per_scale_vec[3], number_of_channels_per_scale_vec[4], self.is_batchnorm)

        # UpSampling 1-4:
        self.up_concat4 = UNET_Up(number_of_channels_per_scale_vec[4], number_of_channels_per_scale_vec[3], self.is_deconv)
        self.up_concat3 = UNET_Up(number_of_channels_per_scale_vec[3], number_of_channels_per_scale_vec[2], self.is_deconv)
        self.up_concat2 = UNET_Up(number_of_channels_per_scale_vec[2], number_of_channels_per_scale_vec[1], self.is_deconv)
        self.up_concat1 = UNET_Up(number_of_channels_per_scale_vec[1], number_of_channels_per_scale_vec[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(number_of_channels_per_scale_vec[0], number_of_classes, 1)

    def forward(self, inputs):
        conv1, maxpool1 = self.down1(inputs)
        conv2, maxpool2 = self.down2(maxpool1)
        conv3, maxpool3 = self.down3(maxpool2)
        conv4, maxpool4 = self.down4(maxpool3)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
######################################################################################################################################################################################################################################################################################################################################################################################################################
#
#
#
#
#
#
#
#
#
#
# ######################################################################################################################################################################################################################################################################################################################################################################################################################
# ####################    Use The Above Defined Model:   ##################
# import copy
# import torchvision.models as models
#
# from ptsemseg.models.fcn import *
# from ptsemseg.models.SEGNET import *
# from ptsemseg.models.UNET import *
# from ptsemseg.models.pspnet import *
# from ptsemseg.models.icnet import *
# from ptsemseg.models.linknet import *
# from ptsemseg.models.FRRN_Network import *
#
#
# def get_model(model_dict, number_of_classes, version=None):
#     name = model_dict['arch']
#     model = _get_model_instance(name)
#     param_dict = copy.deepcopy(model_dict)
#     param_dict.pop('arch')
#
#     if name in ["frrnA", "frrnB"]:
#         model = model(number_of_classes, **param_dict)
#
#     elif name in ["fcn32s", "fcn16s", "fcn8s"]:
#         model = model(number_of_classes=number_of_classes, **param_dict)
#         vgg16 = models.vgg16(pretrained=True)
#         model.init_vgg16_params(vgg16)
#
#     elif name == "SEGNET":
#         model = model(number_of_classes=number_of_classes, **param_dict)
#         vgg16 = models.vgg16(pretrained=True)
#         model.init_vgg16_params(vgg16)
#
#     elif name == "UNET":
#         model = model(number_of_classes=number_of_classes, **param_dict)
#
#     elif name == "PSPNET":
#         model = model(number_of_classes=number_of_classes, **param_dict)
#
#     elif name == "icnet":
#         model = model(number_of_classes=number_of_classes, **param_dict)
#
#     elif name == "icnetBN":
#         model = model(number_of_classes=number_of_classes, **param_dict)
#
#     else:
#         model = model(number_of_classes=number_of_classes, **param_dict)
#
#     return model
#
#
# def _get_model_instance(name):
#     try:
#         return {
#             "fcn32s": fcn32s,
#             "fcn8s": fcn8s,
#             "fcn16s": fcn16s,
#             "UNET": UNET,
#             "SEGNET": SEGNET,
#             "PSPNET": PSPNET,
#             "icnet": ICNet,
#             "icnetBN": ICNet,
#             "linknet": linknet,
#             "frrnA": FRRN_Network,
#             "frrnB": FRRN_Network,
#         }[name]
#     except:
#         raise("Model {} not available".format(name))
# ######################################################################################################################################################################################################################################################################################################################################################################################################################










class RefineNet(nn.Module):
    # Architecture Shtik: take input, pass it through a resnet architecture wil keeping the layers outputs with each resnet layer is downsampled by a factor of 2.
    # now take the most downsampled layer, convolve it, upsample, and add it to the above pyramid layer after convolving the previous layer:
    # refined_HR_lower_layer = conv( conv(HR_lower_layer) + conv(upsample(upper_layer_which_was_itself_refined)) )
    # so you basically refine your prediction at every step by SUMMING (NOT CONCATENATING)
    def __init__(self, num_classes=2):
        """http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0632.pdf

        is temporary placed here
        """

        super(RefineNet, self).__init__()

        resnet50_8s = torchvision.models.resnet18(fully_conv=True,
                                                  pretrained=True,
                                                  output_stride=32,
                                                  remove_avg_pool_layer=True)

        resnet_block_expansion_rate = resnet50_8s.layer1[0].expansion

        self.logit_conv = nn.Conv2d(512,
                                    num_classes,
                                    kernel_size=1)

        self.layer_4_refine_left = BasicBlock(inplanes=512, planes=512)
        self.layer_4_refine_right = BasicBlock(inplanes=512, planes=512)

        self.layer_3_downsample = create_downsample_path(256, 512)
        self.layer_3_refine_left = BasicBlock(inplanes=256, planes=512, downsample=self.layer_3_downsample)
        self.layer_3_refine_right = BasicBlock(inplanes=512, planes=512)

        self.layer_2_downsample = create_downsample_path(128, 512)
        self.layer_2_refine_left = BasicBlock(inplanes=128, planes=512, downsample=self.layer_2_downsample)
        self.layer_2_refine_right = BasicBlock(inplanes=512, planes=512)

        self.layer_1_downsample = create_downsample_path(64, 512)
        self.layer_1_refine_left = BasicBlock(inplanes=64, planes=512, downsample=self.layer_1_downsample)
        self.layer_1_refine_right = BasicBlock(inplanes=512, planes=512)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.resnet50_8s = resnet50_8s

    def forward(self, x):

        # Get spatiall sizes
        input_height = x.size(2)
        input_width = x.size(3)

        # We don't gate the first stage of resnet
        x = self.resnet50_8s.conv1(x)
        x = self.resnet50_8s.bn1(x)
        x = self.resnet50_8s.relu(x)
        x = self.resnet50_8s.maxpool(x)

        layer1_output = self.resnet50_8s.layer1(x)
        layer2_output = self.resnet50_8s.layer2(layer1_output)
        layer3_output = self.resnet50_8s.layer3(layer2_output)
        layer4_output = self.resnet50_8s.layer4(layer3_output)

        global_pool = nn.functional.adaptive_avg_pool2d(layer4_output, 1)
        global_pool = nn.functional.upsample_bilinear(global_pool, size=layer4_output.size()[2:])
        layer_4_refined = self.layer_4_refine_right(self.layer_4_refine_left(layer4_output) + global_pool)

        layer_4_refined = nn.functional.upsample_bilinear(layer_4_refined, size=layer3_output.size()[2:])
        layer_3_refined = self.layer_3_refine_right(self.layer_3_refine_left(layer3_output) + layer_4_refined)

        layer_3_refined = nn.functional.upsample_bilinear(layer_3_refined, size=layer2_output.size()[2:])
        layer_2_refined = self.layer_2_refine_right(self.layer_2_refine_left(layer2_output) + layer_3_refined)

        layer_2_refined = nn.functional.upsample_bilinear(layer_2_refined, size=layer1_output.size()[2:])
        layer_1_refined = self.layer_1_refine_right(self.layer_1_refine_left(layer1_output) + layer_2_refined)

        logits = self.logit_conv(layer_1_refined)

        logits_upsampled = nn.functional.upsample_bilinear(logits,
                                                           size=(input_height, input_width))

        return logits_upsampled


def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)
















class ASPP(nn.Module):
    # Architecture Shtik: take input, pass it simulatenously through multiple 3x3 convolution layers, each with a different (large) dilation rate + a global pooling upsampled layer,
    # CONCATENATE resulting features, pass those through a convolution and predict
    #ASPP = Atrous Spatial Pyramid Pooling
    def __init__(self,
                 in_channels,
                 out_channels_per_branch=256,
                 branch_dilations=(6, 12, 18)):
        super(ASPP, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv_1x1 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels_per_branch,
                                  kernel_size=1,
                                  bias=False)

        self.conv_1x1_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)

        self.conv_3x3_first = conv3x3(in_channels, out_channels_per_branch, dilation=branch_dilations[0])
        self.conv_3x3_first_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)

        self.conv_3x3_second = conv3x3(in_channels, out_channels_per_branch, dilation=branch_dilations[1])
        self.conv_3x3_second_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)

        self.conv_3x3_third = conv3x3(in_channels, out_channels_per_branch, dilation=branch_dilations[2])
        self.conv_3x3_third_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)

        self.conv_1x1_pool = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels_per_branch,
                                       kernel_size=1,
                                       bias=False)
        self.conv_1x1_pool_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)

        self.conv_1x1_final = nn.Conv2d(in_channels=out_channels_per_branch * 5,
                                        out_channels=out_channels_per_branch,
                                        kernel_size=1,
                                        bias=False)
        self.conv_1x1_final_bn = torch.nn.BatchNorm2d(num_features=out_channels_per_branch)

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        conv_1x1_branch = self.relu(self.conv_1x1_bn(self.conv_1x1(x)))
        conv_3x3_first_branch = self.relu(self.conv_3x3_first_bn(self.conv_3x3_first(x)))
        conv_3x3_second_branch = self.relu(self.conv_3x3_second_bn(self.conv_3x3_second(x)))
        conv_3x3_third_branch = self.relu(self.conv_3x3_third_bn(self.conv_3x3_third(x)))

        global_pool_branch = self.relu(
            self.conv_1x1_pool_bn(self.conv_1x1_pool(nn.functional.adaptive_avg_pool2d(x, 1))))
        global_pool_branch = nn.functional.upsample_bilinear(input=global_pool_branch,
                                                             size=input_spatial_dim)

        features_concatenated = torch.cat([conv_1x1_branch,
                                           conv_3x3_first_branch,
                                           conv_3x3_second_branch,
                                           conv_3x3_third_branch,
                                           global_pool_branch],
                                          dim=1)

        features_fused = self.relu(self.conv_1x1_final_bn(self.conv_1x1_final(features_concatenated)))

        return features_fused





















class _DenseUpsamplingConvModule(nn.Module):
    #Regular conv-BN-relu and and use pixel_shuffle to make a learned upsampling.
    def __init__(self, down_factor, in_dim, num_classes):
        super(_DenseUpsamplingConvModule, self).__init__()
        upsample_dim = (down_factor ** 2) * num_classes
        self.conv = nn.Conv2d(in_dim, upsample_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(upsample_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(down_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class ResNetDUC(nn.Module):
    #DUC = Dense Upsampling Convolution
    # the size of image should be multiple of 8
    #Shtik - take a resnet and add a dense upsampling conv (DUC) module.
    def __init__(self, num_classes, pretrained=True):
        super(ResNetDUC, self).__init__()
        resnet = models.resnet152()
        if pretrained:
            resnet.load_state_dict(torch.load(res152_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (4, 4)
                m.padding = (4, 4)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.duc = _DenseUpsamplingConvModule(8, 2048, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.duc(x)
        return x


class ResNetDUCHDC(nn.Module):
    # DUC = Dense Upsampling Convolution
    # the size of image should be multiple of 8
    #Shtik  take a resnet model which is usually devided into 4 "layers" , and change the dilations (and padding) for the different layers to artificially expand perceptive field in the existing architecture.
    def __init__(self, num_classes, pretrained=True):
        super(ResNetDUCHDC, self).__init__()
        resnet = models.resnet152()
        if pretrained:
            resnet.load_state_dict(torch.load(res152_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        layer3_group_config = [1, 2, 5, 9]
        for idx in range(len(self.layer3)):
            self.layer3[idx].conv2.dilation = (layer3_group_config[idx % 4], layer3_group_config[idx % 4])
            self.layer3[idx].conv2.padding = (layer3_group_config[idx % 4], layer3_group_config[idx % 4])
        layer4_group_config = [5, 9, 17]
        for idx in range(len(self.layer4)):
            self.layer4[idx].conv2.dilation = (layer4_group_config[idx], layer4_group_config[idx])
            self.layer4[idx].conv2.padding = (layer4_group_config[idx], layer4_group_config[idx])

        self.duc = _DenseUpsamplingConvModule(8, 2048, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.duc(x)
        return x


















# many are borrowed from https://github.com/ycszen/pytorch-ss/blob/master/gcn.py
#Weird double convolution module (for instance...why the double kernel_size=(1,kernel_size[1])?....
class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) / 2
        pad1 = (kernel_size[1] - 1) / 2
        # kernel size had better be odd number so as to avoid alignment error (!!!!!)
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class GCN(nn.Module):
    #Architecture Shtik: the same as refine net as far as i can see except except that the convolution blocks are different and use seperable convolutions followed by 3x3 "boundary-refinemenet" convolutions
    def __init__(self, num_classes, input_size, pretrained=True):
        super(GCN, self).__init__()
        self.input_size = input_size
        resnet = models.resnet152()
        if pretrained:
            resnet.load_state_dict(torch.load(res152_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcm1 = _GlobalConvModule(2048, num_classes, (7, 7))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (7, 7))
        self.gcm3 = _GlobalConvModule(512, num_classes, (7, 7))
        self.gcm4 = _GlobalConvModule(256, num_classes, (7, 7))

        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)

        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self.brm1, self.brm2, self.brm3,
                           self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)

    def forward(self, x):
        # if x: 512
        fm0 = self.layer0(x)  # 256
        fm1 = self.layer1(fm0)  # 128
        fm2 = self.layer2(fm1)  # 64
        fm3 = self.layer3(fm2)  # 32
        fm4 = self.layer4(fm3)  # 16

        gcfm1 = self.brm1(self.gcm1(fm4))  # 16
        gcfm2 = self.brm2(self.gcm2(fm3))  # 32
        gcfm3 = self.brm3(self.gcm3(fm2))  # 64
        gcfm4 = self.brm4(self.gcm4(fm1))  # 128

        fs1 = self.brm5(F.upsample_bilinear(gcfm1, fm3.size()[2:]) + gcfm2)  # 32
        fs2 = self.brm6(F.upsample_bilinear(fs1, fm2.size()[2:]) + gcfm3)  # 64
        fs3 = self.brm7(F.upsample_bilinear(fs2, fm1.size()[2:]) + gcfm4)  # 128
        fs4 = self.brm8(F.upsample_bilinear(fs3, fm0.size()[2:]))  # 256
        out = self.brm9(F.upsample_bilinear(fs4, self.input_size))  # 512

        return out



















class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):
        if segSize is None: # training
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        else: # inference
            pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='resnet50_dilated8', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18_dilated8':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet18_dilated16':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50_dilated8':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet50_dilated16':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101_dilated8':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet101_dilated16':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    def build_decoder(self, arch='ppm_bilinear_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        if arch == 'c1_bilinear_deepsup':
            net_decoder = C1BilinearDeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1_bilinear':
            net_decoder = C1Bilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear':
            net_decoder = PPMBilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear_deepsup':
            net_decoder = PPMBilinearDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        elif arch == 'upernet_tmp':
            net_decoder = UPerNetTmp(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


# last conv, bilinear upsample
class C1BilinearDeepSup(nn.Module):
    #Architecture Shtik: the most stupid - get the final feature map, do a log softmax (as in log-likelihood), maybe use deep supervision on the one before last layer, but basically that's it...(doesn't even upsample when training)
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1BilinearDeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv, bilinear upsample
class C1Bilinear(nn.Module):
    #Architecture Shtik: same stupid thing as above- take the last layer, pass it through a log-softmax and that's it (don't even upsample when training)
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling, bilinear upsample
class PPMBilinear(nn.Module):
    #PPM - Pyramid Pooling Module(!!!)
    #Architecture Shtik: pass image into backbone network, now take the final output of the backbone (usually with 4096 channels), pass it through multiple large AdaptiveAvgPool2d+Conv+BN+ReLU layers of different pooling scales, concatenate them all
    #                    to try and get as much information from the different scales, concatenate it with the backbone output, pass through convolution block and log-softmax for prediction
    def __init__(self,
                 num_class=150,
                 fc_dim=4096, #fully convolutional output of backbone number of channels, which is usually 4096 with resnets
                 use_softmax=False,
                 pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax

        #Pyramid Pooling Module
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                                        nn.AdaptiveAvgPool2d(scale),
                                        nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                                        SynchronizedBatchNorm2d(512),
                                        nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512,  #Each Pyramid Pooling Module (PPM) "donates" 512 channels, plus we concatenate it with the feature extractor output, which has fc_dim
                      512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for current_ppm_module in self.ppm:
            ppm_out.append(nn.functional.upsample(
                current_ppm_module(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, bilinear upsample
class PPMBilinearDeepsup(nn.Module):
    #Architecture Shtik: the same as above: "#Architecture Shtik: pass image into backbone network, now take the final output of the backbone (usually with 4096 channels), pass it through multiple large AdaptiveAvgPool2d+Conv+BN+ReLU layers of different pooling scales, concatenate them all
    #                    to try and get as much information from the different scales, concatenate it with the backbone output, pass through convolution block and log-softmax for prediction"
    #(only with deep supervision)
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinearDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    #Architecture Shtik: basically expand the PPM thought process to the decoding part but apply it a bit differently there.
    #                    once we have the features from the backbone we pass it through a PPM module including the final conv of the concatenated Adaptive_Pooling+Conv layers, and we take that final "scale rich" feature maps, and start to
    #                    upsample and refine it in a module we call Feature Pyramid Network (or FPM). as i said we take those final scale reach feature maps, we take the final backbone feature maps (BEFORE PPM) and conv+Upsample it and ADD it to the PPM output.
    #                    then we take that output, and the one before last backbone feature map and conv+Upsample it and ADD it to the before result ETC' untill we've had enough.
    #                    we take EACH of the refined "results" of the FPN and pass it through a conv+Upsample(to final image prediction grid size) layer, Concatenate All of those, and pass all of that through another final conv layer which output a number of channels equaling number of classes

    # As far as i understand this is very similiar to UNET except that instead of horizontal concatenation we conv the left side and ADD it instead of concatenating like we do in UNET (so perhapse a generalization would be to both add and concatenate).
    # moreover we CONCATENATE all of the ADD results and Conv everything together to allow possibly for reacher scale information in the final result, unlike UNET which simply fuses (through concatenation) and upsamples and etc' like that.
    def __init__(self, num_class=150,
                 fc_dim=4096,
                 flag_use_softmax=False,
                 pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256,512,1024,2048),
                 fpn_dim=256):
        super(UPerNet, self).__init__()
        self.flag_use_softmax = flag_use_softmax

        #Pyramid Pooling Module (PPM Module):
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1) #After Concatenation


        #Feature Pyramid Network (FPN) Module:
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                                        nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                                        SynchronizedBatchNorm2d(fpn_dim),
                                        nn.ReLU(inplace=True)))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)): ##### reversed(range(10)) - what a cute trick!@$@%^
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.upsample(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch (TODO: how do i decide whether to use align_corners=True/False????)
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f)) #take the refined fused result into a conv layer

        fpn_feature_list.reverse() # [P2 - P5] - Another Cute Trick!@@%
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.upsample(
                                                    fpn_feature_list[i],
                                                    output_size,
                                                    mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.flag_use_softmax:  # is True during inference
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x


















##########################################################################################################################################################################################################################################
### Mobile-Net for fast on-mobile image segmentation!





def conv_bn(number_of_input_channels, number_of_output_channels, stride):
    return nn.Sequential(
        nn.Conv2d(number_of_input_channels, number_of_output_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(number_of_output_channels),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(number_of_input_channels, number_of_output_channels):
    return nn.Sequential(
        nn.Conv2d(number_of_input_channels, number_of_output_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(number_of_output_channels),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    #(*). Why INVERTED residual? what's inverted about it? - that unlike regular residual block, where number of channels is large-small-large we use smalr-large-small using depthwise convolutions
    def __init__(self, number_of_input_channels, number_of_output_channels, stride, expansion_factor_for_intermediate_layer):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        intermediate_number_of_channels = round(number_of_input_channels * expansion_factor_for_intermediate_layer)

        #If output size equals input size (channels, height & width!) then use residual connection (because you actually add the two):
        self.flag_use_residual_connection = self.stride == 1 and number_of_input_channels == number_of_output_channels

        if expansion_factor == 1:
            self.conv = nn.Sequential(
                # dw (dw=depthwise)
                #(*). Note: Using groups=number_of_output_channels makes this, effectively, a depthwise convolution. it seems that unlike TensorFlow, Pytorch doesn't have a dedicated block for it
                nn.Conv2d(number_of_input_channels, intermediate_number_of_channels, kernel_size=3, stride=stride, padding=1, groups=intermediate_number_of_channels, bias=False), #NOTE: i think there's a mistake here, first input should be number_of_input_channels
                nn.BatchNorm2d(intermediate_number_of_channels),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(intermediate_number_of_channels, number_of_output_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(number_of_output_channels))
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(number_of_input_channels, intermediate_number_of_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(intermediate_number_of_channels),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(intermediate_number_of_channels, intermediate_number_of_channels, kernel_size=3, stride=stride, padding=1, groups=intermediate_number_of_channels, bias=False),
                nn.BatchNorm2d(intermediate_number_of_channels),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(intermediate_number_of_channels, number_of_output_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(number_of_output_channels))

    def forward(self, x):
        if self.flag_use_residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    #I don't really know what's so "mobile" about it in the since that's there's no seperable convolutions and stuff... it's simple residual blocks with 3x3 convolutions.
    #The only thing fast about it is that instead of Pooling we're using Stride=2 (which is legit...) and that number of channels isn't all that large.
    #Also - at the end of the network instead of using all the feature pixels we use spatial global pooling and input only the final stage channels' averages into the Dense Prediction Layer...

    #By The Way... how is this semantic segmentation with a Dense Layer at the end.... this looks much more like a Classification Model!$#$@#
    #And Indead It is... this is used as a fast BackBone for a UNET fast architecture BELOW!!!
    def __init__(self, number_of_classes=1000, input_image_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        number_of_input_channels = 32
        final_number_of_channels = 1280
        interverted_residual_setting = [
            # t, c, n, s = (expansion_factor, number_of_output_channels, number_of_blocks, stride)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        #(1). Make sure input image size is a whole multiple of 32 (**** Why 32? - i see that number of input channels is 32...)
        assert input_image_size % 32 == 0
        number_of_input_channels = int(number_of_input_channels * width_mult)
        self.final_number_of_channels = int(final_number_of_channels * width_mult) if width_mult > 1.0 else final_number_of_channels
        self.features = [conv_bn(number_of_input_channels=3, number_of_output_channels=number_of_input_channels, stride=2)]

        # building inverted residual blocks
        previous_number_of_output_channels = number_of_input_channels;
        for current_expansion_factor, current_number_of_output_channels, current_number_of_residual_blocks, current_stride in interverted_residual_setting:
            current_number_of_output_channels = int(current_number_of_output_channels * width_mult)
            for i in range(current_number_of_residual_blocks):
                if i == 0:
                    self.features.append(InvertedResidual(previous_number_of_output_channels, current_number_of_output_channels, current_stride, expansion_factor_for_intermediate_layer=current_expansion_factor))
                else:
                    self.features.append(InvertedResidual(previous_number_of_output_channels, current_number_of_output_channels, 1, expansion_factor_for_intermediate_layer=current_expansion_factor))
                #Assign number_of_input_channels with previous number_of_output_channels:
                previous_number_of_output_channels = current_number_of_output_channels

        # building last several layers
        self.features.append(conv_1x1_bn(previous_number_of_output_channels, self.final_number_of_channels))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.final_number_of_channels, number_of_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2) #weird...after getting the last of the convoltional features i'm doing a global average pooling over all the spatial pixels...leaving just the channels--> [batch_size,number_of_channels,1,1] and sending that to the classifier
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()












import logging

class MobileNetV2_unet(nn.Module):
    #Architecture Shtik: use a "fast" MobileNetV2 BackBone as a feature pyramid of scales (with the downsampling effectively being done by strided convolution instead of max pooling),
    #then Take the final (most downsampled stage), use deconvolution (transposed convolution) to upsample it, CONCATENATE it with the higher resolution stage below it, Use InvertedResidual Layer (supposed to be faster than regular residual layer),
    #on the concatenation --> upsample result using deconvolution etc' untill we reach wanted resolution, and use convolution to get score maps.

    def __init__(self, pre_trained='weights/mobilenet_v2.pth.tar'):
        super(MobileNetV2_unet, self).__init__()

        ###
        #BackBone/DownScale-Pyramid/Feature Extractor:
        self.backbone = MobileNetV2()
        ###


        ###
        #UpScale-Pyramid/Feature-Fusion:
        #(*). Note: number of output channels of InvertedResidual is always X2 number of output channels of nn.ConvTranspose2d because we FUSE to pyramid layers by concatenation:
        #(*). Note: stride controls the UpSample Factor!
        #End Layer Fusion:
        self.dconv1 = nn.ConvTranspose2d(in_channels=1280, out_channels=96, kernel_size=4, padding=1, stride=2)
        self.invres1 = InvertedResidual(number_of_input_channels=192, number_of_output_channels=96, stride=1, expansion_factor_for_intermediate_layer=6)
        #End-1 Layer Fusion:
        self.dconv2 = nn.ConvTranspose2d(in_channels=96, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.invres2 = InvertedResidual(number_of_input_channels=64, number_of_output_channels=32, stride=1, expansion_factor_for_intermediate_layer=6)
        #End-2 Layer Fusion:
        self.dconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=24, kernel_size=4, padding=1, stride=2)
        self.invres3 = InvertedResidual(number_of_input_channels=48, number_of_output_channels=24, stride=1, expansion_factor_for_intermediate_layer=6)
        #End-3 Layer Fusion:
        self.dconv4 = nn.ConvTranspose2d(in_channels=24, out_channels=16, kernel_size=4, padding=1, stride=2)
        self.invres4 = InvertedResidual(number_of_input_channels=32, number_of_output_channels=16, stride=1, expansion_factor_for_intermediate_layer=6)
        ####


        #TODO: Generalize This By Allowing More Then One Segmentation Class (plus background):
        self.conv_last = nn.Conv2d(16, 3, kernel_size=1)
        self.conv_score = nn.Conv2d(3, 1, kernel_size=1) #here we have only 1 class so instead of multi channel final output with softmax we output one channel...not very general

        self._init_weights()

        if pre_trained is not None:
            self.backbone.load_state_dict(torch.load(pre_trained))

    def forward(self, x):
        ### Down Pyramid: ###
        #(*). Pyramid Layer 1:
        for n in range(0, 2):
            x = self.backbone.features[n](x)
        x1 = x
        logging.debug((x1.shape, 'x1'))
        #(*). Pyramid Layer 2:
        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x
        logging.debug((x2.shape, 'x2'))
        #(*). Pyramid Layer 3:
        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x
        logging.debug((x3.shape, 'x3'))
        #(*). Pyramid Layer 4:
        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x
        logging.debug((x4.shape, 'x4'))
        #(*). Pyramid Layer 5:
        for n in range(14, 19):
            x = self.backbone.features[n](x)
        x5 = x
        logging.debug((x5.shape, 'x5'))


        ### Up Pyramid: ###
        #(1). Deconvolve and Fuse (by concatenation) End,End-1 Pyramid Layers:
        up1 = torch.cat( [x4, self.dconv1(x5)] , dim=1)
        up1 = self.invres1(up1)
        logging.debug((up1.shape, 'up1'))
        #(2). Deconvolve and Fuse (by concatenation) End-1,End-2 Pyramid Layers:
        up2 = torch.cat( [x3, self.dconv2(up1)] , dim=1)
        up2 = self.invres2(up2)
        logging.debug((up2.shape, 'up2'))
        #(3). Deconvolve and Fuse (by concatenation) End-2,End-3 Pyramid Layers:
        up3 = torch.cat( [x2, self.dconv3(up2)], dim=1)
        up3 = self.invres3(up3)
        logging.debug((up3.shape, 'up3'))
        #(4). Deconvolve and Fuse (by concatenation) End-3,End-4 Pyramid Layers:
        up4 = torch.cat( [x1, self.dconv4(up3)], dim=1)
        up4 = self.invres4(up4)
        logging.debug((up4.shape, 'up4'))

        #(*). Pre-Final Convolution :
        x = self.conv_last(up4)
        logging.debug((x.shape, 'conv_last'))

        ### Final Convolution With Channels=Number_Of_Classes: ####
        x = self.conv_score(x)
        logging.debug((x.shape, 'conv_score'))

        # x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # logging.debug((x.shape, 'interpolate'))

        ### Use Sigmoid To Scale Outputs To Between [0,1] As In Probabilities: ###
        x = torch.sigmoid(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    # Debug
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = MobileNetV2_unet(pre_trained=None)
    net(torch.randn(1, 3, 224, 224))












def _init_unet(state_dict):
    unet = MobileNetV2_unet(pre_trained=None)
    unet.load_state_dict(state_dict)
    return unet


class ImgWrapNet(nn.Module):
    def __init__(self, state_dict, scale=255.):
        super().__init__()
        self.scale = scale
        self.unet = _init_unet(state_dict)

    def forward(self, x):
        x = x / self.scale
        x = self.unet(x)
        x = x * self.scale
        x = torch.cat((x, x, x), dim=1)
        return x


if __name__ == '__main__':
    WEIGHT_PATH = 'outputs/train_unet/0-best.pth'
    net = ImgWrapNet(torch.load(WEIGHT_PATH, map_location='cpu'))
    net(torch.randn(1, 3, 224, 224))


##################################################################################################################################################################################################################################################################################################################################################################################################################################################################











##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
### DeepLab3+ ResNet: ###
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class Bottleneck(nn.Module):
    #TODO: change name to more descriptive residual_131_block or something.....
    #Another kind of BottleNeck.... i really should have some place where i have the different bottlenecks sorted out.
    #This kind of bottleneck is a 131 bottleneck with activation ONLY at the end of the bottleneck (why???....)
    channel_expansion_factor = 4 #Totally a hyper parameter to be passed into the __init__ function!

    #(*). Note: this is, again, a Residual Block... this imposes (as can be seen in the implementation below) that number_of_input_channels=number_of_intermediate_channels*channel_expansion_factor.
    def __init__(self, number_of_input_channels, number_of_intermediate_channels, stride=1, dilation_and_padding=1, Channel_Projector_Block=None):
        #This is Kind of stupid - insert channel expansion factor into the __init__ function (maybe even as a standard number_of_output_channels...)
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(number_of_input_channels, number_of_intermediate_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(number_of_intermediate_channels)
        self.conv2 = nn.Conv2d(number_of_intermediate_channels, number_of_intermediate_channels, kernel_size=3, stride=stride, dilation=dilation_and_padding, padding=dilation_and_padding, bias=False)
        self.bn2 = nn.BatchNorm2d(number_of_intermediate_channels)
        self.conv3 = nn.Conv2d(number_of_intermediate_channels, number_of_intermediate_channels * channel_expansion_factor, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(number_of_intermediate_channels * channel_expansion_factor) #Note here! - we implicitly must have number_of_input_channels = number_of_intermediate_channels*channel_expansion_factor!!!!
        self.relu = nn.ReLU(inplace=True)
        self.Channel_Projector_Block = Channel_Projector_Block
        self.stride = stride
        self.dilation_and_padding = dilation_and_padding

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.Channel_Projector_Block is not None: #Notice that conditions for the necessity of channel_projector_block ARE NOT TESTED HERE...it is expected from the user to do so and make sure to implement if needed... not the brightessed idea every time
            residual = self.Channel_Projector_Block(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, number_of_input_channels, Basic_Block, number_of_basic_blocks_per_BlockBatch, final_output_stride=16, flag_pretrained=False):
        self.number_of_input_channels_Basic_Block = 64
        super(ResNet, self).__init__()
        if final_output_stride == 16:
            stride_per_BlockBatch = [1, 2, 2, 1]
            dilation_and_padding_per_BlockBatch = [1, 1, 1, 2]
            dilation_and_padding_per_MG_sub_block = [1, 2, 4]
        elif final_output_stride == 8:
            stride_per_BlockBatch = [1, 2, 1, 1]
            dilation_and_padding_per_BlockBatch = [1, 1, 2, 2]
            dilation_and_padding_per_MG_sub_block = [1, 2, 1]
        else:
            raise NotImplementedError

        #Input/Initial Layers:
        self.conv1 = nn.Conv2d(number_of_input_channels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Block Batches:
        self.layer1 = self.make_Block_Batch(Basic_Block, 64, number_of_basic_blocks_per_BlockBatch[0], stride=stride_per_BlockBatch[0], dilation_and_padding=dilation_and_padding_per_BlockBatch[0])
        self.layer2 = self.make_Block_Batch(Basic_Block, 128, number_of_basic_blocks_per_BlockBatch[1], stride=stride_per_BlockBatch[1], dilation_and_padding=dilation_and_padding_per_BlockBatch[1])
        self.layer3 = self.make_Block_Batch(Basic_Block, 256, number_of_basic_blocks_per_BlockBatch[2], stride=stride_per_BlockBatch[2], dilation_and_padding=dilation_and_padding_per_BlockBatch[2])
        self.layer4 = self._make_MG_unit(Basic_Block, 512, dilation_and_padding_per_MG_sub_block=dilation_and_padding_per_MG_sub_block, stride=number_of_basic_blocks_per_BlockBatch[3], dilation_and_padding=dilation_and_padding_per_BlockBatch[3])

        self._init_weight()

        if flag_pretrained:
            self._load_pretrained_model()


    #Make Block_Batch - basically pile on top of each other multiple Basic Blocks:
    def make_Block_Batch(self, Basic_Block, number_of_intermediate_channels, number_of_blocks, stride=1, dilation_and_padding=1):
        Channel_Projector_Block = None
        #(*). Channel & Size Matching Projection Layer 1x1 conv (if there's a channels mismatch):
        #TODO: it's kind of stupid to make the projector block here.... a much better way to avoid mistakes is to implement it INSIDE the Basic_Block (and maybe throw a warning if condition is not met in advance that we've created a projector block)
        if stride != 1 or self.number_of_input_channels_Basic_Block != number_of_intermediate_channels * Basic_Block.channel_expansion_factor:
            Channel_Projector_Block = nn.Sequential(
                                       nn.Conv2d(self.number_of_input_channels_Basic_Block, number_of_intermediate_channels * Basic_Block.channel_expansion_factor, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(number_of_intermediate_channels * Basic_Block.expansion)
                                     )
        #(*). Start Appending the Basic Blocks on top of each other:
        #(1). Initial Block With Channel Project Block if one is needed:
        layers = []
        layers.append(Basic_Block(self.number_of_input_channels_Basic_Block, number_of_intermediate_channels, stride, dilation_and_padding, Channel_Projector_Block))
        #(2). Equalize number_of_input_channels to number_of_intermediate_channels*channel_expansion_factor to meet the constant number of channels residual block condition:
        self.number_of_input_channels_Basic_Block = number_of_intermediate_channels * Basic_Block.channel_expansion_factor
        #(3). Pile Basic Blocks on top of each other:
        for i in range(1, number_of_blocks):
            layers.append(Basic_Block(self.number_of_input_channels_Basic_Block, number_of_intermediate_channels)) #(*). Note!: again...basic block is a residual block so number of output channels equals number of input channels....so we only specify number_of_intermediate_channels
        #Return Block Batch consisting of multiple basic blocks on top of each other
        return nn.Sequential(*layers)


    #Make MG (Multi G.....???) Block:
    def _make_MG_unit(self, Basic_Block, number_of_intermediate_channels, dilation_and_padding_per_MG_sub_block=[1,2,4], stride=1, dilation_and_padding=1):
        #Shtik: same concept as implemented in make_Block_Batch except that there's something special here - instead of repetitive blocks, each sequential block doubles the dilation rate
        Channel_Projector_Block = None
        # (*). Channel Matching Projection Layer 1x1 conv (if there's a channels mismatch):
        if stride != 1 or self.number_of_input_channels_Basic_Block != number_of_intermediate_channels * Basic_Block.channel_expansion_factor:
            Channel_Projector_Block = nn.Sequential(
                                                nn.Conv2d(self.number_of_input_channels_Basic_Block, number_of_intermediate_channels * Basic_Block.channel_expansion_factor, kernel_size=1, stride=stride, bias=False),
                                                nn.BatchNorm2d(number_of_intermediate_channels * Basic_Block.channel_expansion_factor),
                                            )
        # (*). Start Appending the Basic Blocks on top of each other:
        layers = []
        layers.append(Basic_Block(self.number_of_input_channels_Basic_Block, number_of_intermediate_channels, stride, dilation_and_padding=dilation_and_padding_per_MG_sub_block[0]*dilation_and_padding, Channel_Projector_Block=Channel_Projector_Block))
        self.number_of_input_channels_Basic_Block = number_of_intermediate_channels * Basic_Block.channel_expansion_factor
        for i in range(1, len(dilation_and_padding_per_MG_sub_block)):
            layers.append(Basic_Block(self.number_of_input_channels_Basic_Block, number_of_intermediate_channels, stride=1, dilation_and_padding=dilation_and_padding_per_MG_sub_block[i]*dilation_and_padding))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat


    #TODO: get on top of initialization strategies if there are diverse ones for BatchNorm:
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #TODO: Make Sure i've already implemented this:
    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)




#Use Basic ResNet Creation Block to create a ResNet101 with the same overall structure as the known resnet101 but with out shtik
#TODO: understand if there's actually any difference between the standard resnet101 and the above defined one
def ResNet101(number_of_input_channels=3, final_output_stride=16, flag_pretrained=False):
    model = ResNet(number_of_input_channels=number_of_input_channels,
                   Basic_Block=Bottleneck,
                   number_of_basic_blocks_per_BlockBatch=[3, 4, 23, 3],
                   final_output_stride=final_output_stride,
                   flag_pretrained=flag_pretrained)
    return model



#Atrous Spatial Pyramid Pooling (ASPP):
class ASPP_Basic_Branch(nn.Module):
    #TODO: implement ASPP module which gets all the dilation rates of the different branches all at once and creates the complete ASPP module
    #Simple Standard Conv Block Except that it's Attrous
    def __init__(self, number_of_input_channels, number_of_output_channels, dilation):
        super(ASPP_Basic_Branch, self).__init__()
        if dilation_and_padding == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation_and_padding
        self.atrous_convolution = nn.Conv2d(number_of_input_channels, number_of_output_channels, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(number_of_output_channels)
        self.relu = nn.ReLU()

        #Initialize the module layer weights according to our policy:
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()





class DeepLabv3_plus(nn.Module):
    #Architecture Shtik: pretty much a basic backbone ResNet101, which is then inserted into a ASPP module to get multi scale information and convolved again.
    #                    then we fuse it with the higher resolution features of the backbone for more high res info and output the predictions.
    #(*).Note: we can again see three basic approaches to gain multi scale info:
    #          (1). ASPP - Take final layer of backbone and pass it through something like an ASPP module to extend to reception field at the final (very downsampled) feature layer of the backbone:
    #          (2). PPM (pyramid pooling module) - same shtik at the end except that instead of different dilation rates it's different AdaptiveAvgPool2d scales
    #          (3). SPP (Scene Pyramid Parsing) - Combine Information from the different layers of the backbone features by concatenation or addition/refinement
    #          (4). UNET - whether with regular conv blocks, or residual or dense or stacked-dense or dense-in-dense-residual...the UNET structure is by definition multi-scale
    #          (5). UperNet/FPM(feature pyramid network) - very similar to UNET except that before fusion we do some convolutions and the fusion is mainly ADDITION BASED
    def __init__(self, number_of_input_channels=3, number_of_classes=21, final_output_stride=16, flag_pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(number_of_classes))
            print("Output stride: {}".format(final_output_stride))
            print("Number of Input Channels: {}".format(number_of_input_channels))
        super(DeepLabv3_plus, self).__init__()

        #BackBone:
        self.resnet_features = ResNet101(number_of_input_channels, final_output_stride, flag_pretrained=flag_pretrained)


        #ASPP Module
        if final_output_stride == 16:
            dilation_and_padding_per_BlockBatch = [1, 6, 12, 18]
        elif final_output_stride == 8:
            dilation_and_padding_per_BlockBatch = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = ASPP_Basic_Branch(2048, 256, dilation_and_padding=dilation_and_padding_per_BlockBatch[0])
        self.aspp2 = ASPP_Basic_Branch(2048, 256, dilation_and_padding=dilation_and_padding_per_BlockBatch[1])
        self.aspp3 = ASPP_Basic_Branch(2048, 256, dilation_and_padding=dilation_and_padding_per_BlockBatch[2])
        self.aspp4 = ASPP_Basic_Branch(2048, 256, dilation_and_padding=dilation_and_padding_per_BlockBatch[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, number_of_classes, kernel_size=1, stride=1))

    def forward(self, input):
        #Get features from BackBone:
        x, low_level_features = self.resnet_features(input)

        #Get the different branchs of the ASPP module:
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.upsample(self.global_avg_pool(x), size=x4.size()[2:], mode='bilinear', align_corners=True)
        #Concatenate the different branches of the ASPP module:
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        #Standard Conv Block + Upssampling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x,
                       size=(int(math.ceil(input.size()[-2]/4)), int(math.ceil(input.size()[-1]/4))),
                       mode='bilinear',
                       align_corners=True)

        #Get information from the low level (High Resolution) features extracted by the BackBone:
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        #Fuse (by Concatenation) the high level and low level features, convolve, and upsample to input image size to predict pixelwise:
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


    #Is this an OverRide of an existing method????
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    model = DeepLabv3_plus(number_of_input_channels=3, number_of_classes=21, final_output_stride=16, flag_pretrained=True, _print=True)
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())













###############################################################################################################################################################################################################################################################
### ENET: ###
import torch.nn as nn
import torch
from torch.autograd import Variable


class InitialBlock(nn.Module):
    """The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.
    Doing both operations in parallel and concatenating their results
    allows for efficient downsampling and expansion. The main branch
    outputs 13 feature maps while the extension branch outputs 3, for a
    total of 16 feature maps after concatenation.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number output channels.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=0,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - As stated above the number of output channels for this
        # branch is the total minus 3, since the remaining channels come from
        # the extension branch
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_prelu(out)


class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.
    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - channels (int): the number of input and output channels.
    - internal_ratio (int, optional): a scale factor applied to
    ``channels`` used to compute the number of
    channels after the projection. eg. given ``channels`` equal to 128 and
    internal_ratio equal to 2 the number of channels after the projection
    is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension
    branch. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout). Number of channels is constant.

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation,
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation)
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_prelu = activation

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out)


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.
    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.
    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``channels``
    used to compute the number of channels after the projection. eg. given
    ``channels`` equal to 128 and internal_ratio equal to 2 the number of
    channels after the projection is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension branch.
    Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    - asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - return_indices (bool, optional):  if ``True``, will return the max
    indices along with the outputs. Useful when unpooling later.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            kernel_size,
            stride=2,
            padding=padding,
            return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = Variable(torch.zeros(n, ch_ext - ch_main, h, w))

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.
    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.
    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``in_channels``
     used to compute the number of channels after the projection. eg. given
     ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
     of channels after the projection is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in the
    convolution layer described above in item 2 of the extension branch.
    Default: 3.
    - padding (int, optional): zero-padding added to both sides of the input.
    Default: 0.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation)

        # Transposed convolution
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x, max_indices):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out)


class ENet(nn.Module):
    """Generate the ENet model.
    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.
    """

    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()

        self.initial_block = InitialBlock(3, 16, padding=1, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            16,
            64,
            padding=1,
            return_indices=True,
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            64,
            128,
            padding=1,
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)

        return x

### END OF ENET ####
#############################################################################################################################################################################################################################
















################################################################################################################################################################################################################################
##### Stacked UNET: #####

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ..loss import cross_entropy2d, prediction_stat, prediction_stat_confusion_matrix

from torch.autograd import Variable

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import models
import os
from itertools import chain
from ..loss import cross_entropy2d, prediction_stat, prediction_stat_confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix


checkpoint = 'pretrained/ResNet'
res18_path = os.path.join(checkpoint, 'resnet18-5c106cde.pth')
res101_path = os.path.join(checkpoint, 'resnet101-5d3b4d8f.pth')

mom_bn = 0.05
dilation = {'16':1, '8':2}

class d_resnet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, use_aux=True, ignore_index=-1, output_stride='16'):
        super(d_resnet18, self).__init__()
        self.use_aux = use_aux
        self.num_classes = num_classes
        resnet = models.resnet18()
        if pretrained:
            resnet.load_state_dict(torch.load(res18_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        d = dilation[output_stride]
        if d > 1:
            for n, m in self.layer3.named_modules():
                if '0.conv1' in n:
                    m.dilation, m.padding, m.stride = (1, 1), (1, 1), (1, 1)
                elif 'conv1' in n:
                    m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if '0.conv1' in n:
                m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
            elif 'conv1' in n:
                m.dilation, m.padding, m.stride = (2*d, 2*d), (2*d, 2*d), (1, 1)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (2*d, 2*d), (2*d, 2*d), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        for n, m in chain(self.layer0.named_modules(), self.layer1.named_modules(), self.layer2.named_modules(), self.layer3.named_modules(), self.layer4.named_modules()):
            if 'downsample.1' in n:
                m.momentum = mom_bn
            elif 'bn' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=mom_bn),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        self.mceloss = cross_entropy2d(ignore=ignore_index, size_average=False)

    def forward(self, x, labels, th=1.0):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        x = F.upsample(x, x_size[2:], mode='bilinear')

        if labels is not None:
            losses, total_valid_pixel = self.mceloss(x, labels, th=th)

            classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([x], labels, self.num_classes)

            # Need to perform this operation for MultiGPU
            classwise_pixel_acc = Variable(torch.FloatTensor([classwise_pixel_acc]).cuda())
            classwise_gtpixels = Variable(torch.FloatTensor([classwise_gtpixels]).cuda())
            classwise_predpixels = Variable(torch.FloatTensor([classwise_predpixels]).cuda())

            return x, losses, classwise_pixel_acc, classwise_gtpixels, classwise_predpixels, total_valid_pixel
        else:
            return x


class d_resnet101(nn.Module):
    def __init__(self, num_classes, pretrained=True, use_aux=True, ignore_index=-1, output_stride='16'):
        super(d_resnet101, self).__init__()
        self.use_aux = use_aux
        self.num_classes = num_classes
        resnet = models.resnet101()
        if pretrained:
            resnet.load_state_dict(torch.load(res101_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        d = dilation[output_stride]
        if d > 1:
            for n, m in self.layer3.named_modules():
                if '0.conv2' in n:
                    m.dilation, m.padding, m.stride = (1, 1), (1, 1), (1, 1)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if '0.conv2' in n:
                m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (2*d, 2*d), (2*d, 2*d), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        for n, m in chain(self.layer0.named_modules(), self.layer1.named_modules(), self.layer2.named_modules(), self.layer3.named_modules(), self.layer4.named_modules()):
            if 'downsample.1' in n:
                m.momentum = mom_bn
            elif 'bn' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=mom_bn),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        self.mceloss = cross_entropy2d(ignore=ignore_index)

    def forward(self, x, labels, th=1.0):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        x = F.upsample(x, x_size[2:], mode='bilinear')

        if labels is not None:
            losses, total_valid_pixel = self.mceloss(x, labels, th=th)

            classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([x], labels, self.num_classes)

            # Need to perform this operation for MultiGPU
            classwise_pixel_acc = Variable(torch.FloatTensor([classwise_pixel_acc]).cuda())
            classwise_gtpixels = Variable(torch.FloatTensor([classwise_gtpixels]).cuda())
            classwise_predpixels = Variable(torch.FloatTensor([classwise_predpixels]).cuda())

            return x, losses, classwise_pixel_acc, classwise_gtpixels, classwise_predpixels, total_valid_pixel
        else:
            return x






class cross_entropy2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore=-100):
        super(cross_entropy2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, size_average=size_average, ignore_index=ignore)
        self.ignore = ignore

    def forward(self, input, target, th=1.0):
        log_p = F.log_softmax(input, dim=1)
        if th < 1: # This is done while using Hardmining. Not used for our model training
            mask = F.softmax(input, dim=1) > th
            mask = mask.data
            new_target = target.data.clone()
            new_target[new_target == self.ignore] = 0
            indx = torch.gather(mask, 1, new_target.unsqueeze(1))
            indx = indx.squeeze(1)
            mod_target = target.clone()
            mod_target[indx] = self.ignore
            target = mod_target

        loss = self.nll_loss(log_p, target)
        total_valid_pixel = torch.sum(target.data != self.ignore)

        return loss, Variable(torch.FloatTensor([total_valid_pixel]).cuda())


def pixel_accuracy(outputs, labels, n_classes):
    lbl = labels.data
    mask = lbl < n_classes

    accuracy = []
    for output in outputs:
        _, pred = output.data.max(dim=1)
        diff = pred[mask] - lbl[mask]
        accuracy += [torch.sum(diff == 0)]

    return accuracy

def prediction_stat(outputs, labels, n_classes):
    lbl = labels.data
    valid = lbl < n_classes

    classwise_pixel_acc = []
    classwise_gtpixels = []
    classwise_predpixels = []
    for output in outputs:
        _, pred = output.data.max(dim=1)
        for m in range(n_classes):
            mask1 = lbl == m
            mask2 = pred[valid] == m
            diff = pred[mask1] - lbl[mask1]
            classwise_pixel_acc += [torch.sum(diff == 0)]
            classwise_gtpixels += [torch.sum(mask1)]
            classwise_predpixels += [torch.sum(mask2)]

    return classwise_pixel_acc, classwise_gtpixels, classwise_predpixels

def prediction_stat_confusion_matrix(logits, annotation, n_classes):
    labels = range(n_classes)

    # First we do argmax on gpu and then transfer it to cpu
    logits = logits.data
    annotation = annotation.data
    _, prediction = logits.max(1)
    prediction = prediction.squeeze(1)

    prediction_np = prediction.cpu().numpy().flatten()
    annotation_np = annotation.cpu().numpy().flatten()

    # Mask-out value is ignored by default in the sklearn
    # read sources to see how that was handled
    current_confusion_matrix = confusion_matrix(y_true=annotation_np,
                                                y_pred=prediction_np,
                                                labels=labels)

    return current_confusion_matrix






checkpoint = 'pretrained/SUNets'
sunet64_path = os.path.join(os.path.dirname(__file__), checkpoint, 'checkpoint_64_2441_residual.pth.tar')
sunet128_path = os.path.join(os.path.dirname(__file__), checkpoint, 'checkpoint_128_2441_residual.pth.tar')
sunet7128_path = os.path.join(os.path.dirname(__file__), checkpoint, 'checkpoint_128_2771_residual.pth.tar')

mom_bn = 0.01
output_stride = {'32':3, '16':2, '8':1}

class UNetConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, dprob, mod_in_planes, is_input_bn, dilation):
        super(UNetConv, self).__init__()
        if mod_in_planes:
            if is_input_bn:
                self.add_module('bn0', nn.BatchNorm2d(in_planes))
                self.add_module('relu0', nn.ReLU(inplace=True))
            self.add_module('conv0', nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True))
            self.add_module('dropout0', nn.Dropout(p=dprob))
            in_planes = out_planes

        self.add_module('bn1', nn.BatchNorm2d(in_planes))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if dilation > 1:
            self.add_module('conv1', nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=True))
        else:
            self.add_module('conv1', nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=2, bias=True))
        self.add_module('dropout1', nn.Dropout(p=dprob))
        self.add_module('bn2', nn.BatchNorm2d(out_planes))
        self.add_module('relu2', nn.ReLU(inplace=True))
        if dilation > 1:
            self.add_module('conv2', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=2*dilation, stride=1, dilation=2*dilation, bias=True))
        else:
            self.add_module('conv2', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=True))
        self.add_module('dropout2', nn.Dropout(p=dprob))


class UNetDeConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, dprob, mod_in_planes, max_planes, dilation, output_padding=1):
        super(UNetDeConv, self).__init__()
        self.add_module('bn0', nn.BatchNorm2d(in_planes))
        self.add_module('relu0', nn.ReLU(inplace=True))
        if dilation > 1:
            self.add_module('deconv0',nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=1, padding=2 * dilation,
                                                         dilation=2 * dilation, bias=True))
        else:
            self.add_module('deconv0',nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1,
                                                         output_padding=output_padding, bias=True))
        self.add_module('dropout0', nn.Dropout(p=dprob))
        self.add_module('bn1', nn.BatchNorm2d(out_planes))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if dilation > 1:
            self.add_module('deconv1', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=True))
        else:
            self.add_module('deconv1', nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=True))
        self.add_module('dropout1', nn.Dropout(p=dprob))
        if mod_in_planes:
            self.add_module('bn2', nn.BatchNorm2d(out_planes))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(out_planes, max_planes, kernel_size=1, bias=True))
            self.add_module('dropout2', nn.Dropout(p=dprob))


class UNetModule(nn.Module):
    def __init__(self, in_planes, nblock, filter_size, dprob, in_dim, index, max_planes, atrous=0):
        super(UNetModule, self).__init__()
        self.nblock = nblock
        self.in_dim = np.array(in_dim, dtype=float)
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        self.upsample = None
        if in_planes != max_planes:
            self.bn = nn.Sequential(OrderedDict([
                ('bn0', nn.BatchNorm2d(in_planes)),
                ('relu0', nn.ReLU(inplace=True))
            ]))
            self.upsample = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_planes, max_planes, kernel_size=1, stride=1, bias=True))
            ]))
        for i in range(nblock):
            if i == 0:
                in_ = in_planes
            else:
                in_ = filter_size
            self.down.append(UNetConv(in_, filter_size, dprob, index and (i == 0), in_planes == max_planes, (2**i)*atrous))
            if i > 1:
                self.down[-1].conv.weight = self.conv_1.conv.weight
                self.down[-1].conv1.weight = self.conv_1.conv1.weight
            if i == nblock-1:
                out_ = filter_size
            else:
                out_ = 2 * filter_size
            self.up.append(UNetDeConv(out_, filter_size, dprob, index and (i == 0), max_planes, (2**i)*atrous, output_padding=1-int(np.mod(self.in_dim,2))))
            if i > 0 and i < nblock-1:
                self.up[-1].deconv.weight = self.deconv_0.deconv.weight
                self.up[-1].deconv1.weight = self.deconv_0.deconv1.weight
            self.in_dim = np.ceil(self.in_dim / 2)

    def forward(self, x):
        xs = []
        if self.upsample is not None:
            x = self.bn(x)
        xs.append(x)
        for i, down in enumerate(self.down):
            xout = down(xs[-1])
            xs.append(xout)

        out = xs[-1]
        for i, (x_skip, up) in reversed(list(enumerate(zip(xs[:-1], self.up)))):
            out = up(out)
            if i:
                out = torch.cat([out, x_skip], 1)
            else:
                if self.upsample is not None:
                    x_skip = self.upsample(x_skip)
                out += x_skip
        return out


class Transition(nn.Sequential):
    def __init__(self, in_planes, out_planes, dprob):
        super(Transition, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_planes))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True))
        self.add_module('dropout', nn.Dropout(p=dprob))


class ResidualBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, dprob, stride=1):
        super(ResidualBlock, self).__init__()
        self.bn = nn.Sequential(nn.BatchNorm2d(in_planes),
                                 nn.ReLU(inplace=True))
        self.conv = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=True),
                                  nn.Dropout(p=dprob),
                                  nn.BatchNorm2d(out_planes),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=True),
                                  nn.Dropout(p=dprob))
        if stride > 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes,out_planes, kernel_size=1, stride=stride, bias=True))
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.bn(x)
        residual = out
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        out = self.conv(out)
        out += residual
        return out


class Stackedunet_imagenet(nn.Module):
    filter_factors = [1, 1, 1, 1]

    def __init__(self, in_dim=224, start_planes=16, filters_base=64, num_classes=1000, depth=1, dprob=1e-7, ost='32'):
        super(Stackedunet_imagenet, self).__init__()
        self.start_planes = start_planes
        self.depth = depth
        feature_map_sizes = [filters_base * s for s in self.filter_factors]

        if filters_base == 128 and depth == 4:
            output_features = [512, 1024, 1536, 2048]
        elif filters_base == 128 and depth == 7:
            output_features = [512, 1280, 2048, 2304]
        elif filters_base == 64 and depth == 4:
            output_features = [256, 512, 768, 1024]

        num_planes = start_planes
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3, bias=True)),
        ]))
        in_dim = in_dim // 2
        self.features.add_module('residual1', ResidualBlock(num_planes, 2*num_planes, dprob, stride=2))
        num_planes *= 2
        in_dim = in_dim // 2

        block_depth = (2, depth, depth, 1)
        nblocks = 2
        for j, d in enumerate(block_depth):
            if j == len(block_depth)-1:
                nblocks = 1
            for i in range(d):
                block = UNetModule(num_planes, nblocks, feature_map_sizes[j], dprob, in_dim, 1, output_features[j], (j-output_stride[ost])*2)
                self.features.add_module('unet%d_%d'% (j+1, i), block)
                num_planes = output_features[j]
            if j != len(block_depth)-1:
                if j > output_stride[ost]-1:
                    self.features.add_module('avgpool%d' % (j + 1), nn.AvgPool2d(kernel_size=1, stride=1))
                else:
                    self.features.add_module('avgpool%d'%(j+1),nn.AvgPool2d(kernel_size=2, stride=2))
                    in_dim = in_dim // 2

        self.features.add_module('bn2', nn.BatchNorm2d(num_planes))
        self.linear = nn.Linear(num_planes, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = F.relu(out, inplace=False)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def stackedunet7128(output_stride='32'):
    return Stackedunet_imagenet(in_dim=512, start_planes=64, filters_base=128, num_classes=1000, depth=7, ost=output_stride)


def stackedunet128(output_stride='32'):
    return Stackedunet_imagenet(in_dim=512, start_planes=64, filters_base=128, num_classes=1000, depth=4, ost=output_stride)


def stackedunet64(output_stride='32'):
    return Stackedunet_imagenet(in_dim=512, start_planes=64, filters_base=64, num_classes=1000, depth=4, ost=output_stride)

# Dilated SUNET models
class d_sunet64(nn.Module):
    def __init__(self, num_classes, pretrained=True, ignore_index=-1, weight=None, output_stride='16'):
        super(d_sunet64, self).__init__()
        self.num_classes = num_classes
        sunet = stackedunet64(output_stride=output_stride)
        sunet = torch.nn.DataParallel(sunet, device_ids=range(torch.cuda.device_count())).cuda()
        if pretrained:
            checkpoint = torch.load(sunet64_path)
            sunet.load_state_dict(checkpoint['state_dict'])

        self.features = sunet.module.features

        for n, m in self.features.named_modules():
            if 'bn' in n:
                m.momentum = mom_bn

        # hack needed as batchnorm modules were not named as bn in residual block
        for n, m in self.features.residual1.conv.named_modules():
            if '2' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1024, num_classes, kernel_size=1)),
        ]))

        self.mceloss = cross_entropy2d(ignore=ignore_index, size_average=False, weight=weight)

    def forward(self, x, labels=None, th=1.0):
        x_size = x.size()
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = self.final(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)

        if labels is not None:
            losses, total_valid_pixel = self.mceloss(x, labels, th=th)

            classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([x], labels, self.num_classes)

            # Need to perform this operation for MultiGPU
            classwise_pixel_acc = Variable(torch.FloatTensor([classwise_pixel_acc]).cuda())
            classwise_gtpixels = Variable(torch.FloatTensor([classwise_gtpixels]).cuda())
            classwise_predpixels = Variable(torch.FloatTensor([classwise_predpixels]).cuda())

            return x, losses, classwise_pixel_acc, classwise_gtpixels, classwise_predpixels, total_valid_pixel
        else:
            return x

class d_sunet128(nn.Module):
    def __init__(self, num_classes, pretrained=True, ignore_index=-1, weight=None, output_stride='16'):
        super(d_sunet128, self).__init__()
        self.num_classes = num_classes
        sunet = stackedunet128(output_stride=output_stride)
        sunet = torch.nn.DataParallel(sunet, device_ids=range(torch.cuda.device_count())).cuda()
        if pretrained:
            checkpoint = torch.load(sunet128_path)
            sunet.load_state_dict(checkpoint['state_dict'])

        self.features = sunet.module.features

        for n, m in self.features.named_modules():
            if 'bn' in n:
                m.momentum = mom_bn

        # hack needed as batchnorm modules were not named as bn in residual block
        for n, m in self.features.residual1.conv.named_modules():
            if '2' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(2048, num_classes, kernel_size=1)),
        ]))

        self.mceloss = cross_entropy2d(ignore=ignore_index, size_average=False, weight=weight)

    def forward(self, x, labels=None, th=1.0):
        x_size = x.size()
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = self.final(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)

        if labels is not None:
            losses, total_valid_pixel = self.mceloss(x, labels, th=th)

            classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([x], labels, self.num_classes)

            # Need to perform this operation for MultiGPU
            classwise_pixel_acc = Variable(torch.FloatTensor([classwise_pixel_acc]).cuda())
            classwise_gtpixels = Variable(torch.FloatTensor([classwise_gtpixels]).cuda())
            classwise_predpixels = Variable(torch.FloatTensor([classwise_predpixels]).cuda())

            return x, losses, classwise_pixel_acc, classwise_gtpixels, classwise_predpixels, total_valid_pixel
        else:
            return x

class d_sunet7128(nn.Module):
    def __init__(self, num_classes, pretrained=True, ignore_index=-1, weight=None, output_stride='16'):
        super(d_sunet7128, self).__init__()
        self.num_classes = num_classes
        sunet = stackedunet7128(output_stride=output_stride)
        sunet = torch.nn.DataParallel(sunet, device_ids=range(torch.cuda.device_count())).cuda()
        if pretrained:
            checkpoint = torch.load(sunet7128_path)
            sunet.load_state_dict(checkpoint['state_dict'])

        self.features = sunet.module.features

        for n, m in self.features.named_modules():
            if 'bn' in n:
                m.momentum = mom_bn

        # hack needed as batchnorm modules were not named as bn in residual block
        for n, m in self.features.residual1.conv.named_modules():
            if '2' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(2304, num_classes, kernel_size=1))
        ]))

        self.mceloss = cross_entropy2d(ignore=ignore_index, size_average=False, weight=weight)

    def forward(self, x, labels=None, th=1.0):
        x_size = x.size()
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = self.final(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)

        if labels is not None:
            losses, total_valid_pixel = self.mceloss(x, labels, th=th)

            classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([x], labels, self.num_classes)

            # Need to perform this operation for MultiGPU
            classwise_pixel_acc = Variable(torch.FloatTensor([classwise_pixel_acc]).cuda())
            classwise_gtpixels = Variable(torch.FloatTensor([classwise_gtpixels]).cuda())
            classwise_predpixels = Variable(torch.FloatTensor([classwise_predpixels]).cuda())

            return x, losses, classwise_pixel_acc, classwise_gtpixels, classwise_predpixels, total_valid_pixel
        else:
            return x

# With degridding filters
class degrid_sunet7128(nn.Module):
    def __init__(self, num_classes, pretrained=True, ignore_index=-1, weight=None, output_stride='8'):
        super(degrid_sunet7128, self).__init__()
        self.num_classes = num_classes
        sunet = stackedunet7128(output_stride=output_stride)
        sunet = torch.nn.DataParallel(sunet, device_ids=range(torch.cuda.device_count())).cuda()
        if pretrained:
            checkpoint = torch.load(sunet7128_path)
            sunet.load_state_dict(checkpoint['state_dict'])

        self.features = sunet.module.features

        for n, m in self.features.named_modules():
            if 'bn' in n:
                m.momentum = mom_bn

        # hack needed as batchnorm modules were not named as bn in residual block
        for n, m in self.features.residual1.conv.named_modules():
            if '2' in n:
                m.momentum = mom_bn

        self.final = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(2304, 512, kernel_size=3, padding=2, dilation=2, bias=True)),
            ('bn1', nn.BatchNorm2d(512, momentum=mom_bn)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=True)),
            ('bn2', nn.BatchNorm2d(512, momentum=mom_bn)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(512, num_classes, kernel_size=1)),
        ]))

        self.mceloss = cross_entropy2d(ignore=ignore_index, size_average=False, weight=weight)

    def forward(self, x, labels=None, th=1.0):
        x_size = x.size()
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = self.final(x)
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=False)

        if labels is not None:
            losses, total_valid_pixel = self.mceloss(x, labels, th=th)

            classwise_pixel_acc, classwise_gtpixels, classwise_predpixels = prediction_stat([x], labels, self.num_classes)

            # Need to perform this operation for MultiGPU
            classwise_pixel_acc = Variable(torch.FloatTensor([classwise_pixel_acc]).cuda())
            classwise_gtpixels = Variable(torch.FloatTensor([classwise_gtpixels]).cuda())
            classwise_predpixels = Variable(torch.FloatTensor([classwise_predpixels]).cuda())

            return x, losses, classwise_pixel_acc, classwise_gtpixels, classwise_predpixels, total_valid_pixel
        else:
            return x

#################################################################################################################################################################################################################################



































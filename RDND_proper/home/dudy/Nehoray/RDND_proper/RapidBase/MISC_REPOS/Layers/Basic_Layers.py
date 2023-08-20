import torch
import torch.nn as nn




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
def Color_Space_Conversion_Layer(number_of_input_channels, number_of_output_channels, flag_3D=False):
    if flag_3D:
        return nn.Conv3d(number_of_input_channels,number_of_output_channels,1,1,0,1,1,True) #Colorspace conversion can be ahieved by a 1X1 convolution with number of groups=1
    else:
        return nn.Conv2d(number_of_input_channels,number_of_output_channels,1,1,0,1,1,True) #Colorspace conversion can be ahieved by a 1X1 convolution with number of groups=1

class Crop_Center_Layer(nn.Module):
    def __init__(self, crop_H, crop_W, *args):
        super(Crop_Center_Layer, self).__init__()
        self.crop_x = crop_W
        self.crop_y = crop_H
    def forward(self, x):
        B,C,H,W = x.shape
        startx = W // 2 - (self.crop_x // 2)
        starty = H // 2 - (self.crop_y // 2)
        return x[:,:,starty:starty + self.crop_y, startx:startx + self.crop_x]


class Crop_Center_Layer_3Values(nn.Module):
    def __init__(self, crop_H, crop_W, *args):
        super(Crop_Center_Layer_3Values, self).__init__()
        self.crop_x = crop_W
        self.crop_y = crop_H
    def forward(self, x):
        C,H,W = x.shape
        startx = W // 2 - (self.crop_x // 2)
        starty = H // 2 - (self.crop_y // 2)
        return x[:, starty:starty + self.crop_y, startx:startx + self.crop_x]


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








def Padding_Layer(padding_type, padding_size):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    padding_type = padding_type.lower()
    if padding_size == 0:
        return nn.ReflectionPad2d(padding_size)
    if padding_type == 'reflect':
        layer = nn.ReflectionPad2d(padding_size)
    elif padding_type == 'replicate':
        layer = nn.ReplicationPad2d(padding_size)
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % padding_type)
    return layer



### I'm Turning Pad_Conv2d into a nn.Module Object to allow smooth .to(device) etc' functionality: ###
class Pad_Conv2d(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride=1,dilation=1,groups=1):
        super(Pad_Conv2d, self).__init__()
        padding_size = get_valid_padding(kernel_size, dilation)
        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups)
        self.padding_layer = Padding_Layer('reflect', padding_size=padding_size);
        print(self.padding_layer)
        self.Combined_Layer = nn.Sequential(self.padding_layer, self.conv_layer)
    def forward(self, x):
        if self.conv_layer.parameters().__next__().device != x.device:
            self.conv_layer = self.conv_layer.to(x.device)
        return self.Combined_Layer(x)

# def Pad_Conv2d(in_channels,out_channels,kernel_size,stride=1,dilation=1,groups=1):
#     padding_size = get_valid_padding(kernel_size, dilation)
#     Combined_Layer = nn.Sequential(Padding_Layer('reflect',padding_size=padding_size), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups))
#     return Combined_Layer


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



class Add_Zero_Layer(nn.Module):
    ### Add zero just to have some ==0 gradients to allow a predetermined model without learning to go through my training loop
    def __init__(self):
        super(Add_Zero_Layer, self).__init__()
        self.zero_conv = None;
    def forward(self, x):
        if self.zero_conv == None:
            self.zero_conv = nn.Conv2d(x.shape[1],x.shape[1],3,1,1)
        output_tensor = x + 0*self.zero_conv(x)
        return output_tensor;



import torch.nn.functional as F
import math
import numpy as np
class Gaussian_Blur_Layer(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussian_Blur_Layer, self).__init__()
        if type(kernel_size) is not list and type(kernel_size) is not tuple:
            kernel_size = [kernel_size] * dim
        if type(sigma) is not list and type(sigma) is not tuple:
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

        self.padding = []
        for i in kernel_size:
            self.padding.append(i//2)
            self.padding.append(i//2)

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(F.pad(input, self.padding), weight=self.weight, groups=self.groups)


def Gaussian_Blur_Wrapper_Torch(gaussian_blur_layer, input_tensor, number_of_channels_per_frame, frames_dim=0, channels_dim=1):
    #TODO: this is extremely inefficient!!!! never have for loops (without parallelization) in script!!! take care of this!!!!
    #TODO: simply have the gaussian blur work on all channels together or something. in case of wanting several blurs within the batch we can do that
    ### indices pre processing: ###
    frames_dim_channels = input_tensor.shape[frames_dim]
    channels_dim_channels = input_tensor.shape[channels_dim]
    flag_frames_concatenated_along_channels = (frames_dim == channels_dim)
    if frames_dim == channels_dim:  # frames concatenated along channels dim
        number_of_frames = int(channels_dim_channels / number_of_channels_per_frame)
    else:
        number_of_frames = frames_dim_channels
    output_tensor = torch.zeros_like(input_tensor)

    if len(input_tensor.shape) == 4: #[B,C,H,W]
        B,C,H,W = input_tensor.shape

        if flag_frames_concatenated_along_channels:  #[B,C*T,H,W]
            for i in np.arange(number_of_frames):
                output_tensor[:,i*number_of_channels_per_frame*(i+1)*number_of_channels_per_frame,:,:] = \
                    gaussian_blur_layer(input_tensor[:,i*number_of_channels_per_frame*(i+1)*number_of_channels_per_frame,:,:])
        else: #[T,C,H,W] or [B,C,H,W]
            for i in np.arange(number_of_frames):
                output_tensor[i:i+1,:,:,:] = gaussian_blur_layer(input_tensor[i:i+1,:,:,:])

    elif len(input_tensor.shape) == 5: #[B,T,C,H,W]
        B, T, C, H, W = input_tensor.shape

        for i in np.arange(number_of_frames):
            output_tensor[:,i,:,:,:] = gaussian_blur_layer(input_tensor[:,i,:,:,:])

    return output_tensor








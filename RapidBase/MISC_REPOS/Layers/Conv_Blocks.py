# from pytorch_deform_conv.torch_deform_conv.deform_conv import th_batch_map_offsets, th_generate_grid


import RapidBase.Basic_Import_Libs
from RapidBase.Basic_Import_Libs import *

# import RapidBase.Utils_Import_Libs
# from RapidBase.Utils_Import_Libs import *


#(5). Layers:
import RapidBase.MISC_REPOS.Layers.Activations
import RapidBase.MISC_REPOS.Layers.Basic_Layers
import RapidBase.MISC_REPOS.Layers.DownSample_UpSample
import RapidBase.MISC_REPOS.Layers.Memory_and_RNN
import RapidBase.MISC_REPOS.Layers.Refinement_Modules
import RapidBase.MISC_REPOS.Layers.Special_Layers
import RapidBase.Utils.Registration.Warp_Layers
import RapidBase.MISC_REPOS.Layers.Wrappers
from RapidBase.MISC_REPOS.Layers.Activations import *
from RapidBase.MISC_REPOS.Layers.Basic_Layers import *
from RapidBase.MISC_REPOS.Layers.DownSample_UpSample import *
from RapidBase.MISC_REPOS.Layers.Memory_and_RNN import *
from RapidBase.MISC_REPOS.Layers.Refinement_Modules import *
from RapidBase.MISC_REPOS.Layers.Special_Layers import *
from RapidBase.Utils.Registration.Warp_Layers import *
from RapidBase.MISC_REPOS.Layers.Wrappers import *




############################################################################################################################################################################################################################################################
### Auxiliary: ###
def combined_dirac_and_xavier_initialization(layer):
    dirac_weights = torch.Tensor(nn.init.dirac_(layer.weight).data.numpy().copy())
    xavier_weights = torch.Tensor(nn.init.xavier_normal_(layer.weight).data.numpy().copy())
    layer.weight = nn.Parameter(dirac_weights + xavier_weights)


class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                v = normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)

        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)

        u = normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # buffer, which will cause weight to be included in the state dict
        # and also supports nn.init due to shared storage.
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)

        module.register_forward_pre_hook(fn)
        return fn



def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    r"""Applies spectral normalization to a parameter in the given module.
    .. math::
         \mathbf{W} &= \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) &= \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.
    See `Spectral Normalization for Generative Adversarial Networks`_ .
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is 0, except for modules that are instances of
            ConvTranspose1/2/3d, when it is 1
    Returns:
        The original module with the spectal norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40))
        Linear (20 -> 40)
        >>> m.weight_u.size()
        torch.Size([20])
    """
    if dim is None:
        if isinstance(
                module,
            (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))




def Normalization_Function(normalization_function, number_of_input_channels, flag_3D=False):
    # print(normalization_function)
    # helper selecting normalization layer
    normalization_function = normalization_function.lower()
    if normalization_function == 'batch_normalization':
        if flag_3D:
            layer = nn.BatchNorm3d(number_of_input_channels, affine=True)
        else:
            layer = nn.BatchNorm2d(number_of_input_channels, affine=True)
    elif normalization_function == 'instance_normalization':
        if flag_3D:
            layer = nn.InstanceNorm3d(number_of_input_channels, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        else:
            layer = nn.InstanceNorm2d(number_of_input_channels, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    elif normalization_function == 'layer_normalization':
        layer = nn.LayerNorm(number_of_input_channels, eps=1e-05, elementwise_affine=True)
    elif normalization_function == None or normalization_function=='none':
        layer = Identity_Layer();
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % normalization_function)
    return layer



def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block_by_string(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'P':
            L.append(nn.PReLU())
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)



def standard_conv_block(in_planes, out_planes, kernel_size, stride, pad, dilation=1, flag_use_BN=True):
    ### Probably Change to my own conv_block: ###
    if flag_use_BN:
        return nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False)) ### Why No Bias?????
    else:
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False)) ### Why No Bias?????


def grouped_conv_block(in_planes, out_planes, kernel_size, stride, pad, dilation=1, flag_use_BN=True, flag_3D=False):
    if flag_use_BN:
        if flag_3D:
            return nn.Sequential(nn.BatchNorm3d(in_planes),
                                 nn.ReLU(inplace=False),
                                 nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False, groups=in_planes))  ### Why No Bias?????
        else:
            return nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False, groups=in_planes)) ### Why No Bias?????
    else:
        if flag_3D:
            return nn.Sequential(nn.ReLU(inplace=False), nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False, groups=in_planes))  ### Why No Bias?????
        else:
            return nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False, groups=in_planes)) ### Why No Bias?????


### Details: ###
#(1). outputs features at x4->x16
#(2). guidance maps at x4 scale
#(3). guidance maps number of channels = Refinement_number_of_base_channels * number_of_CSPN_neighbors(=8).
#(4). to use with SPN simply take the first 3*Refinement_number_of_base_channels number of channels
#(5). WITH IIR AT THE CENTER LAYER OF THE UNET
class IIR_MagIC_V2(nn.Module):
    def __init__(self, number_of_input_channels=3, IIR_filter_order=1, flag_direction='horizontal', flag_reverse=False):
        super(IIR_MagIC_V2, self).__init__()
        ### Define auxiliary padding layers: ###
        self.left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))  # the padding notation is (left,right,top,bottom)  (assuming we diffuse an affinity of 3x3 neighborhood)
        self.center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        self.right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        self.left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        self.right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        self.left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        self.center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        self.right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))
        self.center_center_pad = nn.ZeroPad2d((1, 1, 1, 1))

        self.center_center_pad = nn.ZeroPad2d((2,2,2,2))

        ### Initialize Parameters: ###
        self.IIR_filter_order = IIR_filter_order
        self.flag_direction = flag_direction
        self.flag_reverse = flag_reverse
        self.FIR_coefficients1 = nn.ParameterList()
        self.FIR_coefficients2 = nn.ParameterList()
        self.IIR_coefficients1 = nn.ParameterList()
        self.IIR_coefficients2 = nn.ParameterList()
        # ### Initialize between [0,1]: ###
        # epsilon = 0.05
        # self.FIR_coefficients.append(nn.Parameter(torch.Tensor([np.random.uniform(0 + epsilon, 1 - epsilon)])))
        # self.FIR_coefficients.append(nn.Parameter(torch.Tensor([np.random.uniform(0 + epsilon, 1 - epsilon)])))
        # self.IIR_coefficients.append(nn.Parameter(torch.Tensor([np.random.uniform(0 + epsilon, 1 - epsilon)])))
        ### Initialize between [-std,std]: ###
        std = 1 / sqrt(3)
        self.FIR_coefficients1.append(nn.Parameter(torch.Tensor([np.random.uniform(-std,std,number_of_input_channels)]).view(1,number_of_input_channels,1)))
        self.FIR_coefficients2.append(nn.Parameter(torch.Tensor([np.random.uniform(-std,std,number_of_input_channels)]).view(1,number_of_input_channels,1)))
        self.IIR_coefficients1.append(nn.Parameter(torch.Tensor([np.random.uniform(-std,std,number_of_input_channels)]).view(1,number_of_input_channels,1)))
        if self.IIR_filter_order == 2:
            self.IIR_coefficients2.append(nn.Parameter(torch.Tensor([np.random.uniform(-std, std, number_of_input_channels)]).view(1, number_of_input_channels, 1)))
        else:
            self.IIR_coefficients2.append(nn.Parameter(torch.Tensor([np.random.uniform(-0, 0, number_of_input_channels)]).view(1, number_of_input_channels, 1)))
            self.IIR_coefficients2[0].requires_grad = False

    def forward(self, input_tensor):
        # TODO: this is written horribly -> write more efficient!
        # [Guidance_Maps] = [N,C_base*3,H,W]  (C_base=number of channels we have in the final estimate of whatever we're trying to refine, 3=number of immediate neighbors we're trying to model the affinity relations of)
        # [blurred_depth] = [N,C_base,H,W]  (here i'm calling it by the informative name of blurred_depth but at the end it will change to "estimate_to_refine" or something)

        N, C, H, W = input_tensor.shape

        ### Initialize results: ###
        rnn_h1 = Variable(torch.zeros((N, C, H, W)).to(input_tensor.device))
        input_tensor_padded = self.center_center_pad(input_tensor)
        rnn_h1_padded = self.center_center_pad(rnn_h1)

        if self.flag_direction=='horizontal' and self.flag_reverse==False:
            for i in range(2,W+2):
                ### Left->Right: ###
                rnn_h1_padded[:,:,2:-2,i] = self.FIR_coefficients1[0]*input_tensor_padded[:,:,2:-2,i] + \
                                            self.FIR_coefficients2[0]*input_tensor_padded[:,:,2:-2,i-1] + \
                                            self.IIR_coefficients1[0]*rnn_h1_padded.clone()[:,:,2:-2,i-1] + \
                                            self.IIR_coefficients2[0]*rnn_h1_padded.clone()[:,:,2:-2,i-2]


        if self.flag_direction=='horizontal' and self.flag_reverse==True:
            for i in range(2,W+2):
                ### Right->Left: ###
                # (W+3) = (W+1+2*(filter_order-2))
                rnn_h1_padded[:, :, 2:-2, (W+3)-i] = self.FIR_coefficients1[0] * input_tensor_padded[:, :, 2:-2, (W+3)-i] +\
                                               self.FIR_coefficients2[0] * input_tensor_padded[:, :, 2:-2, (W+3)-i+1] +\
                                               self.IIR_coefficients1[0] * rnn_h1_padded.clone()[:, :, 2:-2, (W+3)-i+1] + \
                                               self.IIR_coefficients2[0] * rnn_h1_padded.clone()[:, :, 2:-2, (W+3)-i+2]


        if self.flag_direction=='vertical' and self.flag_reverse==False:
            for i in range(2,H+2):
                ### Up->Down: ###
                rnn_h1_padded[:, :, i, 2:-2] = self.FIR_coefficients1[0] * input_tensor_padded[:, :, i, 2:-2] + \
                                                   self.FIR_coefficients2[0] * input_tensor_padded[:, :, i - 1, 2:-2] + \
                                                   self.IIR_coefficients1[0] * rnn_h1_padded.clone()[:, :, i - 1, 2:-2] + \
                                                   self.IIR_coefficients2[0] * rnn_h1_padded.clone()[:, :, i - 2, 2:-2]


        if self.flag_direction=='vertical' and self.flag_reverse==True:
            for i in range(2, H+2):
                ### Down->Up: ###
                rnn_h1_padded[:, :, (H + 3) - i, 2:-2] = self.FIR_coefficients1[0] * input_tensor_padded[:, :, (H + 3) - i, 2:-2] + \
                                                         self.FIR_coefficients2[0] * input_tensor_padded[:, :, (H + 3) - i + 1, 2:-2] + \
                                                         self.IIR_coefficients1[0] * rnn_h1_padded.clone()[:, :, (H + 3) - i + 1, 2:-2] + \
                                                         self.IIR_coefficients2[0] * rnn_h1_padded.clone()[:, :, (H + 3) - i + 2, 2:-2]

        return rnn_h1_padded[:,:,2:-2,2:-2]
#########################################################################################################################################################################################################################################################################################################











###############################################################################################################################################################################################################################################################################################
class Conv2d_Multiple_Kernels(nn.Module):
    # Initialize this with a module
    def __init__(self,
                 number_of_input_channels,
                 number_of_output_channels,
                 kernel_size_list=1,
                 stride=1,  # Stride must be the same for all because we can't concatenate outputs if output's sizes change!
                 dilation_list=1,
                 groups=1,
                 bias=True,
                 padding_type='zero',  # don't do padding .... this means the output size will be different from input size right?
                 normalization_type=None,
                 activation_type='relu',
                 mode='CNA',
                 initialization_method='dirac',  ### Deformable Convolution: ###
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v3',
                 flag_deformable_convolution_before_or_after_main_convolution='before',  # 'before' / 'after'
                 flag_deformable_convolution_modulation=True,
                 flag_deformable_kernel_size=5,
                 flag_deformable_number_of_deformable_groups=1,
                 flag_deformable_number_of_channels_in_group1=-1,  # for instance, i might want 2 UNEQUAL deformable groups
                 flag_deformable_for_each_sub_block_or_for_super_block='super_block',  # 'super_block' / 'sub_block'
                 flag_deformable_same_on_all_channels=True,  # True/False  - TODO: Remove this! can be done using number_of_deformable_groups=1
                 ### Deformable SFT: ###
                 flag_deformable_SFT_use_outside_conditional=False,
                 flag_deformable_SFT_same_on_all_channels=False,  ### The interactive shift/scale convs by default spits-out x.shape[1] number_of_channels....but maybe to save memory and parameters we want the same interaction on all channels
                 flag_deformable_SFT_base_convs_mix='x',
                 flag_deformable_SFT_SFT_convs_mix='x',
                 flag_deformable_SFT_shift=False,
                 flag_deformable_SFT_scale=False,
                 flag_deformable_SFT_add_y_to_output=False,
                 #####################################
                 ### Cell and Super-Cell Types: ######
                 flag_single_cell_block_type='simple',  # 'simple'/'standard_residual'/'131_residual'
                 flag_super_cell_block_type='131_collapse',  # 'concat' / '131' / '131_collapse' / 'concat_standard_residual' / '131_residual' / '131_collapse_residual'
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
                 ### IIR: ###
                 flag_IIR_filter_horizontal=False,
                 flag_IIR_filter_horizontal_reverse=False,
                 flag_IIR_filter_order_horizontal=1,
                 flag_IIR_filter_vertical=False,
                 flag_IIR_filter_vertical_reverse=False,
                 flag_IIR_filter_order_vertical=1, flag_3D=False
                 ):
        super(Conv2d_Multiple_Kernels, self).__init__()

        ### Choose whether to put deformable at the start/end of each sub_block seperately or at the start/end of the "super_block" of multiple kernels: ###
        # TODO: i think for at least some simplicity i should always have flag_deformable_convolution_sub_blocks=False and only deal with it on the super_block level
        if flag_deformable_for_each_sub_block_or_for_super_block == 'super_block':
            flag_deformable_convolution_sub_blocks = False
        else:
            flag_deformable_convolution_sub_blocks = flag_deformable_convolution


        ###################################################################################################
        ### Get Conv_Blocks: ###
        modules_list = nn.ModuleList()  #TODO: nn.ModuleList() or list ????
        for i in np.arange(len(kernel_size_list)):
            ### TODO: the below commented out block is in case i want even more flexibility and have the different sub-conv_blocks have different things like deformable or not etc'....
            # modules_list.append(Conv_Block(number_of_input_channels,number_of_output_channels,
            #                                kernel_size_list[i], stride, dilation_list[i], groups, bias, padding_type,
            #                                normalization_type, activation_type, mode, initialization_method,
            #                                flag_deformable_convolution[i],
            #                                flag_deformable_convolution_version[i],
            #                                flag_deformable_convolution_before_or_after_main_convolution[i],
            #                                flag_deformable_convolution_modulation[i]))
            modules_list.append(Conv_Block(number_of_input_channels, number_of_output_channels, kernel_size_list[i], stride, dilation_list[i], groups, bias, padding_type, normalization_type, activation_type, mode,  # 'CNA'
                                           initialization_method, flag_deformable_convolution_sub_blocks, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation,  # I GIVE UP ON DEFORMABLE IN SUB-BLOCKS SO THIS DOESN'T MATTER
                                           flag_single_cell_block_type, flag_3D=flag_3D))

        modules_list2 = nn.ModuleList()  # TODO: nn.ModuleList() or list ????
        for i in np.arange(len(kernel_size_list)):
            ### TODO: the below commented out block is in case i want even more flexibility and have the different sub-conv_blocks have different things like deformable or not etc'....
            # modules_list.append(Conv_Block(number_of_input_channels,number_of_output_channels,
            #                                kernel_size_list[i], stride, dilation_list[i], groups, bias, padding_type,
            #                                normalization_type, activation_type, mode, initialization_method,
            #                                flag_deformable_convolution[i],
            #                                flag_deformable_convolution_version[i],
            #                                flag_deformable_convolution_before_or_after_main_convolution[i],
            #                                flag_deformable_convolution_modulation[i]))
            modules_list2.append(Conv_Block(number_of_input_channels, number_of_output_channels, kernel_size_list[i], stride, dilation_list[i], groups, bias, padding_type, normalization_type, activation_type, mode,  # 'CNA'
                                           initialization_method, flag_deformable_convolution_sub_blocks, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation,  # I GIVE UP ON DEFORMABLE IN SUB-BLOCKS SO THIS DOESN'T MATTER
                                           flag_single_cell_block_type, flag_3D=flag_3D))
        ####################################################################################################



        ####################################################################################################
        ### Simple Concat: ###
        if flag_super_cell_block_type == 'concat':
            full_basic_conv_block = Concat_Block(modules_list)  #TODO: uncomment this at the end!!!
            # (1). Vertical IIR:
            if flag_IIR_filter_vertical:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            if flag_IIR_filter_vertical_reverse:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            # (2). Horizontal IIR:
            if flag_IIR_filter_horizontal:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            if flag_IIR_filter_horizontal_reverse:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            # (3). Regular Conv Post-IIR:
            if flag_IIR_filter_horizontal or flag_IIR_filter_vertical or flag_IIR_filter_vertical_reverse or flag_IIR_filter_horizontal_reverse:
                current_post_IIR_conv = grouped_conv_block(number_of_output_channels * len(kernel_size_list), number_of_output_channels * len(kernel_size_list), 3, 1, 1, (normalization_type == 'batch_normalization'), flag_3D=flag_3D)  # TODO: probably should use grouped convolution to save memory and be like magic
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_post_IIR_conv)

        ### 131_Collapse: ###
        elif flag_super_cell_block_type == '131_collapse':
            convolution_layer = Concat_Block(modules_list)
            extra_1x1_conv_block1 = ConvNd_layer(number_of_input_channels, number_of_input_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, flag_3D=flag_3D)
            extra_1x1_conv_block2 = ConvNd_layer(number_of_output_channels * len(kernel_size_list), number_of_output_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, flag_3D=flag_3D)
            if initialization_method == 'combined':
                combined_dirac_and_xavier_initialization(extra_1x1_conv_block1);
                combined_dirac_and_xavier_initialization(extra_1x1_conv_block2);
            elif initialization_method == 'dirac':
                nn.init.dirac_(extra_1x1_conv_block1.weight)
                nn.init.dirac_(extra_1x1_conv_block2.weight)
            elif initialization_method == 'xavier':
                nn.init.xavier_normal_(extra_1x1_conv_block1.weight)
                nn.init.xavier_normal_(extra_1x1_conv_block2.weight)
            activation_layer = Activation_Function(activation_type) if activation_type else None
            activation_layer2 = Activation_Function(activation_type) if activation_type else None
            normalization_layer = Normalization_Function(normalization_type, number_of_input_channels, flag_3D=flag_3D) if normalization_type else None
            normalization_layer2 = Normalization_Function(normalization_type, number_of_output_channels, flag_3D=flag_3D) if normalization_type else None
            if mode=='CNA':
                full_basic_conv_block = nn.Sequential(extra_1x1_conv_block1, normalization_layer, activation_layer, convolution_layer,  # These already have the normaliation and activation layer inside of them and i don't have to differentiate between normalization/activation at each cell and all of them together
                                                                          extra_1x1_conv_block2, normalization_layer2, activation_layer2)
            elif mode=='NAC':
                full_basic_conv_block = nn.Sequential(normalization_layer, activation_layer, extra_1x1_conv_block1, convolution_layer,  # These already have the normaliation and activation layer inside of them and i don't have to differentiate between normalization/activation at each cell and all of them together
                                                                          normalization_layer2, activation_layer2, extra_1x1_conv_block2)
            # (1). Vertical IIR:
            if flag_IIR_filter_vertical:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            if flag_IIR_filter_vertical_reverse:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            # (2). Horizontal IIR:
            if flag_IIR_filter_horizontal:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            if flag_IIR_filter_horizontal_reverse:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            # (3). Regular Conv Post-IIR:
            if flag_IIR_filter_horizontal or flag_IIR_filter_vertical or flag_IIR_filter_vertical_reverse or flag_IIR_filter_horizontal_reverse:
                current_post_IIR_conv = grouped_conv_block(number_of_output_channels, number_of_output_channels, 3, 1, 1, (normalization_type == 'batch_normalization'), flag_3D=flag_3D)  # TODO: probably should use grouped convolution to save memory and be like magic
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_post_IIR_conv)



        ### standard_residual_1 (ReLU after addition): ###
        # x  - Conv-BN-Relu-Conv-BN---|
        # ----------------------------(+)-ReLU
        elif flag_super_cell_block_type == 'standard_residual_1':
            convolution_layer = Concat_Block(modules_list)
            convolution_layer2 = Concat_Block(modules_list2)
            extra_1x1_conv_block1 = ConvNd_layer(number_of_input_channels, number_of_input_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, flag_3D=flag_3D)
            extra_1x1_conv_block2 = ConvNd_layer(number_of_output_channels * len(kernel_size_list), number_of_output_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, flag_3D=flag_3D)
            if initialization_method == 'combined':
                combined_dirac_and_xavier_initialization(extra_1x1_conv_block1);
                combined_dirac_and_xavier_initialization(extra_1x1_conv_block2);
            elif initialization_method == 'dirac':
                nn.init.dirac_(extra_1x1_conv_block1.weight)
                nn.init.dirac_(extra_1x1_conv_block2.weight)
            elif initialization_method == 'xavier':
                nn.init.xavier_normal_(extra_1x1_conv_block1.weight)
                nn.init.xavier_normal_(extra_1x1_conv_block2.weight)
            activation_layer = Activation_Function(activation_type) if activation_type else None
            activation_layer2 = Activation_Function(activation_type) if activation_type else None
            normalization_layer = Normalization_Function(normalization_type, number_of_input_channels, flag_3D=flag_3D) if normalization_type else None
            normalization_layer2 = Normalization_Function(normalization_type, number_of_output_channels, flag_3D=flag_3D) if normalization_type else None
            if mode == 'CNA':
                full_basic_conv_block = nn.Sequential(convolution_layer, normalization_layer, activation_layer, convolution_layer2,  # These already have the normaliation and activation layer inside of them and i don't have to differentiate between normalization/activation at each cell and all of them together
                                                      normalization_layer2)
            # (1). Vertical IIR:
            if flag_IIR_filter_vertical:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            if flag_IIR_filter_vertical_reverse:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            # (2). Horizontal IIR:
            if flag_IIR_filter_horizontal:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            if flag_IIR_filter_horizontal_reverse:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            # (3). Regular Conv Post-IIR:
            if flag_IIR_filter_horizontal or flag_IIR_filter_vertical or flag_IIR_filter_vertical_reverse or flag_IIR_filter_horizontal_reverse:
                current_post_IIR_conv = grouped_conv_block(number_of_output_channels, number_of_output_channels, 3, 1, 1, (normalization_type == 'batch_normalization'), flag_3D=flag_3D)  # TODO: probably should use grouped convolution to save memory and be like magic
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_post_IIR_conv)
            ### Put it all together: ###
            full_basic_conv_block = Input_Output_Sum_Block_With_StraightThrough_Projection_Block(full_basic_conv_block, number_of_output_channels=number_of_output_channels * len(kernel_size_list), residual_scale=1, flag_3D=flag_3D)
            full_basic_conv_block = nn.Sequential(full_basic_conv_block, activation_layer2)



        ### standard_residual_2: ###
        # x  - Conv-BN-Relu-Conv-BN-ReLU-|
        # -------------------------------(+)
        elif flag_super_cell_block_type == 'standard_residual_2':
            convolution_layer = Concat_Block(modules_list)
            convolution_layer2 = Concat_Block(modules_list2)
            extra_1x1_conv_block1 = ConvNd_layer(number_of_input_channels, number_of_input_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, flag_3D=flag_3D)
            extra_1x1_conv_block2 = ConvNd_layer(number_of_output_channels * len(kernel_size_list), number_of_output_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, flag_3D=flag_3D)
            if initialization_method == 'combined':
                combined_dirac_and_xavier_initialization(extra_1x1_conv_block1);
                combined_dirac_and_xavier_initialization(extra_1x1_conv_block2);
            elif initialization_method == 'dirac':
                nn.init.dirac_(extra_1x1_conv_block1.weight)
                nn.init.dirac_(extra_1x1_conv_block2.weight)
            elif initialization_method == 'xavier':
                nn.init.xavier_normal_(extra_1x1_conv_block1.weight)
                nn.init.xavier_normal_(extra_1x1_conv_block2.weight)
            activation_layer = Activation_Function(activation_type) if activation_type else None
            activation_layer2 = Activation_Function(activation_type) if activation_type else None
            normalization_layer = Normalization_Function(normalization_type, number_of_input_channels, flag_3D=flag_3D) if normalization_type else None
            normalization_layer2 = Normalization_Function(normalization_type, number_of_output_channels, flag_3D=flag_3D) if normalization_type else None
            if mode == 'CNA':
                full_basic_conv_block = nn.Sequential(convolution_layer, normalization_layer, activation_layer, convolution_layer2,  # These already have the normaliation and activation layer inside of them and i don't have to differentiate between normalization/activation at each cell and all of them together
                                                      normalization_layer2, activation_layer2)
            # (1). Vertical IIR:
            if flag_IIR_filter_vertical:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            if flag_IIR_filter_vertical_reverse:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            # (2). Horizontal IIR:
            if flag_IIR_filter_horizontal:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            if flag_IIR_filter_horizontal_reverse:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            # (3). Regular Conv Post-IIR:
            if flag_IIR_filter_horizontal or flag_IIR_filter_vertical or flag_IIR_filter_vertical_reverse or flag_IIR_filter_horizontal_reverse:
                current_post_IIR_conv = grouped_conv_block(number_of_output_channels, number_of_output_channels, 3, 1, 1, (normalization_type == 'batch_normalization'), flag_3D=flag_3D)  # TODO: probably should use grouped convolution to save memory and be like magic
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_post_IIR_conv)
            ### Put it all together: ###
            full_basic_conv_block = Input_Output_Sum_Block_With_StraightThrough_Projection_Block(full_basic_conv_block, number_of_output_channels=number_of_output_channels * len(kernel_size_list), residual_scale=1, flag_3D=flag_3D)



        ### Concat Standard Residual: ###
        elif flag_super_cell_block_type == 'concat_standard_residual':
            convolution_layer = Concat_Block(modules_list)
            ### Extra Conv_Block before residual: ###
            activation_layer = Activation_Function(activation_type) if activation_type else None
            normalization_layer = Normalization_Function(normalization_type, number_of_output_channels, flag_3D=flag_3D) if normalization_type else None
            extra_layer_kernel_size = 3;
            padding_needed = extra_layer_kernel_size // 2
            padding_layer = ReflectionPaddingNd_Layer(padding_needed, flag_3D=flag_3D);
            extra_conv_block = ConvNd_layer(number_of_output_channels * len(kernel_size_list), number_of_output_channels * len(kernel_size_list), kernel_size=extra_layer_kernel_size, stride=1, padding=0, dilation=1, groups=1, flag_3D=flag_3D)
            if initialization_method == 'combined':
                combined_dirac_and_xavier_initialization(extra_conv_block);
            elif initialization_method == 'dirac':
                nn.init.dirac_(extra_conv_block.weight)
            elif initialization_method == 'xavier':
                nn.init.xavier_normal_(extra_conv_block.weight)
            full_basic_conv_block = nn.Sequential(convolution_layer, normalization_layer, activation_layer, padding_layer, extra_conv_block)
            # (1). Vertical IIR:
            if flag_IIR_filter_vertical:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            if flag_IIR_filter_vertical_reverse:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            # (2). Horizontal IIR:
            if flag_IIR_filter_horizontal:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            if flag_IIR_filter_horizontal_reverse:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            # (3). Regular Conv Post-IIR:
            if flag_IIR_filter_horizontal or flag_IIR_filter_vertical or flag_IIR_filter_vertical_reverse or flag_IIR_filter_horizontal_reverse:
                current_post_IIR_conv = grouped_conv_block(number_of_output_channels * len(kernel_size_list), number_of_output_channels * len(kernel_size_list), 3, 1, 1, (normalization_type == 'batch_normalization'))  # TODO: probably should use grouped convolution to save memory and be like magic
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_post_IIR_conv)
            full_basic_conv_block = Input_Output_Sum_Block_With_StraightThrough_Projection_Block(full_basic_conv_block, number_of_output_channels=number_of_output_channels * len(kernel_size_list), residual_scale=1, flag_3D=flag_3D)


        ### 131 Residual: ###
        elif flag_super_cell_block_type == '131_residual':
            convolution_layer = Concat_Block(modules_list)
            ### Extra Conv_Block before residual: ###
            activation_layer = Activation_Function(activation_type) if activation_type else None
            activation_layer2 = Activation_Function(activation_type) if activation_type else None
            normalization_layer = Normalization_Function(normalization_type, number_of_input_channels, flag_3D=flag_3D) if normalization_type else None
            normalization_layer2 = Normalization_Function(normalization_type, number_of_output_channels, flag_3D=flag_3D) if normalization_type else None
            padding_layer = ReflectionPaddingNd_Layer(0, flag_3D);
            extra_1x1_conv_block1 = ConvNd_layer(number_of_input_channels, number_of_input_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, flag_3D=flag_3D)
            extra_1x1_conv_block2 = ConvNd_layer(number_of_output_channels * len(kernel_size_list), number_of_output_channels * len(kernel_size_list), kernel_size=1, stride=1, padding=0, dilation=1, groups=1, flag_3D=flag_3D)  #Notice the *len(kernel_size_list) ... no collapsing with 1x1
            if initialization_method == 'combined':
                combined_dirac_and_xavier_initialization(extra_1x1_conv_block1);
                combined_dirac_and_xavier_initialization(extra_1x1_conv_block2);
            elif initialization_method == 'dirac':
                nn.init.dirac_(extra_1x1_conv_block1.weight)
                nn.init.dirac_(extra_1x1_conv_block2.weight)
            elif initialization_method == 'xavier':
                nn.init.xavier_normal_(extra_1x1_conv_block1.weight)
                nn.init.xavier_normal_(extra_1x1_conv_block2.weight)
            full_basic_conv_block = nn.Sequential(extra_1x1_conv_block1, normalization_layer, activation_layer,
                                                  convolution_layer,  # These already have the normaliation and activation layer inside of them and i don't have to differentiate between normalization/activation at each cell and all of them together
                                                  extra_1x1_conv_block2, normalization_layer2, activation_layer2)
            # (1). Vertical IIR:
            if flag_IIR_filter_vertical:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            if flag_IIR_filter_vertical_reverse:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            # (2). Horizontal IIR:
            if flag_IIR_filter_horizontal:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            if flag_IIR_filter_horizontal_reverse:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels * len(kernel_size_list), IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            # (3). Regular Conv Post-IIR:
            if flag_IIR_filter_horizontal or flag_IIR_filter_vertical or flag_IIR_filter_vertical_reverse or flag_IIR_filter_horizontal_reverse:
                current_post_IIR_conv = grouped_conv_block(number_of_output_channels * len(kernel_size_list), number_of_output_channels * len(kernel_size_list), 3, 1, 1, (normalization_type == 'batch_normalization'), flag_3D=flag_3D)  # TODO: probably should use grouped convolution to save memory and be like magic
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_post_IIR_conv)
            full_basic_conv_block = Input_Output_Sum_Block_With_StraightThrough_Projection_Block(full_basic_conv_block, number_of_output_channels=number_of_output_channels * len(kernel_size_list), residual_scale=1, flag_3D=flag_3D)



        ### 131 Collapse Residual: ###
        elif flag_super_cell_block_type == '131_collapse_residual':  # TODO: maybe even differentiate between a '131_collapse_residual' and '131_residual_collapse'.... as i understand it the '131_residual_collapse' has larger capacity....
            convolution_layer = Concat_Block(modules_list)
            ### Extra Conv_Block before residual: ###
            activation_layer = Activation_Function(activation_type) if activation_type else None
            activation_layer2 = Activation_Function(activation_type) if activation_type else None
            normalization_layer = Normalization_Function(normalization_type, number_of_input_channels, flag_3D=flag_3D) if normalization_type else None
            normalization_layer2 = Normalization_Function(normalization_type, number_of_output_channels, flag_3D=flag_3D) if normalization_type else None
            padding_layer = ReflectionPaddingNd_Layer(0, flag_3D);
            extra_1x1_conv_block1 = ConvNd_layer(number_of_input_channels, number_of_input_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, flag_3D=flag_3D)
            extra_1x1_conv_block2 = ConvNd_layer(number_of_output_channels * len(kernel_size_list), number_of_output_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, flag_3D=flag_3D)
            if initialization_method == 'combined':
                combined_dirac_and_xavier_initialization(extra_1x1_conv_block1);
                combined_dirac_and_xavier_initialization(extra_1x1_conv_block2);
            elif initialization_method == 'dirac':
                nn.init.dirac_(extra_1x1_conv_block1.weight)
                nn.init.dirac_(extra_1x1_conv_block2.weight)
            elif initialization_method == 'xavier':
                nn.init.xavier_normal_(extra_1x1_conv_block1.weight)
                nn.init.xavier_normal_(extra_1x1_conv_block2.weight)
            full_basic_conv_block = nn.Sequential(extra_1x1_conv_block1, normalization_layer, activation_layer,
                                                  convolution_layer,  # These already have the normaliation and activation layer inside of them and i don't have to differentiate between normalization/activation at each cell and all of them together
                                                  extra_1x1_conv_block2, normalization_layer2, activation_layer2)
            # (1). Vertical IIR:
            if flag_IIR_filter_vertical:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            if flag_IIR_filter_vertical_reverse:
                current_vertical_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_vertical, flag_direction='vertical', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_vertical_IIR_layer)
            # (2). Horizontal IIR:
            if flag_IIR_filter_horizontal:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=False)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            if flag_IIR_filter_horizontal_reverse:
                current_horizontal_IIR_layer = IIR_MagIC_V2(number_of_output_channels, IIR_filter_order=flag_IIR_filter_order_horizontal, flag_direction='horizontal', flag_reverse=True)
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_horizontal_IIR_layer)
            # (3). Regular Conv Post-IIR:
            if flag_IIR_filter_horizontal or flag_IIR_filter_vertical or flag_IIR_filter_vertical_reverse or flag_IIR_filter_horizontal_reverse:
                current_post_IIR_conv = grouped_conv_block(number_of_output_channels, number_of_output_channels, 3, 1, 1, (normalization_type == 'batch_normalization'), flag_3D=flag_3D)  # TODO: probably should use grouped convolution to save memory and be like magic
                full_basic_conv_block = nn.Sequential(full_basic_conv_block, current_post_IIR_conv)
            full_basic_conv_block = Input_Output_Sum_Block_With_StraightThrough_Projection_Block(full_basic_conv_block, number_of_output_channels=number_of_output_channels, residual_scale=1, flag_3D=flag_3D)
        #############################################################################################################################




        #######################################################################
        ### SFT Wrapper: ###
        # TODO: right now i put it BEFORE the conv_offset....maybe i should have a choice???
        if flag_SuperBlock_SFT:
            full_basic_conv_block = SFT_General_Wrapper(full_basic_conv_block,
                                                        flag_use_outside_conditional=flag_SuperBlock_SFT_use_outside_conditional,
                                                        flag_shift=flag_SuperBlock_SFT_shift,
                                                        flag_scale=flag_SuperBlock_SFT_scale,
                                                        flag_SFT_convs_mix=flag_SuperBlock_SFT_SFT_convs_mix,
                                                        flag_SFT_same_on_all_channels=flag_SuperBlock_SFT_same_on_all_channels,
                                                        flag_SFT_add_y_to_output=flag_SuperBlock_SFT_add_y_to_output,
                                                        )
        self.flag_SuperBlock_SFT = flag_SuperBlock_SFT;
        #######################################################################



        #####################################################################
        ### Add Conv_Offset to the start of the super_block if so wanted: ###
        conv_offset_layer = None
        if flag_deformable_for_each_sub_block_or_for_super_block == 'super_block' and flag_deformable_convolution == True and flag_deformable_convolution_before_or_after_main_convolution == 'before':
            ###
            if flag_deformable_convolution_version == 'v1':  # github bilinear
                conv_offset_layer = ConvOffset2d_v1(number_of_input_channels, kernel_size=3, init_normal_stddev=0.01, flag_same_on_all_channels=flag_deformable_same_on_all_channels)
            if flag_deformable_convolution_version == 'v3':  # my bicubic
                conv_offset_layer = ConvOffset2d_v3_Groups_Conditional_SFT(number_of_deformable_groups=flag_deformable_number_of_deformable_groups,
                                                                           flag_use_outside_conditional=flag_deformable_SFT_use_outside_conditional,
                                                                           flag_base_convs_mix=flag_deformable_SFT_base_convs_mix,
                                                                           flag_SFT_convs_mix=flag_deformable_SFT_SFT_convs_mix,
                                                                           flag_shift=flag_deformable_SFT_shift,
                                                                           flag_scale=flag_deformable_SFT_scale,
                                                                           flag_add_y_to_output=flag_deformable_SFT_add_y_to_output,
                                                                           kernel_size=flag_deformable_kernel_size,
                                                                           flag_same_on_all_channels=flag_deformable_SFT_same_on_all_channels,  # TODO: change this parameter's name to flag_SFT_same_on_all_channels because it's original use is taken care of by deformable groups
                                                                           flag_automatic=True,  # flag_deformable_number_of_channels_in_group1=-1,
                                                                           # flag_deformable_number_of_channels_in_group1=-1,  # for instance, i might want 2 UNEQUAL deformable groups
                                                                           )
            # ### Add the conv_offset to the start: ###
            # full_basic_conv_block = nn.Sequential(conv_offset_layer, full_basic_conv_block);
        ###############################################################################################################################################################################################


        ################################################################################################################################################################################################
        ### Add Conv_Offset to the end of the super_block if so wanted: ###
        if flag_deformable_for_each_sub_block_or_for_super_block == 'super_block' and flag_deformable_convolution == True and flag_deformable_convolution_before_or_after_main_convolution == 'after':
            ###
            if flag_deformable_convolution_version == 'v1':  # github bilinear
                conv_offset_layer = ConvOffset2d_v1(number_of_output_channels, kernel_size=3, init_normal_stddev=0.01, flag_same_on_all_channels=flag_deformable_same_on_all_channels)
            if flag_deformable_convolution_version == 'v3':  # my bicubic
                # conv_offset_layer = ConvOffset2d_v3(number_of_output_channels, kernel_size=3, init_normal_stddev=0.01, flag_same_on_all_channels=flag_deformable_same_on_all_channels)
                conv_offset_layer = ConvOffset2d_v3_Groups_Conditional_SFT(number_of_deformable_groups=flag_deformable_number_of_deformable_groups,
                                                                           flag_use_outside_conditional=flag_deformable_SFT_use_outside_conditional,
                                                                           flag_base_convs_mix=flag_deformable_SFT_base_convs_mix,
                                                                           flag_SFT_convs_mix=flag_deformable_SFT_SFT_convs_mix,
                                                                           flag_shift=flag_deformable_SFT_shift,
                                                                           flag_scale=flag_deformable_SFT_scale,
                                                                           flag_add_y_to_output=flag_deformable_SFT_add_y_to_output,
                                                                           kernel_size=flag_deformable_kernel_size,
                                                                           flag_same_on_all_channels=flag_deformable_SFT_same_on_all_channels,  # TODO: change this parameter's name to flag_SFT_same_on_all_channels because it's original use is taken care of by deformable groups
                                                                           flag_automatic=True,  # flag_deformable_number_of_channels_in_group1=-1,
                                                                           # flag_deformable_number_of_channels_in_group1=-1,  # for instance, i might want 2 UNEQUAL deformable groups
                                                                           )
            # ### Add the conv_offset to the end: ###
            # full_basic_conv_block = nn.Sequential(full_basic_conv_block, conv_offset_layer)
        ####################################################################################################################################################################################################


        #########################################################
        ### Initialize Object Variables Needed Going Forward: ###
        self.flag_deformable_convolution = flag_deformable_convolution
        self.flag_deformable_convolution_before_or_after_main_convolution = flag_deformable_convolution_before_or_after_main_convolution
        self.conv_offset_layer = conv_offset_layer;
        self.full_basic_conv_block = full_basic_conv_block;



    def forward(self, x, y_cell_conditional=None, y_deformable_conditional=None):
        ### TODO: MAYBE DELETE LATER!!! ###
        self = self.to(x.device)
        # output = x.clone()  #TODO: should i use x or x.clone()?   maybe this is a problem in and of itself???.....
        output = x  #TODO: should i use x or x.clone()?   maybe this is a problem in and of itself???.....


        # ### TODO: temporary override: ###
        # output = self.full_basic_conv_block(x)

        ### Forward Through Deformable Convolution Block: ###
        if self.flag_deformable_convolution and self.flag_deformable_convolution_before_or_after_main_convolution == 'before':
            output = self.conv_offset_layer(output, y_deformable_conditional)

        ### Forward Through Conv-Layers (maybe wrapped with SFT_General_Block): ###
        if self.flag_SuperBlock_SFT:
            # print('conv2d multiple kernels' + str(y_cell_conditional.shape))
            output = self.full_basic_conv_block(output,y_cell_conditional)
        else:
            # output = self.full_basic_conv_block(output.clone()) #TODO: return to this at some point
            output = self.full_basic_conv_block(output) #TODO: return to this at some point
            # output = self.full_basic_conv_block(x)

        ### Forward Through Deformable Convolution Block: ###
        if self.flag_deformable_convolution and self.flag_deformable_convolution_before_or_after_main_convolution == 'after':
            output = self.conv_offset_layer(output,y_deformable_conditional)


        return output





def ConvNd_layer(in_channels = 3,
             out_channels = 3,
             kernel_size = 3,
             stride = 1,
             padding = 0,
             dilation = 1,
             groups = 1,
             bias = True,
             padding_mode = 'zeros', flag_3D=False):
    if flag_3D:
        return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

def ReflectionPaddingNd_Layer(padding_needed, flag_3D=False):
    if flag_3D:
        return nn.ReflectionPad3d(padding_needed)
    else:
        return nn.ReflectionPad2d(padding_needed)

def ZeroPaddingNd_Layer(padding_needed, flag_3D=False):
    if flag_3D:
        return nn.ReflectionPad3d(padding_needed)
    else:
        return nn.ReflectionPad2d(padding_needed)


class input_adaptive_linear(nn.Module):
    def __init__(self, output_channels):
        self.Linear = None
        self.output_channels = output_channels
    def forward(self, x):
        if self.Linear is None:
            self.Linear = nn.Linear(x.shape[1], self.output_channels)
        return self.Linear(x)

class input_adaptive_ConvNd(nn.Module):
    # Initialize this with a module
    def __init__(self,
                 number_of_output_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 initialization_method='xavier',
                 flag_learnable_amplitude_strategy='none',  # 'none', 'inputs', 'outputs', 'total'
                 bias_value=None,
                 bias=True, flag_3D=False
                 ):
        super(input_adaptive_ConvNd, self).__init__()
        self.conv_layer = None
        self.flag_3D = flag_3D
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.initialization_method = initialization_method
        self.flag_learnable_amplitude_strategy = flag_learnable_amplitude_strategy
        self.number_of_output_channels = number_of_output_channels
        self.bias = bias
        self.padding = get_valid_padding(kernel_size,dilation)
    def forward(self, x):
        if self.conv_layer is None:
            if self.flag_3D:
                self.conv_layer = nn.Conv3d(x.shape[1], self.number_of_output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=self.bias).to(x.device)
            else:
                self.conv_layer = nn.Conv2d(x.shape[1], self.number_of_output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=self.bias).to(x.device)

        return self.conv_layer(x)


### Learnable Amplitude nn.Conv2d: ###
class ConvNd_Learnable_Amplitude(nn.Module):
    # Initialize this with a module
    def __init__(self, number_of_input_channels,
                 number_of_output_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 initialization_method='xavier',
                 flag_learnable_amplitude_strategy='none',  # 'none', 'inputs', 'outputs', 'total'
                 bias_value=None,
                 bias=True, flag_3D=False
                 ):
        super(ConvNd_Learnable_Amplitude, self).__init__()
        self.number_of_input_channels = number_of_input_channels
        self.number_of_output_channels = number_of_output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.flag_learnable_amplitude_strategy = flag_learnable_amplitude_strategy
        self.flag_3D = flag_3D

        if flag_learnable_amplitude_strategy == 'none':
            self.Conv_Layers = ConvNd_layer(number_of_input_channels, number_of_output_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias, flag_3D=flag_3D)  #TODO: Notice i put bias=False just to equalize to paper
        elif flag_learnable_amplitude_strategy == 'total':
            self.Conv_Layers = nn.ModuleList()
            self.A_input_channels_list = nn.ParameterList()
            for i in arange(number_of_output_channels):
                # self.A_input_channels = nn.Parameter(torch.Tensor(1, number_of_input_channels * number_of_output_channels, 1, 1))
                self.A_input_channels_list.append(nn.Parameter(torch.ones(1, number_of_input_channels, 1, 1)))
                current_layer = ConvNd_layer(number_of_input_channels, 1, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, flag_3D=flag_3D)
                ### Conv Layer Initialization: ####
                if initialization_method == 'combined':
                    combined_dirac_and_xavier_initialization(current_layer.weight);
                elif initialization_method == 'dirac':
                    nn.init.dirac_(current_layer.weight)
                elif initialization_method == 'xavier':
                    nn.init.xavier_normal_(current_layer.weight)
                ### Bias Conv Layer: ###
                if bias_value is not None:
                    init.constant(current_layer.bias, bias_value)
                ### Append: ###
                self.Conv_Layers.append(current_layer)
        elif flag_learnable_amplitude_strategy == 'inputs':
            self.A_input_channels = nn.Parameter(torch.ones(1, number_of_input_channels, 1, 1))
            self.Conv_Layers = ConvNd_layer(number_of_input_channels, number_of_output_channels ,kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, flag_3D=flag_3D)
        elif flag_learnable_amplitude_strategy == 'outputs':
            self.A_input_channels = nn.Parameter(torch.ones(1, number_of_output_channels, 1, 1))
            self.Conv_Layers = ConvNd_layer(number_of_input_channels, number_of_output_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, flag_3D=flag_3D)

        # ### Conv Layer Initialization: ####
        # #TODO: uncomment this at the end hopefully
        # if flag_learnable_amplitude_strategy != 'total':
        #     if initialization_method == 'combined':
        #         combined_dirac_and_xavier_initialization(self.Conv_Layers);
        #     elif initialization_method == 'dirac':
        #         nn.init.dirac_(self.Conv_Layers.weight)
        #     elif initialization_method == 'xavier':
        #         nn.init.xavier_normal_(self.Conv_Layers.weight)
        #     ### Bias Conv Layer: ###
        #     if bias_value is not None:
        #         init.constant(self.Conv_Layers.bias, bias_value)
        #
        # ### Make .weight Attribute to make it compatible: ###
        # self.weight = self.Conv_Layers.weight;

    # Elementwise sum the output of a submodule to its input
    def forward(self, x):
        if self.flag_learnable_amplitude_strategy == 'none':
            return self.Conv_Layers(x);
        elif self.flag_learnable_amplitude_strategy == 'inputs':
            return self.Conv_Layers(self.A_input_channels * x)
        elif self.flag_learnable_amplitude_strategy == 'outputs':
            return self.A_input_channels * self.Conv_Layers(x)
        elif self.flag_learnable_amplitude_strategy == 'total':
            if self.flag_3D:
                output_tensor = torch.zeros((x.shape[0], self.number_of_output_channels, x.shape[2]-self.kernel_size//2, x.shape[3]-self.kernel_size//2, x.shape[4]-self.kernel_size//2))
            else:
                output_tensor = torch.zeros((x.shape[0], self.number_of_output_channels, x.shape[2]-self.kernel_size//2, x.shape[3]-self.kernel_size//2))
            for i in arange(self.number_of_output_channels):
                if i==0:
                    # output_tensor = self.A_input_channels[:,i*self.number_of_input_channels:(i+1)*self.number_of_input_channels,:,:] * self.Conv_Layers[i](x)
                    output_tensor = self.A_input_channels_list[i] * self.Conv_Layers[i](x)
                else:
                    # current_output_tensor = self.A_input_channels[:,i*self.number_of_input_channels:(i+1)*self.number_of_input_channels,:,:] * self.Conv_Layers[i](x)
                    # output_tensor = torch.cat([output_tensor,current_output_tensor],dim=1)

                    # output_tensor[:,i*self.number_of_output_channels:(i+1)*self.number_of_output_channels,:,:] = self.Conv_Layers[i](self.A_input_channels[:,i*self.number_of_input_channels:(i+1)*self.number_of_input_channels,:,:]  * x)
                    if self.flag_3D:
                        output_tensor[:,i*self.number_of_output_channels:(i+1)*self.number_of_output_channels,:,:,:] = self.Conv_Layers[i](self.A_input_channels_list[i] * x)
                    else:
                        output_tensor[:,i*self.number_of_output_channels:(i+1)*self.number_of_output_channels,:,:] = self.Conv_Layers[i](self.A_input_channels_list[i] * x)
            return output_tensor

        return 1;





# Basic Conv Block (every project has to have it's own....):
def Conv_Block(number_of_input_channels,
               number_of_output_channels,
               kernel_size,
               stride=1,
               dilation=1,
               groups=1,
               bias=True,
               padding_type='reflect',
               # don't do padding .... this means the output size will be different from input size right?
               normalization_function=None,
               activation_function='relu',
               mode='CNA',
               initialization_method = 'combined',
               ### Deformable: ###
               flag_deformable_convolution = False,
               flag_deformable_convolution_version = 'v3',
               flag_deformable_convolution_before_or_after_main_convolution = 'before', #'before' / 'after'
               flag_deformable_convolution_modulation = True,
               flag_deformable_number_of_deformable_groups = -1,
               flag_deformable_same_on_all_channels = False,
               ### Deformable SFT: ###
               flag_deformable_use_outside_conditional = False,
               flag_deformable_base_convs_mix = 'x',
               flag_deformable_SFT_convs_mix = 'x',
               flag_deformable_shift = False,
               flag_deformable_scale = False,
               flag_deformable_add_y_to_output = False,
               flag_deformable_kernel_size = 5,
               ### Block Type: ###
               flag_block_type = 'simple', #'simple'/'standard_residual'/'131_residual'
               bias_value = None,
               ### Learnable Amplitude or not: ###
               flag_learnable_amplitude_strategy = 'none', flag_3D = False
               ):




    #TODO: add possibility of Deformable Convolution!
    """
    Conv layer with padding, normalization, activation
    mode: CNA: Conv -> Norm -> Act
          NAC: Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """

    # Check that input mode is one that's implemented
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [%s]' % mode

    # Pre-Padding:
    # (1). Get the padding size needed to pad the input with in order for the convolution operation output to be the same size as the origianl input:
    if type(kernel_size)==tuple:
        padding_needed = []
        for i in arange(len(kernel_size)):
            current_kernel_size = kernel_size[i]
            current_dilation = dilation[i]
            current_padding_needed = get_valid_padding(current_kernel_size, current_dilation)
            # padding_needed = padding_needed + [current_padding_needed, current_padding_needed]
            padding_needed = [current_padding_needed, current_padding_needed] + padding_needed
    else:
        padding_needed = get_valid_padding(kernel_size, dilation)
    print('bla')
    # (2). Actually do the padding if wanted:
    #TODO: make possible zero padding and not only reflection padding - should i use padding?
    # padding_layer = nn.ReflectionPad2d(padding_needed);
    # padding_layer2 = nn.ReflectionPad2d(padding_needed);
    if flag_3D:
        padding_layer = nn.ConstantPad3d(padding_needed,0);
        padding_layer2 = nn.ConstantPad3d(padding_needed,0);
    else:
        padding_layer = nn.ZeroPad2d(padding_needed);
        padding_layer2 = nn.ZeroPad2d(padding_needed);



    ### Deformable Convolution: ###
    #TODO: generalize the deformable convolutions to be able to return 3D Convs and deformations
    if flag_deformable_convolution:
        if flag_deformable_convolution_version == 'v1':
            conv_offset_layer = ConvOffset2d_v1(number_of_input_channels, kernel_size=3, init_normal_stddev=0.01, flag_same_on_all_channels=flag_deformable_same_on_all_channels)
        elif flag_deformable_convolution_version == 'v2':
            conv_offset_layer = ConvOffset2d_v2(number_of_input_channels, kernel_size=3, stride=1, bias=None, modulation=flag_deformable_convolution_modulation) # TODO: can't choose number of groups with the Non-CUDA deform_conv_v2!!! - for that i should install the CUDA version
        elif flag_deformable_convolution_version == 'v3':
            conv_offset_layer = ConvOffset2d_v3_Groups_Conditional_SFT(number_of_deformable_groups=flag_deformable_number_of_deformable_groups,
                                                                       flag_use_outside_conditional=flag_deformable_use_outside_conditional,
                                                                       flag_base_convs_mix=flag_deformable_base_convs_mix,
                                                                       flag_SFT_convs_mix=flag_deformable_SFT_convs_mix,
                                                                       flag_shift=flag_deformable_shift,
                                                                       flag_scale=flag_deformable_scale,
                                                                       flag_add_y_to_output = flag_deformable_add_y_to_output,
                                                                       kernel_size=flag_deformable_kernel_size,
                                                                       flag_same_on_all_channels=flag_deformable_same_on_all_channels,
                                                                       flag_automatic=True,
                                                                       # flag_deformable_number_of_channels_in_group1=-1,
                                                                       )



    ### Main Convolution Layer: ###
    #TODO: NOTICE!!!! MODELS WHICH HAVE BEEN TRAINED WITH nn.Conv2d WILL NOT WORK ONCE TRANSFERRED TO ConvNd_Learnable_Amplitude
    # convolution_layer = ConvNd_Learnable_Amplitude(number_of_input_channels, number_of_output_channels,
    #                                                kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups, initialization_method=initialization_method, flag_learnable_amplitude_strategy=flag_learnable_amplitude_strategy,bias=bias, flag_3D=flag_3D)
    if flag_3D==False:
        convolution_layer = nn.Conv2d(number_of_input_channels,
                                      number_of_output_channels,
                                      kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias, groups=groups)
    else:
        convolution_layer = nn.Conv3d(number_of_input_channels,
                                      number_of_output_channels,
                                      kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias, groups=groups)


    ### Activation Layer: ###
    activation_layer = Activation_Function(activation_function) if activation_function else None
    activation_layer2 = Activation_Function(activation_function) if activation_function else None
    activation_layer3 = Activation_Function(activation_function) if activation_function else None

    # ### Conv Layer Initialization: ####
    #TODO: add kaiming initialization and BN and Linear initializations
    # if initialization_method == 'combined':
    #     combined_dirac_and_xavier_initialization(convolution_layer);
    # elif initialization_method == 'dirac':
    #     nn.init.dirac_(convolution_layer.weight)
    # elif initialization_method == 'xavier':
    #     nn.init.xavier_normal_(convolution_layer.weight)

    # ### Bias Conv Layer: ###
    # if bias_value is not None:
    #     init.constant(convolution_layer.bias, bias_value)

    ### Normalization: ###
    normalization_layer = Normalization_Function(normalization_function, number_of_output_channels, flag_3D) if normalization_function else None
    normalization_layer2 = Normalization_Function(normalization_function, number_of_output_channels, flag_3D) if normalization_function else None
    normalization_layer3 = Normalization_Function(normalization_function, number_of_output_channels, flag_3D) if normalization_function else None


    ### Combine Together To Full Basic Conv Block: ###
    #TODO: accomodate NAC/CNA possibilities for standard_residual and 131_residual as well
    if flag_block_type=='simple':
        if mode=='CNA':
            ### CNA: ###
            normalization_layer = Normalization_Function(normalization_function, number_of_output_channels, flag_3D) if normalization_function else None
            full_basic_conv_block = nn.Sequential(padding_layer, convolution_layer, normalization_layer, activation_layer)
        if mode=='NAC':
            ### NAC: ###
            normalization_layer = Normalization_Function(normalization_function, number_of_input_channels, flag_3D) if normalization_function else None
            full_basic_conv_block = nn.Sequential(normalization_layer, activation_layer, padding_layer, convolution_layer)

    if flag_block_type=='standard_residual':
        extra_conv_block = ConvNd_layer(number_of_output_channels,
                                     number_of_output_channels,
                                     kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias, groups=groups)
        full_basic_conv_block = nn.Sequential(padding_layer, convolution_layer, normalization_layer, activation_layer, padding_layer2, extra_conv_block)
        full_basic_conv_block = Input_Output_Sum_Block_With_StraightThrough_Projection_Block(full_basic_conv_block, number_of_output_channels=number_of_output_channels, residual_scale=1)

    if flag_block_type=='131_residual':
        extra_1x1_conv_block1 = ConvNd_layer(number_of_input_channels, number_of_input_channels, kernel_size=1, stride=stride, padding=0, dilation=dilation, bias=bias, groups=groups)
        extra_1x1_conv_block2 = ConvNd_layer(number_of_output_channels, number_of_output_channels, kernel_size=1, stride=stride, padding=0, dilation=dilation, bias=bias, groups=groups)
        full_basic_conv_block = nn.Sequential(extra_1x1_conv_block1, normalization_layer, activation_layer, padding_layer, convolution_layer, normalization_layer2, activation_layer2, extra_1x1_conv_block2, normalization_layer3, activation_layer3)
        full_basic_conv_block = Input_Output_Sum_Block_With_StraightThrough_Projection_Block(full_basic_conv_block, number_of_output_channels=number_of_output_channels, residual_scale=1)


    ### Add deformable convolution if so wanted: ###
    if flag_deformable_convolution:
        if flag_deformable_convolution_before_or_after_main_convolution == 'before':
            full_basic_conv_block = nn.Sequential(conv_offset_layer, full_basic_conv_block)
        elif flag_deformable_convolution_before_or_after_main_convolution == 'after':
            full_basic_conv_block = nn.Sequential(full_basic_conv_block, conv_offset_layer)


    ### TODO: the initialization from githun, decide whether to embrace it or not
    ### Weight-Initialize Layers: ###
    for m in get_network_layers_list_flat(full_basic_conv_block):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            # if initialization_method == 'dirac':
            #     nn.init.dirac_(convolution_layer.weight)
            # else:
            #     nn.init.kaiming_normal_(m.weight)
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)

    # if initialization_method == 'combined':
    #     combined_dirac_and_xavier_initialization(convolution_layer);
    # elif initialization_method == 'dirac':
    #     nn.init.dirac_(convolution_layer.weight)
    # elif initialization_method == 'xavier':
    #     nn.init.xavier_normal_(convolution_layer.weight)

    return full_basic_conv_block






class Sequential_Conv_Block_General(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self,
                 number_of_input_channels,
                 number_of_output_channels,
                 kernel_sizes,
                 strides=1,
                 dilations=1,
                 groups=1,
                 padding_type='reflect',
                 normalization_function=None,
                 activation_function='relu',
                 mode='CNA',
                 initialization_method='xavier',
                 flag_dense=False,
                 flag_resnet=False,
                 flag_concat=False,
                 stack_residual_scale=1,
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
                 flag_super_cell_block_type='concat',  # 'concat' / '131' / '131_collapse' / 'concat_standard_residual' / '131_residual' / '131_collapse_residual'
                 ####################################
                 ### IIR Filters: ###
                 flag_IIR_filter_horizontal = False,
                 flag_IIR_filter_horizontal_reverse = False,
                 flag_IIR_filter_order_horizontal = [1],
                 flag_IIR_filter_vertical = False,
                 flag_IIR_filter_vertical_reverse = False,
                 flag_IIR_filter_order_vertical = [1],
                 flag_identity_block = False,
                 bias = True,
                 flag_3D = False,
                 )                 :
        super(Sequential_Conv_Block_General, self).__init__()

        # TODO: change to a list to make it a single list of layers instead of a sequential in sequential object
        # TODO: add the possibility of Residual connection AFTER EACH LAYER (and not just over all the block)....
        # TODO: add SFT layer possibility....this is probably a great possibility to enhance non-linearity greatly but still being able to learn well
        # TODO: consider adding more self modulating layers like SFT....
        if type(number_of_output_channels) != list and type(number_of_output_channels) != tuple:
            number_of_output_channels = [number_of_output_channels]
        number_of_layers = len(number_of_output_channels);
        kernel_sizes = to_list_of_certain_size(kernel_sizes, number_of_layers)
        strides = to_list_of_certain_size(strides, number_of_layers)
        dilations = to_list_of_certain_size(dilations, number_of_layers)
        groups = to_list_of_certain_size(groups, number_of_layers)
        activation_function = to_list_of_certain_size(activation_function, number_of_layers)
        initialization_method = to_list_of_certain_size(initialization_method, number_of_layers)

        flag_SuperBlock_SFT = to_list_of_certain_size(flag_SuperBlock_SFT, number_of_layers)
        flag_SuperBlock_SFT_use_outside_conditional = to_list_of_certain_size(flag_SuperBlock_SFT_use_outside_conditional, number_of_layers)
        flag_SuperBlock_SFT_same_on_all_channels = to_list_of_certain_size(flag_SuperBlock_SFT_same_on_all_channels, number_of_layers)
        flag_SuperBlock_SFT_base_convs_mix = to_list_of_certain_size(flag_SuperBlock_SFT_base_convs_mix, number_of_layers)
        flag_SuperBlock_SFT_SFT_convs_mix = to_list_of_certain_size(flag_SuperBlock_SFT_SFT_convs_mix, number_of_layers)
        flag_SuperBlock_SFT_add_y_to_output = to_list_of_certain_size(flag_SuperBlock_SFT_add_y_to_output, number_of_layers)
        flag_SuperBlock_SFT_shift = to_list_of_certain_size(flag_SuperBlock_SFT_shift, number_of_layers)
        flag_SuperBlock_SFT_scale = to_list_of_certain_size(flag_SuperBlock_SFT_scale, number_of_layers)

        flag_deformable_convolution = to_list_of_certain_size(flag_deformable_convolution, number_of_layers)
        flag_deformable_convolution_version = to_list_of_certain_size(flag_deformable_convolution_version, number_of_layers)
        flag_deformable_convolution_before_or_after_main_convolution = to_list_of_certain_size(flag_deformable_convolution_before_or_after_main_convolution, number_of_layers)
        flag_deformable_convolution_modulation = to_list_of_certain_size(flag_deformable_convolution_modulation, number_of_layers)
        flag_deformable_kernel_size = to_list_of_certain_size(flag_deformable_kernel_size, number_of_layers)
        flag_deformable_number_of_deformable_groups = to_list_of_certain_size(flag_deformable_number_of_deformable_groups, number_of_layers)
        flag_deformable_number_of_channels_in_group1 = to_list_of_certain_size(flag_deformable_number_of_channels_in_group1, number_of_layers)
        flag_deformable_same_on_all_channels = to_list_of_certain_size(flag_deformable_same_on_all_channels, number_of_layers)

        flag_deformable_SFT_use_outside_conditional = to_list_of_certain_size(flag_deformable_SFT_use_outside_conditional, number_of_layers)
        flag_deformable_SFT_same_on_all_channels = to_list_of_certain_size(flag_deformable_SFT_same_on_all_channels, number_of_layers)
        flag_deformable_SFT_base_convs_mix = to_list_of_certain_size(flag_deformable_SFT_base_convs_mix, number_of_layers)
        flag_deformable_SFT_SFT_convs_mix = to_list_of_certain_size(flag_deformable_SFT_SFT_convs_mix, number_of_layers)
        flag_deformable_SFT_shift = to_list_of_certain_size(flag_deformable_SFT_shift, number_of_layers)
        flag_deformable_SFT_scale = to_list_of_certain_size(flag_deformable_SFT_scale, number_of_layers)
        flag_deformable_SFT_add_y_to_output = to_list_of_certain_size(flag_deformable_SFT_add_y_to_output, number_of_layers)
        flag_deformable_for_each_sub_block_or_for_super_block = to_list_of_certain_size(flag_deformable_for_each_sub_block_or_for_super_block, number_of_layers)

        flag_single_cell_block_type = to_list_of_certain_size(flag_single_cell_block_type, number_of_layers)
        flag_super_cell_block_type = to_list_of_certain_size(flag_super_cell_block_type, number_of_layers)

        flag_IIR_filter_horizontal = to_list_of_certain_size(flag_IIR_filter_horizontal, number_of_layers)
        flag_IIR_filter_horizontal_reverse = to_list_of_certain_size(flag_IIR_filter_horizontal_reverse, number_of_layers)
        flag_IIR_filter_order_horizontal = to_list_of_certain_size(flag_IIR_filter_order_horizontal, number_of_layers)
        flag_IIR_filter_vertical = to_list_of_certain_size(flag_IIR_filter_vertical, number_of_layers)
        flag_IIR_filter_vertical_reverse = to_list_of_certain_size(flag_IIR_filter_vertical_reverse, number_of_layers)
        flag_IIR_filter_order_vertical = to_list_of_certain_size(flag_IIR_filter_order_vertical, number_of_layers)
        bias = to_list_of_certain_size(bias, number_of_layers)
        normalization_function = to_list_of_certain_size(normalization_function, number_of_layers)



        ### Get Layers List: ###
        # layers_list = []
        layers_list = nn.ModuleList()  #TODO: decide whether to use nn.ModuleList() or simple list. (*). I think i need a nn.ModuleList() because at the end we're using pile_modules_on_top_of_each_other_layer which takes list elements once at a time which i saw needs nn.ModuleList()
        # (1). First Layer:
        if flag_dense:
            layers_list.append(Input_Output_Concat_Block(Conv2d_Multiple_Kernels(number_of_input_channels, number_of_output_channels[0], kernel_sizes[0], strides[0], dilations[0], groups[0], bias=bias[0], mode=mode, padding_type='reflect', normalization_type=normalization_function[0], activation_type=activation_function[0], initialization_method=initialization_method[0], flag_deformable_convolution=flag_deformable_convolution[0], flag_deformable_convolution_version=flag_deformable_convolution_version[0], flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution[0],  # 'before' / 'after'
                                                                                 flag_deformable_convolution_modulation=flag_deformable_convolution_modulation[0], flag_single_cell_block_type=flag_single_cell_block_type[0], flag_super_cell_block_type=flag_super_cell_block_type[0], flag_deformable_for_each_sub_block_or_for_super_block=flag_deformable_for_each_sub_block_or_for_super_block[0], flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels[0], flag_3D=flag_3D)))
        else:
            layers_list.append(Conv2d_Multiple_Kernels(number_of_input_channels, number_of_output_channels[0], kernel_sizes[0], strides[0], dilations[0], groups[0], bias=bias[0], mode=mode, padding_type='reflect', normalization_type=normalization_function[0], activation_type=activation_function[0], initialization_method=initialization_method[0], flag_deformable_convolution=flag_deformable_convolution[0], flag_deformable_convolution_version=flag_deformable_convolution_version[0], flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution[0],  # 'before' / 'after'
                                                       flag_deformable_convolution_modulation=flag_deformable_convolution_modulation[0],
                                                       flag_single_cell_block_type=flag_single_cell_block_type[0],
                                                       flag_super_cell_block_type=flag_super_cell_block_type[0],
                                                       flag_deformable_for_each_sub_block_or_for_super_block=flag_deformable_for_each_sub_block_or_for_super_block[0],
                                                       flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels[0],
                                                       flag_SuperBlock_SFT=flag_SuperBlock_SFT[0],
                                                       flag_SuperBlock_SFT_use_outside_conditional=flag_SuperBlock_SFT_use_outside_conditional[0],
                                                       flag_SuperBlock_SFT_same_on_all_channels=flag_SuperBlock_SFT_same_on_all_channels[0],
                                                       flag_SuperBlock_SFT_base_convs_mix=flag_SuperBlock_SFT_base_convs_mix[0],  # 'x', 'y', 'xy'
                                                       flag_SuperBlock_SFT_SFT_convs_mix=flag_SuperBlock_SFT_SFT_convs_mix[0],  # 'x', 'y', 'xy'
                                                       flag_SuperBlock_SFT_add_y_to_output=flag_SuperBlock_SFT_add_y_to_output[0],
                                                       flag_SuperBlock_SFT_shift=flag_SuperBlock_SFT_shift[0],
                                                       flag_SuperBlock_SFT_scale=flag_SuperBlock_SFT_scale[0],
                                                       flag_deformable_SFT_use_outside_conditional=flag_deformable_SFT_use_outside_conditional[0],
                                                       flag_deformable_SFT_same_on_all_channels=flag_deformable_SFT_same_on_all_channels[0],  ### The interactive shift/scale convs by default spits-out x.shape[1] number_of_channels....but maybe to save memory and parameters we want the same interaction on all channels
                                                       flag_deformable_SFT_base_convs_mix=flag_deformable_SFT_base_convs_mix[0],
                                                       flag_deformable_SFT_SFT_convs_mix=flag_deformable_SFT_SFT_convs_mix[0],
                                                       flag_deformable_SFT_shift=flag_deformable_SFT_shift[0],
                                                       flag_deformable_SFT_scale=flag_deformable_SFT_scale[0],
                                                       flag_deformable_SFT_add_y_to_output=flag_deformable_SFT_add_y_to_output[0],
                                                       flag_IIR_filter_horizontal = flag_IIR_filter_horizontal[0],
                                                       flag_IIR_filter_horizontal_reverse= flag_IIR_filter_horizontal_reverse[0],
                                                       flag_IIR_filter_order_horizontal = flag_IIR_filter_order_horizontal[0],
                                                       flag_IIR_filter_vertical = flag_IIR_filter_vertical[0],
                                                       flag_IIR_filter_vertical_reverse = flag_IIR_filter_vertical_reverse[0],
                                                       flag_IIR_filter_order_vertical = flag_IIR_filter_order_vertical[0], flag_3D=flag_3D
                                                       )

                               )
            if 'collapse' in flag_super_cell_block_type[0]:
                input_channels_factor = 1;
            else:
                input_channels_factor = len(kernel_sizes[0])

        # (2). Rest of the Layers:
        number_of_concatenated_channels_so_far = 0;
        if flag_dense:
            number_of_concatenated_channels_so_far += number_of_input_channels
            for i in arange(len(kernel_sizes) - 1):
                ### Get number of input channels to current conv_block according to previous block's flags/outputs: ###
                if 'collapse' in flag_super_cell_block_type[i]:
                    input_channels_factor = 1;
                else:
                    input_channels_factor = len(kernel_sizes[i])
                ###
                layers_list.append(Input_Output_Concat_Block(# TODO: correct the default number_of_output_channels[i]*len(kernel_sizes[i]) because i can now choose to collapse that to only number_of_output_channels!!@$#@$#%^$
                    Conv2d_Multiple_Kernels(number_of_output_channels[i] * input_channels_factor + number_of_concatenated_channels_so_far, number_of_output_channels[i + 1], kernel_sizes[i + 1], strides[i + 1], dilations[i + 1], groups[i + 1], bias=bias[i+1], mode=mode, padding_type='reflect', normalization_type=normalization_function[i+1], activation_type=activation_function[i + 1], initialization_method=initialization_method[i + 1], flag_deformable_convolution=flag_deformable_convolution[i + 1], flag_deformable_convolution_version=flag_deformable_convolution_version[i + 1], flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution[i + 1],  # 'before' / 'after'
                                            flag_deformable_convolution_modulation=flag_deformable_convolution_modulation[i + 1], flag_single_cell_block_type=flag_single_cell_block_type[i + 1],
                                            flag_super_cell_block_type=flag_super_cell_block_type[i + 1], flag_deformable_for_each_sub_block_or_for_super_block=flag_deformable_for_each_sub_block_or_for_super_block[i + 1],
                                            flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels[i + 1],
                                            flag_SuperBlock_SFT=flag_SuperBlock_SFT[i+1],
                                            flag_SuperBlock_SFT_use_outside_conditional=flag_SuperBlock_SFT_use_outside_conditional[i+1],
                                            flag_SuperBlock_SFT_same_on_all_channels=flag_SuperBlock_SFT_same_on_all_channels[i+1],
                                            flag_SuperBlock_SFT_base_convs_mix=flag_SuperBlock_SFT_base_convs_mix[i+1],  # 'x', 'y', 'xy'
                                            flag_SuperBlock_SFT_SFT_convs_mix=flag_SuperBlock_SFT_SFT_convs_mix[i+1],  # 'x', 'y', 'xy'
                                            flag_SuperBlock_SFT_add_y_to_output=flag_SuperBlock_SFT_add_y_to_output[i+1],
                                            flag_SuperBlock_SFT_shift=flag_SuperBlock_SFT_shift[i+1],
                                            flag_SuperBlock_SFT_scale=flag_SuperBlock_SFT_scale[i+1],
                                            flag_deformable_SFT_use_outside_conditional=flag_deformable_SFT_use_outside_conditional[i+1],
                                            flag_deformable_SFT_same_on_all_channels=flag_deformable_SFT_same_on_all_channels[i+1],  ### The interactive shift/scale convs by default spits-out x.shape[1] number_of_channels....but maybe to save memory and parameters we want the same interaction on all channels
                                            flag_deformable_SFT_base_convs_mix=flag_deformable_SFT_base_convs_mix[i+1],
                                            flag_deformable_SFT_SFT_convs_mix=flag_deformable_SFT_SFT_convs_mix[i+1],
                                            flag_deformable_SFT_shift=flag_deformable_SFT_shift[i+1],
                                            flag_deformable_SFT_scale=flag_deformable_SFT_scale[i+1],
                                            flag_deformable_SFT_add_y_to_output=flag_deformable_SFT_add_y_to_output[i+1],
                                            flag_IIR_filter_horizontal=flag_IIR_filter_horizontal[i+1],
                                            flag_IIR_filter_horizontal_reverse=flag_IIR_filter_horizontal_reverse[i+1],
                                            flag_IIR_filter_order_horizontal=flag_IIR_filter_order_horizontal[i+1],
                                            flag_IIR_filter_vertical=flag_IIR_filter_vertical[i+1],
                                            flag_IIR_filter_vertical_reverse=flag_IIR_filter_vertical_reverse[i+1],
                                            flag_IIR_filter_order_vertical=flag_IIR_filter_order_vertical[i+1], flag_3D=flag_3D
                                            )))
                number_of_concatenated_channels_so_far += number_of_output_channels[i] * len(kernel_sizes[i])
        else:
            for i in arange(len(kernel_sizes) - 1):
                ### Get number of input channels to current conv_block according to previous block's flags/outputs: ###
                if 'collapse' in flag_super_cell_block_type[i]:
                    input_channels_factor = 1;
                else:
                    input_channels_factor = len(kernel_sizes[i])
                ###
                layers_list.append(Conv2d_Multiple_Kernels(number_of_output_channels[i] * input_channels_factor, number_of_output_channels[i + 1], kernel_sizes[i + 1], strides[i + 1], dilations[i + 1], groups[i + 1], bias=bias[i+1], mode=mode, padding_type='reflect', normalization_type=normalization_function[i+1], activation_type=activation_function[i + 1], initialization_method=initialization_method[i + 1], flag_deformable_convolution=flag_deformable_convolution[i + 1], flag_deformable_convolution_version=flag_deformable_convolution_version[i + 1], flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution[i + 1],  # 'before' / 'after'
                                                           flag_deformable_convolution_modulation=flag_deformable_convolution_modulation[i + 1], flag_single_cell_block_type=flag_single_cell_block_type[i + 1],
                                                           flag_super_cell_block_type=flag_super_cell_block_type[i + 1], flag_deformable_for_each_sub_block_or_for_super_block=flag_deformable_for_each_sub_block_or_for_super_block[i + 1],
                                                           flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels[i + 1],
                                                           flag_SuperBlock_SFT=flag_SuperBlock_SFT[i+1],
                                                           flag_SuperBlock_SFT_use_outside_conditional=flag_SuperBlock_SFT_use_outside_conditional[i+1],
                                                           flag_SuperBlock_SFT_same_on_all_channels=flag_SuperBlock_SFT_same_on_all_channels[i+1],
                                                           flag_SuperBlock_SFT_base_convs_mix=flag_SuperBlock_SFT_base_convs_mix[i+1],  # 'x', 'y', 'xy'
                                                           flag_SuperBlock_SFT_SFT_convs_mix=flag_SuperBlock_SFT_SFT_convs_mix[i+1],  # 'x', 'y', 'xy'
                                                           flag_SuperBlock_SFT_add_y_to_output=flag_SuperBlock_SFT_add_y_to_output[i+1],
                                                           flag_SuperBlock_SFT_shift=flag_SuperBlock_SFT_shift[i+1],
                                                           flag_SuperBlock_SFT_scale=flag_SuperBlock_SFT_scale[i+1],
                                                           flag_deformable_SFT_use_outside_conditional=flag_deformable_SFT_use_outside_conditional[i+1],
                                                           flag_deformable_SFT_same_on_all_channels=flag_deformable_SFT_same_on_all_channels[i+1],  ### The interactive shift/scale convs by default spits-out x.shape[1] number_of_channels....but maybe to save memory and parameters we want the same interaction on all channels
                                                           flag_deformable_SFT_base_convs_mix=flag_deformable_SFT_base_convs_mix[i+1],
                                                           flag_deformable_SFT_SFT_convs_mix=flag_deformable_SFT_SFT_convs_mix[i+1],
                                                           flag_deformable_SFT_shift=flag_deformable_SFT_shift[i+1],
                                                           flag_deformable_SFT_scale=flag_deformable_SFT_scale[i+1],
                                                           flag_deformable_SFT_add_y_to_output=flag_deformable_SFT_add_y_to_output[i+1],
                                                           flag_IIR_filter_horizontal=flag_IIR_filter_horizontal[i+1],
                                                           flag_IIR_filter_horizontal_reverse=flag_IIR_filter_horizontal_reverse[i+1],
                                                           flag_IIR_filter_order_horizontal=flag_IIR_filter_order_horizontal[i+1],
                                                           flag_IIR_filter_vertical=flag_IIR_filter_vertical[i+1],
                                                           flag_IIR_filter_vertical_reverse=flag_IIR_filter_vertical_reverse[i+1],
                                                           flag_IIR_filter_order_vertical=flag_IIR_filter_order_vertical[i+1], flag_3D=flag_3D
                                                           ))



        ### If i want to override input variables and simply make the conv block be the Identity_Layer: ###
        if flag_identity_block:
            layers_list = nn.ModuleList()
            layers_list.append(Identity_Layer())

        ### Wrap the module list inside a Pile_Modules_On_Top_Of_Each_Other_Layer which chains the multiple_inputs from one layer to the other: ###
        # Conv_Layers = Pile_Modules_On_Top_Of_Each_Other(*layers_list)
        Conv_Layers = Pile_Modules_On_Top_Of_Each_Other_Layer(layers_list)  #TODO: uncomment this at the end!

        # ### TODO: temp - remove this at the end: ###
        # if type(layers_list) == nn.ModuleList or type(layers_list) == list:
        #     Conv_Layers = nn.Sequential(*layers_list)
        # else:
        #     Conv_Layers = layers_list


        ### ResNet: ###
        #TODO: add SFT over entire block
        if flag_identity_block:
            Conv_Layers = Input_Output_Identity_Block(Conv_Layers,number_of_output_channels[-1])
        else:
            if flag_resnet:
                ### Get number of input channels to current conv_block according to previous block's flags/outputs: ###
                if 'collapse' in flag_super_cell_block_type[-1]:
                    input_channels_factor = 1;
                else:
                    input_channels_factor = len(kernel_sizes[i])
                ###
                #TODO: input_output_sum_block can use projection blocks with 2D convolution -> it also needs to know whether to use 2D or 3D
                Conv_Layers = Input_Output_Sum_Block_With_StraightThrough_Projection_Block(Conv_Layers, number_of_output_channels[-1] * input_channels_factor + number_of_concatenated_channels_so_far, residual_scale=stack_residual_scale, flag_3D=flag_3D)
            if flag_concat:
                Conv_Layers = Input_Output_Concat_Block(Conv_Layers)
                number_of_concatenated_channels_so_far += number_of_input_channels


        ### Keep track of final number of channels: ###
        setattr(Conv_Layers, 'final_number_of_channels', number_of_concatenated_channels_so_far + number_of_output_channels[-1] * input_channels_factor)  #TODO: make sure logic is still consistent with flag_identity_block
        setattr(self, 'final_number_of_channels', number_of_concatenated_channels_so_far + number_of_output_channels[-1] * input_channels_factor)

        ### Initialize Object Variables: ###
        self.Conv_Layers = Conv_Layers;


    def forward(self, x, y_cell_conditional=None, y_deformable_conditional=None):
        # print('Sequential conv block: ' + str(y_cell_conditional.shape))
        return self.Conv_Layers(x, y_cell_conditional, y_deformable_conditional)  #TODO: uncomment this at the end
        # return self.Conv_Layers(x)






def Sequential_Conv_Block(number_of_input_channels,
                          number_of_output_channels,
                          kernel_sizes,
                          strides=1,
                          dilations=1,
                          groups=1,
                          padding_type='reflect',
                          normalization_function=None,
                          activation_function='relu',
                          mode='CNA',
                          initialization_method = 'xavier',
                          flag_resnet = False,
                          flag_concat = False):
    #TODO: Add Possibility of Deformable Convolution!
    #TODO: change to a list to make it a single list of layers instead of a sequential in sequential object
    if type(number_of_output_channels) != list and type(number_of_output_channels) != tuple:
        number_of_output_channels = [number_of_output_channels]
    number_of_layers = len(number_of_output_channels);
    kernel_sizes = to_list_of_certain_size(kernel_sizes, number_of_layers)
    strides = to_list_of_certain_size(strides, number_of_layers)
    dilations = to_list_of_certain_size(dilations, number_of_layers)
    groups = to_list_of_certain_size(groups, number_of_layers)
    activation_function = to_list_of_certain_size(activation_function, number_of_layers)


    ### Get Layers List: ###
    layers_list = []
    layers_list.append(Conv_Block(number_of_input_channels, number_of_output_channels[0], kernel_sizes[0], strides[0], dilations[0], groups[0], bias=True, padding_type='reflect', normalization_function=normalization_function, activation_function=activation_function[0]))
    for i in arange(len(kernel_sizes) - 1):
        layers_list.append(Conv_Block(number_of_output_channels[i], number_of_output_channels[i + 1], kernel_sizes[i + 1], strides[i + 1], dilations[i + 1], groups[i + 1], bias=True, padding_type='reflect',
                                      normalization_function=normalization_function, activation_function=activation_function[i+1]))
    Conv_Layers = Pile_Modules_On_Top_Of_Each_Other(*layers_list)

    ### ResNet: ###
    if flag_resnet:
        Conv_Layers = Input_Output_Sum_Block_With_StraightThrough_Projection_Block(Conv_Layers, number_of_output_channels[-1])
    if flag_concat:
        Conv_Layers = Input_Output_Concat_Block(Conv_Layers)

    return Conv_Layers


############################################################################################################################################################################################################################################################








############################################################################################################################################################################################################################################################
### SFT Interaction: ###

class SFT_self_interaction(nn.Module):
    # Initialize this with a module
    def __init__(self, submodule, flag_shift=True, flag_scale=True):
        super(SFT_self_interaction, self).__init__()
        self.sub = submodule
        self.flag_first = False;
        self.flag_shift = flag_shift
        self.flag_scale = flag_scale

    def forward(self, x):
        if self.flag_first == False:
            ### Send submodule to input's device: ###
            self.sub = self.sub.to(x.device)
            current_submodule_result = self.sub(x);
            ### Define Conv Layers: ###
            #TODO: generalize this to Conv_block / Sequential_Conv_Block_General(????)
            self.Conv_scale = nn.Conv2d(x.shape[1],current_submodule_result.shape[1],kernel_size=5, stride=1, padding=0, dilation=1, groups=1)
            self.Conv_shift = nn.Conv2d(x.shape[1],current_submodule_result.shape[1],kernel_size=5, stride=1, padding=0, dilation=1, groups=1)
            ### Change flag to only initialize layers at the first go: ###
            self.flag_first = True;
        else:
            current_submodule_result = self.sub(x)

        ### Scale & Shift: ###
        if self.flag_scale:
            output = self.Conv_scale(x)*current_submodule_result
        else:
            output = current_submodule_result;
        if self.flag_shift:
            output += self.Conv_shift(x)

        return output

    def __repr__(self):
        tmpstr = 'SFT_self_interaction .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr




class SFT_simple_conditional(nn.Module):
    # Initialize this with a module
    def __init__(self, submodule, flag_shift=True, flag_scale=True):
        super(SFT_simple_conditional, self).__init__()
        self.sub = submodule
        self.flag_first = False;
        self.flag_shift = flag_shift
        self.flag_scale = flag_scale

    def forward(self, x, y):
        if self.flag_first == False:
            ### Send submodule to input's device: ###
            self.sub = self.sub.to(x.device)
            current_submodule_result = self.sub(x);
            ### Define Conv Layers: ###
            # TODO: generalize this to Conv_block / Sequential_Conv_Block_General(????)
            self.Conv_scale = nn.Conv2d(y.shape[1], current_submodule_result.shape[1], kernel_size=5, stride=1, padding=0, dilation=1, groups=1)
            self.Conv_shift = nn.Conv2d(y.shape[1], current_submodule_result.shape[1], kernel_size=5, stride=1, padding=0, dilation=1, groups=1)
            ### Change flag to only initialize layers at the first go: ###
            self.flag_first = True;
        else:
            current_submodule_result = self.sub(x)

        ### Scale & Shift: ###
        if self.flag_scale:
            output = self.Conv_scale(y) * current_submodule_result
        else:
            output = current_submodule_result;
        if self.flag_shift:
            output += self.Conv_shift(y)

        return output

    def __repr__(self):
        tmpstr = 'SFT_simple_conditional .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr







class SFT_mixed_conditional(nn.Module):
    # Initialize this with a module
    def __init__(self, submodule, flag_shift=True, flag_scale=True):
        super(SFT_mixed_conditional, self).__init__()
        self.sub = submodule
        self.flag_first = False;
        self.flag_shift = flag_shift
        self.flag_scale = flag_scale

    def forward(self, x, y):
        if self.flag_first == False:
            ### Send submodule to input's device: ###
            self.sub = self.sub.to(x.device)
            current_submodule_result = self.sub(x);
            ### Define Conv Layers: ###
            # TODO: generalize this to Conv_block / Sequential_Conv_Block_General(????)
            self.Conv_scale = nn.Conv2d(x.shape[1] + y.shape[1], current_submodule_result.shape[1], kernel_size=5, stride=1, padding=0, dilation=1, groups=1)
            self.Conv_shift = nn.Conv2d(x.shape[1] + y.shape[1], current_submodule_result.shape[1], kernel_size=5, stride=1, padding=0, dilation=1, groups=1)
            ### Change flag to only initialize layers at the first go: ###
            self.flag_first = True;
        else:
            current_submodule_result = self.sub(x)

        ### Scale & Shift: ###
        concat_xy = torch.cat([x,y],dim=1)
        if self.flag_scale:
            output = self.Conv_scale(concat_xy) * current_submodule_result
        else:
            output = current_submodule_result;
        if self.flag_shift:
            output += self.Conv_shift(concat_xy)

        return output

    def __repr__(self):
        tmpstr = 'SFT_mixed_conditional .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr








class SFT_General_Wrapper(nn.Module):
    # Initialize this with a module
    def __init__(self,
                 submodule,
                 flag_use_outside_conditional = False,
                 flag_shift=True,
                 flag_scale=True,
                 flag_SFT_convs_mix = 'x',
                 flag_SFT_same_on_all_channels = False,
                 flag_SFT_add_y_to_output=False,
                 ):
        super(SFT_General_Wrapper, self).__init__()
        self.sub = submodule
        self.flag_first = False;
        self.flag_shift = flag_shift
        self.flag_scale = flag_scale
        self.flag_SFT_convs_mix = flag_SFT_convs_mix
        self.flag_SFT_same_on_all_channels = flag_SFT_same_on_all_channels
        self.flag_SFT_add_y_to_output = flag_SFT_add_y_to_output
        self.flag_use_outside_conditional = flag_use_outside_conditional
        print(self.flag_SFT_convs_mix)
        if self.flag_use_outside_conditional==False:
            self.flag_SFT_convs_mix = 'x'
            self.flag_SFT_add_y_to_output = False
        self.Conv_scale = None
        self.Conv_shift = None

    def forward(self, x, y_cell_conditional=None):
        ### Iteration = 1:
        # print('SFT wrapper ' + str(y_cell_conditional.shape))
        if self.flag_first == False:
            ### Send submodule to input's device: ###
            self.sub = self.sub.to(x.device)
            current_submodule_result = self.sub(x);

            ### Define Conv Layers: ###
            #(1). Number of Input Channels:
            if self.flag_SFT_convs_mix == 'x':
                number_of_input_channels = x.shape[1]
            elif self.flag_SFT_convs_mix == 'y':
                number_of_input_channels = y_cell_conditional.shape[1]
            elif self.flag_SFT_convs_mix == 'xy':
                number_of_input_channels = x.shape[1] + y_cell_conditional.shape[1]
            #(2). Number of Output Channels:
            if self.flag_SFT_same_on_all_channels:
                number_of_output_channels = 1;
            else:
                number_of_output_channels = current_submodule_result.shape[1]
            #(3). Initialize Scale&Shift Layers:
            self.Conv_scale = Pad_Conv2d(number_of_input_channels, number_of_output_channels, kernel_size=5, stride=1, dilation=1, groups=1)
            self.Conv_shift = Pad_Conv2d(number_of_input_channels, number_of_output_channels, kernel_size=5, stride=1, dilation=1, groups=1)
            ### Change flag to only initialize layers at the first go: ###
            print('finished SFT wrapper first initialization')
            print(self.Conv_scale.parameters().__next__().requires_grad)
            self.flag_first = True;


        else:
            ### Iteration > 1:
            current_submodule_result = self.sub(x)

        ### Scale & Shift: ###
        #(1). SFT inputs:
        if self.flag_SFT_convs_mix == 'x':
            input_tensor = x
        elif self.flag_SFT_convs_mix == 'y':
            input_tensor = y_cell_conditional
        elif self.flag_SFT_convs_mix == 'xy':
            input_tensor = torch.cat([x,y_cell_conditional],dim=1)
        #(2). SFT Scale:
        if self.flag_scale:
            output = self.Conv_scale(input_tensor) * current_submodule_result
        else:
            output = current_submodule_result;
        #(3). SFT Shift:
        if self.flag_shift:
            output += self.Conv_shift(input_tensor)
        #(4). Add y to output if so wanted (Must have y.shape[1]=x.shape[1] or y.shape[1]=1 unless i want to use projection block which i don't):
        if self.flag_SFT_add_y_to_output:
            output += y_cell_conditional;

        return output

    def __repr__(self):
        tmpstr = 'SFT_General_Wrapper .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


# ### Example Of Use: ###
# sub_module = nn.Conv2d(3,3,3,1,1);
# layer = SFT_General_Wrapper(sub_module,
#                  flag_use_outside_conditional = True,
#                  flag_shift=True,
#                  flag_scale=True,
#                  flag_SFT_convs_mix = 'xy',
#                  flag_SFT_same_on_all_channels = True,
#                  flag_SFT_add_y_to_output=True,)
# input_tensor = torch.Tensor(randn(1,3,100,100))
# conditional_tensor = torch.Tensor(randn(1,3,100,100))
# output_tensor = layer(input_tensor, conditional_tensor)





class Conv2d_MixSFT(nn.Module):
    # Initialize this with a module
    def __init__(self,
                 flag_use_outside_conditional = True,
                 flag_base_convs_mix='x', #'x'-> base=conv(x),  'y'-> base=conv(y),  'xy'-> base=conv(x+y)
                 flag_SFT_convs_mix='x', #'x'-> output=conv(x)*base+conv(x),  'y'->output=conv(y)*base+conv(y),  'xy'->output=conv(x+y)*base+conv(x+y)
                 flag_shift=True,
                 flag_scale=True,
                 number_of_x_channels=3,
                 number_of_y_channels=3,
                 number_of_output_channels=3,
                 kernel_size=3):
        super(Conv2d_MixSFT, self).__init__()
        ### Parameters: ###
        self.flag_use_outside_conditional = flag_use_outside_conditional
        self.flag_first = False;
        self.flag_shift = flag_shift
        self.flag_scale = flag_scale
        self.flag_base_convs_mix = flag_base_convs_mix
        self.flag_SFT_convs_mix = flag_SFT_convs_mix;
        self.number_of_x_channels = number_of_x_channels
        self.number_of_y_channels = number_of_y_channels
        self.number_of_output_channels = number_of_output_channels
        self.kernel_size = kernel_size
        if flag_use_outside_conditional==False:
            self.number_of_y_channels = 0
            self.flag_base_convs_mix = 'x'
            self.flag_SFT_convs_mix = 'x';

        ### Layers: ###
        #(1). Base Layer:
        if self.flag_base_convs_mix=='xy':
            self.base_layer = Pad_Conv2d(self.number_of_x_channels + self.number_of_y_channels, number_of_output_channels, kernel_size=kernel_size, stride=1)
        elif self.flag_base_convs_mix=='x':
            self.base_layer = Pad_Conv2d(self.number_of_x_channels, self.number_of_output_channels, kernel_size=kernel_size, stride=1)
        elif self.flag_base_convs_mix=='y':
            self.base_layer = Pad_Conv2d(self.number_of_y_channels, self.number_of_output_channels, kernel_size=kernel_size, stride=1)

        #(2). SFT:
        if self.flag_SFT_convs_mix=='x':
            self.conv_layer_scale = Pad_Conv2d(self.number_of_x_channels, number_of_output_channels, kernel_size=kernel_size, stride=1)
            self.conv_layer_shift = Pad_Conv2d(self.number_of_x_channels, number_of_output_channels, kernel_size=kernel_size, stride=1)
        elif self.flag_SFT_convs_mix=='y':
            self.conv_layer_scale = Pad_Conv2d(self.number_of_y_channels, number_of_output_channels, kernel_size=kernel_size, stride=1)
            self.conv_layer_shift = Pad_Conv2d(self.number_of_y_channels, number_of_output_channels, kernel_size=kernel_size, stride=1)
        elif self.flag_SFT_convs_mix=='xy':
            self.conv_layer_scale = Pad_Conv2d(self.number_of_x_channels + self.number_of_y_channels, number_of_output_channels, kernel_size=kernel_size, stride=1)
            self.conv_layer_shift = Pad_Conv2d(self.number_of_x_channels + self.number_of_y_channels, number_of_output_channels, kernel_size=kernel_size, stride=1)




    def forward(self, x, y_conditional=None):
        ### Base Conv: ###
        if self.flag_base_convs_mix=='xy':
            output = self.base_layer(torch.cat([x,y_conditional],dim=1))
        elif self.flag_base_convs_mix=='y':
            output = self.base_layer(y_conditional);
        elif self.flag_base_convs_mix=='x':
            output = self.base_layer(x)

        ### SFT Input: ###
        if self.flag_SFT_convs_mix=='xy':
            sft_input = torch.cat([x,y_conditional], dim=1)
        elif self.flag_SFT_convs_mix=='x':
            sft_input = x;
        elif self.flag_SFT_convs_mix=='y':
            sft_input = y_conditional;

        ### SFT Output: ###
        if self.flag_scale:
            output = self.conv_layer_scale(sft_input)*output;
        if self.flag_shift:
            output = output + self.conv_layer_shift(sft_input)

        return output
############################################################################################################################################################################################################################################################













############################################################################################################################################################################################################################################################
### Deformable Convolutions: ###


class ConvOffset2d_v1(nn.Module):
    #TODO: supposedly i can always make the module that outputs the offsets more complicated and not just a simple conv layer....

    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """
    def __init__(self, number_of_input_channels, kernel_size=3, init_normal_stddev=0.01, flag_same_on_all_channels=False, flag_automatic=True):
        super(ConvOffset2d_v1,self).__init__()
        """Init

        Parameters
        ----------
        number_of_input_channels : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.number_of_input_channels = number_of_input_channels
        self._grid_param = None
        self.kernel_size = kernel_size
        self.padding_size = self.kernel_size//2;
        self.padding_layer = nn.ReflectionPad2d(self.padding_size)
        self.flag_same_on_all_channels = flag_same_on_all_channels
        if self.flag_same_on_all_channels:
            self.conv_offset_layer = nn.Conv2d(self.number_of_input_channels, 2, kernel_size=kernel_size, padding=0, bias=False)
        else:
            self.conv_offset_layer = nn.Conv2d(self.number_of_input_channels, self.number_of_input_channels*2, kernel_size=kernel_size, padding=0, bias=False)  #(*). makes sense to have no bias here i think
        self.conv_offset_layer.weight.data.copy_(self._init_weights(self.conv_offset_layer.weight, init_normal_stddev))
        self.conv_offset_layer = nn.Sequential(self.padding_layer, self.conv_offset_layer)

        if flag_automatic:
            self.conv_offset_layer = None

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()

        ### Get conv_offset_layer if it doesn't exist: ###
        if self.conv_offset_layer is None:
            self.number_of_input_channels = x.shape[1]
            ### Initialize Conv Layer: ###
            if self.flag_same_on_all_channels:
                self.conv_offset_layer = nn.Conv2d(x.shape[1] + y.shape[1], 2, self.kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
                self.halfway_index = 1;
                self.number_of_offset_output_channels = 2;
            else:
                self.conv_offset_layer = nn.Conv2d(x.shape[1] + y.shape[1], x.shape[1], self.kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
                self.halfway_index = self.number_of_input_channels;
                self.number_of_offset_output_channels = self.number_of_input_channels * 2;
            ### Initialize Weights: ###
            self.conv_offset_layer.weight.data.copy_(self._init_weights(self.conv_offset_layer.weight, init_normal_stddev))
            self.conv_offset_layer = nn.Sequential(self.padding_layer, self.conv_offset_layer)


        ### Get Offset Map: ###
        offsets = self.conv_offset_layer(x)
        if self.flag_same_on_all_channels:
            #in this case offsets=[B,1,H,W]->[B,C,H,W] with all channels being the same
            offsets = torch.cat([offsets]*x.shape[1], dim=1)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self,x))

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)

        ### TODO: It seems that when using conv_offset that i get a larger spatial size afterwards....so for now i'll try to simply crop the result...but i should really try and understand exactly what's going on and maybe even create this layer myself: ###
        # x_offset = x_offset[]

        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().reshape(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().reshape(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().reshape(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x








class ConvOffset2d_v3(nn.Module):
    ### Bicubic Interpolation: ###

    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """
    def __init__(self, number_of_input_channels, kernel_size=3, init_normal_stddev=0.01, flag_same_on_all_channels=False, flag_automatic=True):
        super(ConvOffset2d_v3,self).__init__()
        """Init

        Parameters
        ----------
        number_of_input_channels : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        ### Initialize Parameters: ###
        self.number_of_input_channels = number_of_input_channels
        self._grid_param = None
        self.kernel_size = kernel_size
        self.padding_size = self.kernel_size//2;
        self.padding_layer = nn.ReflectionPad2d(self.padding_size)
        self.flag_same_on_all_channels = flag_same_on_all_channels

        ### Initialize Conv layer which outputs offsets: ###
        #TODO: should i make the deducing conv block more complicated?...what about activation functions?!?... maybe as long as it's not the very first layer than there's enough logic that can come about before it to make it take smart decisions as to delta's
        if self.flag_same_on_all_channels:
            self.conv_offset_layer = nn.Conv2d(self.number_of_input_channels, 2, kernel_size=kernel_size, padding=0, bias=False)
            self.halfway_index = 1;
            self.number_of_offset_output_channels = 2;
        else:
            self.conv_offset_layer = nn.Conv2d(self.number_of_input_channels, self.number_of_input_channels*2, kernel_size=kernel_size, padding=0, bias=False)  #(*). makes sense to have no bias here i think
            self.halfway_index = self.number_of_input_channels;
            self.number_of_offset_output_channels = self.number_of_input_channels*2
        self.conv_offset_layer.weight.data.copy_(self._init_weights(self.conv_offset_layer.weight, init_normal_stddev))
        self.conv_offset_layer = nn.Sequential(self.padding_layer, self.conv_offset_layer)

        ### Build Conv_Offset According To Input: ###
        if flag_automatic:
            self.conv_offset_layer = None;

        ### Initialize Bicubic_Interpolator Layer: ###
        self.bicubic_interpolator_layer = Bicubic_Interpolator()

    def forward(self, x):
        x_shape = x.size()

        ### Get conv_offset_layer if it doesn't exist: ###
        if self.conv_offset_layer is None:
            self.number_of_input_channels = x.shape[1]
            ### Initialize Conv Layer: ###
            if self.flag_same_on_all_channels:
                self.conv_offset_layer = nn.Conv2d(x.shape[1] + y.shape[1], 2, self.kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
                self.halfway_index = 1;
                self.number_of_offset_output_channels = 2;
            else:
                self.conv_offset_layer = nn.Conv2d(x.shape[1] + y.shape[1], x.shape[1], self.kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
                self.halfway_index = self.number_of_input_channels;
                self.number_of_offset_output_channels = self.number_of_input_channels * 2;
            ### Initialize Weights: ###
            self.conv_offset_layer.weight.data.copy_(self._init_weights(self.conv_offset_layer.weight, init_normal_stddev))
            self.conv_offset_layer = nn.Sequential(self.padding_layer, self.conv_offset_layer)

        ### Get Offset Map: ###
        offsets = self.conv_offset_layer(x)
        ### Warp: ###
        x_warped = self.bicubic_interpolator_layer(x, offsets[:,0:self.halfway_index,:,:], offsets[:,self.halfway_index:self.number_of_offset_output_channels,:,:])








################################################################################################################################################################################################################################################################################################
class ConvOffset2d_v3_Groups_Conditional_SFT(nn.Module):
    ### Most General ConvOffset2d with possible deformable-groups and SFT interaction with possible outside contingency: ###

    def __init__(self,
                 number_of_input_channels=3,
                 number_of_deformable_groups=2,
                 flag_use_outside_conditional = False, #True-> deformable contingent also on outside "y_tensor", False-> only rely on input x (CLASSICAL)
                 flag_base_convs_mix = 'x', #'x','y','xy'
                 flag_SFT_convs_mix = 'x', #'x','y','xy'
                 flag_shift = True,
                 flag_scale = True,
                 flag_add_y_to_output = False, #Bias output optical flow to input y_tensor
                 kernel_size=5,
                 init_normal_stddev=0.01,
                 flag_same_on_all_channels=False,
                 flag_automatic=True):
        super(ConvOffset2d_v3_Groups_Conditional_SFT,self).__init__()
        #TODO: add possibility of modulation (deformable_convolution_V2)
        #TODO: add possibility of unequal number of groups with deformable convolution... for instance: 2 groups/outputs with [3,N-3] input channels to be warped
        #TODO: add possibility of multiple_kernels with input parameters like: number_of_scales, flag_same_kernels_on_all_dilations
        ### Groups=1 for THE SAME deformation on all channels: ###
        ### Groups=2 for Two Groups of deformations (maybe appropriate for a mix of features from originally two seperate features/images: ####
        ### Groups=-1 for a Different deformation for each channel: ###


        ### Initialize Parameters: ###
        self.number_of_input_channels = number_of_input_channels
        self._grid_param = None
        self.kernel_size = kernel_size
        self.padding_size = self.kernel_size//2;
        self.padding_layer = nn.ReflectionPad2d(self.padding_size)
        self.flag_same_on_all_channels = flag_same_on_all_channels
        self.flag_add_y_to_output = flag_add_y_to_output
        ### SFT Parameters: ###
        self.flag_use_outside_conditional = flag_use_outside_conditional;
        self.flag_base_convs_mix = flag_base_convs_mix;
        self.flag_SFT_convs_mix = flag_SFT_convs_mix;
        self.flag_shift = flag_shift;
        self.flag_scale = flag_scale;
        ### Initialize Conv layer which outputs offsets: ###
        self.number_of_deformable_groups = number_of_deformable_groups;
        if self.flag_same_on_all_channels:
            self.number_of_deformable_groups = 1;
        self.halfway_index = 1;
        self.number_of_offset_output_channels = 2;

        ### Build Conv_Offset According To Input: ###
        self.conv_offset_layer = None;

        ### Initialize Bicubic_Interpolator Layer: ###
        self.bicubic_interpolator_layer = Bicubic_Interpolator('bilinear')


    def forward(self, x, y_deformable_conditional=None):
        x_shape = x.size()

        ##################################################
        ### Get conv_offset_layer if it doesn't exist: ###
        if self.conv_offset_layer is None:
            self.number_of_input_channels = x.shape[1]
            if self.number_of_deformable_groups == -1:
                self.number_of_deformable_groups = x.shape[1];
            ### Initialize number of channels according to y: ###
            if y_deformable_conditional is None:
                self.number_of_y_channels = 0;
                self.flag_use_outside_conditional = False; #there's nothing to condition on!
                self.flag_add_y_to_output = False; #there's nothing to add!
            else:
                self.number_of_y_channels = y_deformable_conditional.shape[1]
            ### Initialize Conv Layer: ###
            self.conv_offset_layer = Conv2d_MixSFT(flag_use_outside_conditional=self.flag_use_outside_conditional,
                                                   flag_base_convs_mix=self.flag_base_convs_mix, #True-> base=conv(x),  False-> base=conv(x+y)
                                                   flag_SFT_convs_mix=self.flag_SFT_convs_mix, #True-> output=conv(x+y)*base+conv(x+y),  False->output=conv(y)*base+conv(y)
                                                   flag_shift=self.flag_shift,
                                                   flag_scale=self.flag_scale,
                                                   number_of_x_channels=x.shape[1],
                                                   number_of_y_channels=self.number_of_y_channels,
                                                   number_of_output_channels=2*self.number_of_deformable_groups,
                                                   kernel_size=self.kernel_size)

            self.halfway_index = x.shape[1]//2; #Two   Deformable Groups
            self.number_of_offset_output_channels = x.shape[1]
            self.number_of_channels_per_group = x.shape[1]//self.number_of_deformable_groups
        ###################################################


        ### Get Offset Map: ###
        #TODO: there is the question of how to receive y_deformable_conditional....we obviously have two maps -> delta_x,delta_y....
        #TODO: i need to make it flexible such that if i get [y_deformable_conditional]=[B,2,H,W] that i add the first channel to all delta_x and second channel to all delta_y.
        #TODO: also if i receive [y_deformable_conditional]=[B,2*C,H,W] i need to know by what strategy do i distribute...i think what makes most sense is in sequential pairs-> [B,0:1,H,W] to first input channels, [B,2:3,H,W] to second input channels etc'...
        ###(***). HOWEVER - i need to remember that offsets itself might be of size x.shape[1]*2 or simple with 2 channels (uniform for all channels)
        offsets = self.conv_offset_layer(x, y_deformable_conditional)
        if self.flag_add_y_to_output:
            if y_deformable_conditional.shape[1] == 1: #input conditional has 1 channel -> simply add it to everything
                offsets = offsets + y_deformable_conditional
            elif y_deformable_conditional.shape[1] == 2: #input conditional has 2 channels -> add first channel to all delta_x and second channel to all delta_y (which by convention come in sequential pairs)
                offsets[:,0:-1:2,:,:] += y_deformable_conditional[:,0,:,:]
                offsets[:,1:-1:2,:,:] += y_deformable_conditional[:,1,:,:]
            elif y_deformable_conditional.shape[1] == self.number_of_deformable_groups*2: #received as many channels as the number of deformable groups (i inputed y_conditional with the knowledge of the number of deformable groups)
                for i in arange(self.number_of_deformable_groups):
                    offsets[:, i*2, :, :] += y_deformable_conditional[:, i*2, :, :]
                    offsets[:, 1*2+1, :, :] += y_deformable_conditional[:, 1*2+1, :, :]
            elif y_deformable_conditional.shape[1] == x.shape[1]*2: #received as many channels as appropriate to shift each ORIGINAL INPUT CHANNEL Seperately
                    1; #in this case i add it later on in the loop. TODO: implement!!!
        # offsets_x = offsets[:,0:offsets.shape[1]//2,:,:]
        # offsets_y = offsets[:,offsets.shape[1]//2:,:,:]


        ### Warp Each Input Deformable Group: ###
        x_offset = torch.zeros_like(x)
        for i in arange(self.number_of_deformable_groups):
            x_start_index = i*self.number_of_channels_per_group
            x_stop_index = (i+1)*self.number_of_channels_per_group
            offsets_start_index = i*2
            offsets_stop_index = i*2 + 1
            current_offsets_x = offsets[:,offsets_start_index:offsets_stop_index,:,:]
            current_offsets_y = offsets[:,offsets_start_index:offsets_stop_index,:,:]
            x_offset[:, x_start_index:(i+1)*x_stop_index, :, :] = self.bicubic_interpolator_layer(x[:, x_start_index:(i+1)*x_stop_index, :, :], current_offsets_x, current_offsets_y)

        return x_offset;
########################################################################################################################################################################################################################################################################################################################################

















#################################################################################################################################################################################################################################
class RDB(nn.Module):
    #TODO: residual block which can handle different number of channels / spatial size on the residual addition (projection block)
    def __init__(self, number_of_input_channels, kernel_size=3, number_of_output_channels_for_each_conv_block=32, number_of_conv_blocks = 3,
                 stride=1, dilations=1, groups=1, bias=True, padding_type='zero', \
                 normalization_function=None, activation_function='leakyrelu', mode='CNA', final_residual_branch_scale_factor = 0.2, initialization_method='xavier'):
        super(RDB, self).__init__()
        # number_of_output_channels: growth channel, i.e. intermediate channels
        self.number_of_conv_blocks = number_of_conv_blocks;
        self.layer_names = []
        self.layers_list = []
        self.final_residual_branch_scale_factor = final_residual_branch_scale_factor;
        for layer_index in arange(number_of_conv_blocks):
            layer_name = 'conv'+str(layer_index);
            self.layer_names.append(layer_name)
            setattr(self,layer_name,
                    Input_Output_Concat_Block(
                        Conv_Block(number_of_input_channels + layer_index * number_of_output_channels_for_each_conv_block,
                                   number_of_output_channels_for_each_conv_block, kernel_size, stride, dilation=dilations, groups=groups, bias=bias,
                                   padding_type=padding_type, \
                                   normalization_function=normalization_function, activation_function=activation_function, mode=mode, initialization_method=initialization_method)
                    ))
            self.layers_list.append(getattr(self,layer_name))


        ### Final Layer: ###
        #(*). Notice that for the last layer we put NO ACTIVATION
        if mode == 'CNA':
            last_act = None
        else:
            last_act = activation_function
        self.conv_final = Conv_Block(number_of_input_channels + number_of_conv_blocks * number_of_output_channels_for_each_conv_block,
                                     number_of_input_channels,
                                     kernel_size, stride, bias=bias, padding_type=padding_type, \
                                     normalization_function=normalization_function, activation_function=last_act, mode=mode)

        self.layers_list.append(self.conv_final);

        ### Create main block consisting of Dense blocks: ###
        self.dense_layers_block = Pile_Modules_On_Top_Of_Each_Other(*self.layers_list);

        ### Final Residual Wrapper: ###
        self.conv_combined = Input_Output_Sum_Block(self.dense_layers_block,residual_scale=final_residual_branch_scale_factor);


    def forward(self, x):
        return self.conv_combined(x)




class RDB_projection(nn.Module):
    #TODO: residual block which can handle different number of channels / spatial size on the residual addition (projection block)
    def __init__(self, number_of_input_channels, kernel_size=3,
                 number_of_output_channels_for_each_conv_block=32,
                 number_of_conv_blocks = 3,
                 number_of_output_channel_final = 100,
                 stride=1, dilations=1, groups=1, bias=True, padding_type='zero', \
                 normalization_function=None, activation_function='leakyrelu', mode='CNA', final_residual_branch_scale_factor = 0.2, initialization_method='xavier'):
        super(RDB, self).__init__()
        # number_of_output_channels: growth channel, i.e. intermediate channels
        self.number_of_conv_blocks = number_of_conv_blocks;
        self.layer_names = []
        self.layers_list = []
        self.final_residual_branch_scale_factor = final_residual_branch_scale_factor;
        for layer_index in arange(number_of_conv_blocks):
            layer_name = 'conv'+str(layer_index);
            self.layer_names.append(layer_name)
            setattr(self,layer_name,
                    Input_Output_Concat_Block(
                        Conv_Block(number_of_input_channels + layer_index * number_of_output_channels_for_each_conv_block,
                                   number_of_output_channels_for_each_conv_block, kernel_size, stride, dilation=dilations, groups=groups, bias=bias,
                                   padding_type=padding_type, \
                                   normalization_function=normalization_function, activation_function=activation_function, mode=mode, initialization_method=initialization_method)
                    ))
            self.layers_list.append(getattr(self,layer_name))


        ### Final Layer: ###
        #(*). Notice that for the last layer we put NO ACTIVATION
        if mode == 'CNA':
            last_act = None
        else:
            last_act = activation_function
        self.conv_final = Conv_Block(number_of_input_channels + number_of_conv_blocks * number_of_output_channels_for_each_conv_block,
                                     number_of_output_channel_final,
                                     kernel_size, stride, bias=bias, padding_type=padding_type, \
                                     normalization_function=normalization_function, activation_function=last_act, mode=mode)

        self.layers_list.append(self.conv_final);

        ### Create main block consisting of Dense blocks: ###
        self.dense_layers_block = Pile_Modules_On_Top_Of_Each_Other(*self.layers_list);

        ### Final Residual Wrapper: ###
        self.conv_combined = Input_Output_Sum_Block_With_StraightThrough_Projection_Block(self.dense_layers_block,residual_scale=final_residual_branch_scale_factor);


    def forward(self, x):
        return self.conv_combined(x)



#(2). RRDB block!- an imrovement to the RDB block!:
class RRDB(nn.Module):
    def __init__(self,
                 number_of_input_channels,
                 number_of_intermediate_channels_for_each_conv_block=32,
                 number_of_conv_blocks_per_RDB_cell = 2,
                 number_of_RDB_cells = 2,
                 kernel_size=3,
                 stride=1, dilation=1, groups=1, bias=True, padding_type='zero', \
                 normalization_function=None,
                 activation_function='leakyrelu',
                 flags_deformable_convolution=False,
                 initialization_method='xavier',
                 individual_cell_residual_scaling_factor=0.2,
                 residual_scaling_factor = 0.2,
                 mode='CNA'):
        super(RRDB, self).__init__()


        self.number_of_RDB_cells = number_of_RDB_cells;

        self.number_of_intermediate_channels_for_each_conv_block,\
        self.number_of_conv_blocks_per_RDB_cell, \
        self.kernel_sizes, self.strides, self.dilations, self.groups,\
        self.flags_deformable_convolution, \
        self.initialization_method, \
        self.normalization_function,\
        self.activation_function, self.individual_cell_residual_scaling_factor =\
            make_list_of_certain_size(number_of_RDB_cells,
                                      number_of_intermediate_channels_for_each_conv_block, number_of_conv_blocks_per_RDB_cell,
                                      kernel_size, stride, dilation, groups,
                                      flags_deformable_convolution,
                                      initialization_method,
                                      normalization_function,
                                      activation_function, individual_cell_residual_scaling_factor)

        layers_list = []
        for i in arange(number_of_RDB_cells):
            layers_list.append(RDB(
                                   number_of_input_channels,
                                    self.kernel_sizes[i],
                                    self.number_of_intermediate_channels_for_each_conv_block[i],
                                    self.number_of_conv_blocks_per_RDB_cell[i],
                                    self.strides[i], self.dilations[i], self.groups[i], bias=True, padding_type='reflect',
                                    normalization_function=self.normalization_function[i],
                                    activation_function=self.activation_function[i],
                                    final_residual_branch_scale_factor=self.individual_cell_residual_scaling_factor[i],
                                    initialization_method=self.initialization_method[i]))
        self.Conv_Layers = Pile_Modules_On_Top_Of_Each_Other(*layers_list)
        self.residual_scaling_factor = residual_scaling_factor;



    def forward(self, x):
        output_tensor = self.Conv_Layers(x)
        # Again!- Notice that this is a residual block. final output size is the same as the input size. number_of_output_channels is only used internally.
        return self.residual_scaling_factor*output_tensor + x  # TODO: isn't the 0.2 a hyper parameter!??
####################################################################################################################################################################################################################################










class Input_Output_Concat_Block(nn.Module):
    # Initialize this with a module
    def __init__(self, submodule):
        super(Input_Output_Concat_Block, self).__init__()
        self.sub = submodule

        self.flag_first = False;
    # Feed the module an input and output the module output concatenated with the input itself.
    def forward(self, x, y_cell_conditional=None, y_deformable_conditional=None):
        if self.flag_first == False:
            self.sub = self.sub.to(x.device)
            self.flag_first = True;

        ### To make this wrapper compatible with the most general configuration which includes modules receiving y_cell_conditional and y_deformable_conditional, check what parameters sub_module accepts: ###
        if 'y_cell_conditional' in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' in signature(self.sub.forward).parameters.keys():
            output = torch.cat([x, self.sub(x, y_cell_conditional, y_deformable_conditional)], dim=1)
        elif 'y_cell_conditional' in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' not in signature(self.sub.forward).parameters.keys():
            output = torch.cat([x, self.sub(x, y_cell_conditional)], dim=1)
        elif 'y_cell_conditional' not in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' in signature(self.sub.forward).parameters.keys():
            output = torch.cat([x, self.sub(x, y_deformable_conditional)], dim=1)
        else:
            output = torch.cat([x, self.sub(x)], dim=1)
        ########################################################################################################################################################################################################

        return output

    def __repr__(self):
        tmpstr = 'Input_Output_Concat .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class Input_Output_Sum_Block(nn.Module):
    # Initialize this with a module
    def __init__(self, submodule, residual_scale=1):
        super(Input_Output_Sum_Block, self).__init__()
        self.sub = submodule
        self.residual_scale = residual_scale;
        self.flag_first = False;
    # Elementwise sum the output of a submodule to its input
    def forward(self, x):
        if self.flag_first == False:
            self.sub = self.sub.to(x.device)
            self.flag_first = True;
        output = x + self.sub(x)*self.residual_scale
        return output

    def __repr__(self):
        tmpstr = 'Input_Output_Sum + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr



class Input_Output_Sum_Block_With_Projection_Block(nn.Module):
    #TODO: implement!@@!#%$%*(^
    # Initialize this with a module
    def __init__(self, submodule, residual_scale=1):
        super(Input_Output_Sum_Block, self).__init__()
        self.sub = submodule
        self.residual_scale = residual_scale;
        self.flag_first = False;
    # Elementwise sum the output of a submodule to its input
    def forward(self, x):
        if self.flag_first == False:
            self.sub = self.sub.to(x.device)
            self.flag_first = True;
        output = x + self.sub(x)*self.residual_scale
        return output

    def __repr__(self):
        tmpstr = 'Input_Output_Sum_With_Projection + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr



class Input_Output_Identity_Block(nn.Module):
    # Initialize this with a module
    def __init__(self, submodule, number_of_output_channel):
        super(Input_Output_Identity_Block, self).__init__()
        self.sub = submodule
        self.number_of_output_channels = number_of_output_channel

    # Elementwise sum the output of a submodule to its input
    def forward(self, x, y_cell_conditional=None, y_deformable_conditional=None):
        return x



class Input_Output_Sum_Block_With_StraightThrough_Projection_Block(nn.Module):
    # Initialize this with a module
    def __init__(self, submodule, number_of_output_channels, residual_scale=1, flag_3D=False):
        super(Input_Output_Sum_Block_With_StraightThrough_Projection_Block, self).__init__()
        self.sub = submodule
        self.residual_scale = residual_scale;
        self.number_of_output_channels = number_of_output_channels;
        self.projection_block = None;
        self.flag_first = False;
        self.flag_3D = flag_3D
    # Elementwise sum the output of a submodule to its input
    def forward(self, x, y_cell_conditional=None, y_deformable_conditional=None):
        if self.projection_block == None:
            self.number_of_input_channels = x.shape[1]
            if self.number_of_output_channels != x.shape[1]:
                if self.number_of_output_channels > x.shape[1]:
                    self.projection_block = Conv_Block(self.number_of_input_channels,self.number_of_output_channels-x.shape[1], kernel_size=3,stride=1,dilation=1,groups=1,bias=True,padding_type='reflect',
                                                       normalization_function='none', activation_function='none', mode='CNA', initialization_method='xavier', flag_3D=self.flag_3D)
                    self.projection_block = self.projection_block.to(x.device)
                    self.projection_block = Input_Output_Concat_Block(self.projection_block)  #Concat original input with conv_block output to produce needed number of output channels
                    self.projection_block = self.projection_block.to(x.device)
                elif self.number_of_output_channels < x.shape[1]:
                    self.projection_block = Conv_Block(self.number_of_input_channels,  self.number_of_output_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',
                                                       normalization_function='none', activation_function='none', mode='CNA', initialization_method='dirac', flag_3D=self.flag_3D) #use conv block with dirac initialization to bias towards pass through
                    self.projection_block = self.projection_block.to(x.device)
            elif self.number_of_output_channels == x.shape[1]:
                self.projection_block = Identity_Layer();
            self.projection_block = self.projection_block.to(x.device)

        if self.flag_first == False:
            self.sub = self.sub.to(x.device)
            self.flag_first = True;


        ### To make this wrapper compatible with the most general configuration which includes modules receiving y_cell_conditional and y_deformable_conditional, check what parameters sub_module accepts: ###
        # print('input output sum block ' + str(y_cell_conditional.shape))
        if 'y_cell_conditional' in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' in signature(self.sub.forward).parameters.keys():
            output = self.projection_block(x) + self.sub(x,y_cell_conditional,y_deformable_conditional) * self.residual_scale
        elif 'y_cell_conditional' in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' not in signature(self.sub.forward).parameters.keys():
            output = self.projection_block(x) + self.sub(x,y_cell_conditional) * self.residual_scale
        elif 'y_cell_conditional' not in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' in signature(self.sub.forward).parameters.keys():
            output = self.projection_block(x) + self.sub(x,y_deformable_conditional) * self.residual_scale
        else:
            output = self.projection_block(x) + self.sub(x)*self.residual_scale
        ########################################################################################################################################################################################################

        return output

    def __repr__(self):
        tmpstr = 'Input_Output_Sum_With_Dirac_Projection + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr





# Concatenate a list of input modules on top of each other to make on long module!:
def Pile_Modules_On_Top_Of_Each_Other(*args):
    # Flatten Sequential. It unwraps nn.Sequential.

    # If i only got one module (assuming it's a module) just return it:
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.

    # Take all the modules i got in the args input and simply concatenate them / pile them on top of each other and return a single "long" module.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)




class Pile_Modules_On_Top_Of_Each_Other_Layer(nn.Module):
    def __init__(self, module_list):
        super(Pile_Modules_On_Top_Of_Each_Other_Layer, self).__init__()
        ### module_list shouuld be of type nn.ModuleList() as far as i can understand: ###
        self.module_list = module_list

        # ### Temp - take input module_list and make a sequential block out of it...this can't accomodate conditional inputs etc': ###
        # if type(self.module_list) == nn.ModuleList or type(self.module_list) == list:
        #     self.sequential_model = nn.Sequential(*self.module_list)
        # else:
        #     self.sequential_model = self.module_list


    def forward(self, x, y_cell_conditional=None, y_deformable_conditional=None):
        # output = x.clone();  ###TODO: wait....why the .clone()?!?!?!!?
        output = x  ###TODO: wait....why the .clone()?!?!?!!?

        ### Check whether i received list (which means a different conditional input for each module) or a single tensor (the same conditional input for all modules): ###
        if type(y_cell_conditional) == list:
            flag_y_cell_conditional_list = True;
        else:
            flag_y_cell_conditional_list = False
        if type(y_deformable_conditional) == list:
            flag_y_deformable_conditional_list = True;
        else:
            flag_y_deformable_conditional_list = False


        # ### Pass Through Networks: ###
        # #TODO: maybe using signature(module).parameters().keys() is slow and i should only use it once and save them as internal variables......why not?!
        # for i, module in enumerate(self.module_list):
        #     if i == 0:
        #         output = module(x)
        #     else:
        #         output = module(output)


        # ### TODO: temporary override to check if this works: ###
        # output = self.sequential_model(x)


        ### TODO: uncomment this at the end: ###
        for i, module in enumerate(self.module_list):
            ### To make this wrapper compatible with the most general configuration which includes modules receiving y_cell_conditional and y_deformable_conditional, check what parameters sub_module accepts: ###
            if 'y_cell_conditional' in signature(module.forward).parameters.keys() and 'y_deformable_conditional' in signature(module.forward).parameters.keys():
                if flag_y_cell_conditional_list and flag_y_deformable_conditional_list:
                    output = module(output, y_cell_conditional[i], y_deformable_conditional[i])
                elif flag_y_cell_conditional_list:
                    output = module(output, y_cell_conditional[i], y_deformable_conditional)
                elif flag_y_deformable_conditional_list:
                    output = module(output, y_cell_conditional, y_deformable_conditional[i])
                else:
                    output = module(output, y_cell_conditional, y_deformable_conditional)  #TODO: uncomment this at some point. TODO: should i use clone?
                    # output = module(output.clone(), y_cell_conditional, y_deformable_conditional)  #TODO: uncomment this at some point. TODO: should i use clone?
                    # output = module(x, y_cell_conditional, y_deformable_conditional)
            elif 'y_cell_conditional' in signature(module.forward).parameters.keys() and 'y_deformable_conditional' not in signature(module.forward).parameters.keys():
                if flag_y_cell_conditional_list:
                    output = module(output, y_cell_conditional[i])
                else:
                    output = module(output, y_cell_conditional)
            elif 'y_cell_conditional' not in signature(module.forward).parameters.keys() and 'y_deformable_conditional' in signature(module.forward).parameters.keys():
                output = module(output, y_deformable_conditional)
                if flag_y_deformable_conditional_list:
                    output = module(output, y_deformable_conditional[i])
                else:
                    output = module(output, y_deformable_conditional)
            else:
                output = module(output)

        return output



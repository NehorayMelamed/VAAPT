import RapidBase.Basic_Import_Libs
from RapidBase.Basic_Import_Libs import *


# #(5). Layers:
# # import RapidBase.MISC_REPOS.Layers.Activations
# import RapidBase.MISC_REPOS.Layers.Basic_Layers
# import RapidBase.MISC_REPOS.Layers.Conv_Blocks
# import RapidBase.MISC_REPOS.Layers.DownSample_UpSample
# import RapidBase.MISC_REPOS.Layers.Memory_and_RNN
# import RapidBase.MISC_REPOS.Layers.Refinement_Modules
# import RapidBase.MISC_REPOS.Layers.Special_Layers
# import RapidBase.MISC_REPOS.Layers.Unet_UpDown_Blocks
# import RapidBase.MISC_REPOS.Layers.Warp_Layers
import RapidBase.MISC_REPOS.Layers.Wrappers
# # from RapidBase.MISC_REPOS.Layers.Activations import *
# from RapidBase.MISC_REPOS.Layers.Basic_Layers import *
# from RapidBase.MISC_REPOS.Layers.Conv_Blocks import *
# from RapidBase.MISC_REPOS.Layers.DownSample_UpSample import *
# from RapidBase.MISC_REPOS.Layers.Memory_and_RNN import *
# from RapidBase.MISC_REPOS.Layers.Refinement_Modules import *
# from RapidBase.MISC_REPOS.Layers.Special_Layers import *
# from RapidBase.MISC_REPOS.Layers.Unet_UpDown_Blocks import *
# from RapidBase.MISC_REPOS.Layers.Warp_Layers import *
from RapidBase.MISC_REPOS.Layers.Wrappers import *



class double_relu(nn.Module):
    def __init__(self, slope=0.05):
        super(double_relu, self).__init__()
        self.slope = slope

    def forward(self, x):
        x[x<0] = self.slope*x[x<0];
        x[x>1] = self.slope*x[x>1];
        return x;

class no_activation(nn.Module):
    def __init__(self):
        super(no_activation, self).__init__()

    def forward(self, x):
        return x;


###
class swish_torch(nn.Module):
    def __init__(self):
        super(swish_torch, self).__init__()
    def forward(self, x_tensor):
        return x_tensor / (1 + torch.exp(-x_tensor))
###
class inverse_torch01(nn.Module):
    def __init__(self):
        super(inverse_torch01, self).__init__()
        self.epsilon = 0.1
    def forward(self, x_tensor):
        return 1/(torch.abs(x_tensor) + self.epsilon)
###
class inverse_torch1(nn.Module):
    def __init__(self):
        super(inverse_torch1, self).__init__()
        self.epsilon = 1
    def forward(self, x_tensor):
        return 1/(torch.abs(x_tensor) + self.epsilon)
###
class linear_tanh01(nn.Module):
    def __init__(self):
        super(linear_tanh01, self).__init__()
        self.epsilon = 0.1
    def forward(self, x_tensor):
        return x_tensor/(torch.abs(x_tensor) + self.epsilon)
###
class linear_tanh1(nn.Module):
    def __init__(self):
        super(linear_tanh1, self).__init__()
        self.epsilon = 1
    def forward(self, x_tensor):
        return x_tensor/(torch.abs(x_tensor) + self.epsilon)
###
class linear_plus_inverse_learnable(nn.Module):
    def __init__(self):
        super(linear_inverse_learnable, self).__init__()
        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parmaeter(torch.Tensor([1]))
        self.epsilon = 0.2
    def forward(self, x_tensor):
        return self.a*x_tensor + self.b/(torch.abs(x_tensor)+self.epsilon)
###
class linear_plus_linear_inverse_learnable(nn.Module):
    def __init__(self):
        super(linear_inverse_learnable, self).__init__()
        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parmaeter(torch.Tensor([1]))
        self.epsilon = 0.2
    def forward(self, x_tensor):
        return self.a*x_tensor + self.b*x_tensor/(torch.abs(x_tensor)+self.epsilon)
###
class leakyrelu_plus_linear_inverse_learnable(nn.Module):
    def __init__(self):
        super(linear_inverse_learnable, self).__init__()
        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parmaeter(torch.Tensor([1]))
        self.epsilon = 0.2
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x_tensor):
        return self.a*self.leakyrelu(x_tensor) + self.b*x_tensor/(torch.abs(x_tensor)+self.epsilon)
###

# bla = matlab_arange(-4,0.05,4)
# bla = torch.Tensor(bla)
# activation = nn.LeakyReLU()(bla) + 1/(torch.abs(bla)+0.1)
# activation = bla + 1/(torch.abs(bla)+0.2)
# # activation = (bla) + bla/(torch.abs(bla)+0.5)
# bla = bla.numpy()
# activation = activation.numpy()
# plot(bla, activation)



class diode_torch(nn.Module):
    def __init__(self, flag_learnable=True):
        super(diode_torch, self).__init__()
        ### Parameters: ###
        self.V_zener_reverse = nn.Parameter(torch.Tensor([0]))
        self.V_zener_forward = nn.Parameter(torch.Tensor([1]))
        self.C1 = nn.Parameter(torch.Tensor([1]))
        self.C2 = nn.Parameter(torch.Tensor([1]))
        self.sigma = nn.Parameter(torch.Tensor([1]))
        self.flag_learnable = flag_learnable;
        if ~self.flag_learnable:
            self.mean.requires_grad = False
            self.sigma.requires_grad = False  ####################
    def forward(self, x_tensor):
        y_diode = A * (-torch.exp((-x_tensor - self.V_zener_reverse) / self.C1) + torch.exp((x_tensor-self.V_zener_forward) / self.C2) - self.B)
        return y_diode
###
class diode_torch_no_learning(nn.Module):
    def __init__(self, flag_learnable=True):
        super(diode_torch_no_learning, self).__init__()
        ### Parameters: ###
    def forward(self, x_tensor):
        y_diode = (-torch.exp(-x_tensor) + torch.exp(x_tensor-1))
        return y_diode
###
class abs_activation(nn.Module):
    def __init__(self, flag_learnable=True):
        super(abs_activation, self).__init__()
        ### Parameters: ###
    def forward(self, x_tensor):
        y_abs = torch.abs(x_tensor)
        return y_abs
###
class parabola_activation(nn.Module):
    #TODO: maybe make a learnable polynomial / parabola
    def __init__(self, flag_learnable=True):
        super(parabola_activation, self).__init__()
        ### Parameters: ###
    def forward(self, x_tensor):
        y_parabola = x_tensor**2
        return y_parabola
###
class normal_torch(nn.Module):
    def __init__(self, flag_learnable=False):
        super(normal_torch, self).__init__()
        ### Parameters: ###
        self.mean = nn.Parameter(torch.Tensor([0]))
        self.sigma = nn.Parameter(torch.Tensor([1]))
        self.flag_learnable = flag_learnable;
        if ~self.flag_learnable:
            self.mean.requires_grad = False
            self.sigma.requires_grad = False
        ####################
    def forward(self, x_tensor):
        y_normal = torch.exp(-(x_tensor - self.mean) ** 2 / self.sigma ** 2)
        return y_normal
###
class diode_torch_PositivityByExp(nn.Module):
    def __init__(self, flag_learnable=True):
        super(diode_torch_PositivityByExp, self).__init__()
        ### Diode Parameters: ###
        self.V_zener_reverse = nn.Parameter(torch.Tensor([0]))
        self.V_zener_forward = nn.Parameter(torch.Tensor([1]))
        # self.C1 = nn.Parameter(torch.Tensor([1]))
        # self.C2 = nn.Parameter(torch.Tensor([1]))
        # self.sigma = nn.Parameter(torch.Tensor([1]))
        # self.A_forward = nn.Parameter(torch.Tensor([1]))
        # self.A_backward = nn.Parameter(torch.Tensor([1]))
        self.flag_learnable = flag_learnable;
        ### Abs Exp Parameters: ###
        self.mean = nn.Parameter(torch.Tensor([0]))
        self.sigma = nn.Parameter(torch.Tensor([1]))

        ### Make Sure certain variables are Positive BY USING AN EXP: ###
        self.sigma = nn.Parameter(torch.Tensor([0]))
        self.sigma = torch.exp(self.sigma);
        self.C1 = nn.Parameter(torch.Tensor([0]))
        self.C1 = torch.exp(self.C1);
        self.C2 = nn.Parameter(torch.Tensor([0]))
        self.C2 = torch.exp(self.C2);
        self.A_forward = nn.Parameter(torch.Tensor([0]))
        self.A_forward = torch.exp(self.A_forward)
        self.A_backward = nn.Parameter(torch.Tensor([0]))
        self.A_backward = torch.exp(self.A_backward)

        ### Set Parameters as learnable or not: ###
        if ~self.flag_learnable:
            self.mean.requires_grad = False
            self.sigma.requires_grad = False  ####################
    def forward(self, x_tensor):
        y_diode = (self.A_backward*(-torch.exp((-x_tensor - self.V_zener_reverse) / self.C1)) +  self.A_forward*(torch.exp((x_tensor-self.V_zener_forward) / self.C2) - self.B)) * torch.exp(-torch.abs(x_tensor-self.mean)/self.sigma)
        return y_diode
###
class diode_torch_PositivityByAbs(nn.Module):
    def __init__(self, flag_learnable=True):
        super(diode_torch_PositivityByAbs, self).__init__()
        ### Diode Parameters: ###
        self.V_zener_reverse = nn.Parameter(torch.Tensor([0]))
        self.V_zener_forward = nn.Parameter(torch.Tensor([1]))
        self.C1 = nn.Parameter(torch.Tensor([1]))
        self.C2 = nn.Parameter(torch.Tensor([1]))
        self.sigma = nn.Parameter(torch.Tensor([1]))
        self.A_forward = nn.Parameter(torch.Tensor([1]))
        self.A_backward = nn.Parameter(torch.Tensor([1]))
        self.flag_learnable = flag_learnable;
        self.mean = nn.Parameter(torch.Tensor([0]))
        self.B = nn.Parameter(torch.Tensor([0]))
        ### Set Parameters as learnable or not: ###
        if ~self.flag_learnable:
            self.mean.requires_grad = False
            self.sigma.requires_grad = False  ####################

    def forward(self, x_tensor):
        y_diode = (torch.abs(self.A_backward) * (-torch.exp((-x_tensor - self.V_zener_reverse) / torch.abs(self.C1))) + torch.abs(self.A_forward) * (torch.exp((x_tensor - self.V_zener_forward) / torch.abs(self.C2)) - self.B))\
                  * torch.exp(-torch.abs(x_tensor - self.mean) / torch.abs(self.sigma))
        return y_diode
###
class diode_torch_OnlyCentersLearned(nn.Module):
    def __init__(self, flag_learnable=True):
        super(diode_torch_OnlyCentersLearned, self).__init__()
        ### Diode Parameters: ###
        self.V_zener_reverse = nn.Parameter(torch.Tensor([0]))
        self.V_zener_forward = nn.Parameter(torch.Tensor([1]))
        self.C1 = torch.Tensor([1])
        self.C2 = torch.Tensor([1])
        self.mean = nn.Parameter(torch.Tensor([0]))
        self.sigma = torch.Tensor([1])
        self.A_forward = torch.Tensor([1])
        self.A_backward = torch.Tensor([1])
        self.flag_learnable = flag_learnable;

        ### Set Parameters as learnable or not: ###
        if ~self.flag_learnable:
            self.mean.requires_grad = False
            self.sigma.requires_grad = False  ####################

    def forward(self, x_tensor):
        y_diode = (torch.abs(self.A_backward) * (-torch.exp((-x_tensor - self.V_zener_reverse) / torch.abs(self.C1))) + torch.abs(self.A_forward) * (torch.exp((x_tensor - self.V_zener_forward) / torch.abs(self.C2)) - self.B))\
                  * torch.exp(-torch.abs(x_tensor - self.mean) / torch.abs(self.sigma))
        return y_diode
###
class normal_derivative_torch(nn.Module):
    def __init__(self, flag_learnable=False):
        super(normal_derivative_torch, self).__init__()
        ### Parameters: ###
        self.mean = nn.Parameter(torch.Tensor([0]))
        self.sigma = nn.Parameter(torch.Tensor([1]))
        self.flag_learnable = flag_learnable;
        if ~self.flag_learnable:
            self.mean.requires_grad = False
            self.sigma.requires_grad = False
        ####################
    def forward(self, x_tensor):
        y_normal_derivative = x_tensor * torch.exp(-(x_tensor - self.mean) ** 2 / self.sigma ** 2)
        return y_normal_derivative
###
class normal_modified_torch(nn.Module):
    def __init__(self, flag_learnable=False):
        super(normal_modified_torch, self).__init__()
        ### Parameters: ###
        self.mean = nn.Parameter(torch.Tensor([0]))
        self.sigma = nn.Parameter(torch.Tensor([1]))
        self.flag_learnable = flag_learnable;
        if ~self.flag_learnable:
            self.mean.requires_grad = False
            self.sigma.requires_grad = False
        ####################
    def forward(self, x_tensor):
        y_normal = torch.exp(-(x_tensor - self.mean) ** 2 / self.sigma ** 2) + 0.1*x_tensor
        return y_normal
###
class normal_derivative_modified_torch(nn.Module):
    def __init__(self, flag_learnable=False):
        super(normal_derivative_modified_torch, self).__init__()
        ### Parameters: ###
        self.mean = nn.Parameter(torch.Tensor([0]))
        self.sigma = nn.Parameter(torch.Tensor([1]))
        self.flag_learnable = flag_learnable;
        if ~self.flag_learnable:
            self.mean.requires_grad = False
            self.sigma.requires_grad = False
        ####################
    def forward(self, x_tensor):
        y_normal_derivative_modified = x_tensor * torch.exp(-(x_tensor - self.mean) ** 2 / self.sigma ** 2) + 0.1 * x_tensor
        return y_normal_derivative_modified
###
class learnable_amplitude_torch(nn.Module):
    def __init__(self):
        super(learnable_amplitude_torch, self).__init__()
        self.A = nn.Parameter(torch.Tensor([1]))
    def forward(self, x_tensor):
        return self.A * x_tensor
###
class pseudo_binary_torch(nn.Module):
    def __init__(self):
        super(pseudo_binary_torch, self).__init__()
    def forward(self, x_tensor, m=3):
        mask1 = torch.abs(x_tensor) < 1/m
        mask2 = x_tensor >= 1/m
        mask3 = x_tensor <= -1/m
        output_tensor = torch.zeros_like(x_tensor)
        output_tensor[mask1] = m*x_tensor[mask1]
        output_tensor[mask2] = 1+1/m*x_tensor[mask2]
        output_tensor[mask3] = -1+1/m*x_tensor[mask3]
        return output_tensor
###
class pseudo_binary_SFT_torch(nn.Module):
    def __init__(self):
        super(pseudo_binary_SFT_torch, self).__init__()
    def forward(self, x_tensor, m=3):
        mask1 = torch.abs(x_tensor) < 1/m
        mask2 = x_tensor >= 1/m
        mask3 = x_tensor <= -1/m
        output_tensor = torch.zeros_like(x_tensor)
        output_tensor[mask1] = m*x_tensor[mask1]
        output_tensor[mask2] = 1+1/m*x_tensor[mask2]
        output_tensor[mask3] = -1+1/m*x_tensor[mask3]
        output_tensor = output_tensor * input_tensor
        return output_tensor
###



# m = 4
# mask1 = torch.abs(input_tensor) < 1/m
# mask2 = torch.abs(input_tensor) >= 1/m
# output_tensor = torch.zeros_like(input_tensor)
# output_tensor[mask1] = m*input_tensor
# output_tensor[mask2] = 1/m*input_tensor
#
#
# input_tensor = torch.Tensor(arange(-2,2,0.01))
# layer = pseudo_binary_SFT_torch()
# output_tensor = layer(input_tensor)
# plot_torch_xy(input_tensor, output_tensor)



####
class Pseudo_PixelWise_CrossCorrelation_Layer(nn.Module):
    def __init__(self):
        super(Pseudo_PixelWise_CrossCorrelation_Layer, self).__init__()

    def forward(self, x,y):
        output_tensor = 2*(x*y)/(x**2+y**2 + 1e-6)
        return output_tensor;


class Regularized_Inverse_Abs_Layer(nn.Module):
    def __init__(self, epsilon=0.05):
        super(Regularized_Inverse_Abs_Layer, self).__init__()
        self.epsilon = epsilon;
    def forward(self, x):
        output_tensor = 1/(abs(x)+self.epsilon)
        return output_tensor;


class Regularized_Inverse_Layer(nn.Module):
    def __init__(self, epsilon=0.05):
        super(Regularized_Inverse_Layer, self).__init__()
        self.epsilon = epsilon;
    def forward(self, x):
        output_tensor = 1/(x+self.epsilon)
        return output_tensor;


class Residual_Energy_Layer(nn.Module):
    def __init__(self):
        super(Residual_Energy_Layer, self).__init__()

    def forward(self, x,y):
        output_tensor = (x-y)**2
        return output_tensor;


class Energy_Residual_Layer(nn.Module):
    def __init__(self):
        super(Energy_Residual_Layer, self).__init__()

    def forward(self, x,y):
        output_tensor = x**2-y**2
        return output_tensor;


class Multiplication_Layer(nn.Module):
    def __init__(self):
        super(Multiplication_Layer, self).__init__()

    def forward(self, x,y):
        output_tensor = x*y
        return output_tensor;
####




class prelu_per_channel(nn.Module):
    def __init__(self):
        super(prelu_per_channel, self).__init__()
        self.number_of_channels = None
        self.activation_function = None
    def forward(self, x):
        if self.activation_function is None:
            self.activation_function = nn.PReLU(x.shape[1]).to(x.device)
        return self.activation_function(x)







#################################################################################################################################################################################################################################
def Activation_Function(activation_function, negative_activation_slope=0.2, prelu_number_of_parameters=1, flag_learnable_input_scale=False, flag_learnable_output_scale=False, inplace=True):
    #TODO: the learnable input/output scales is for all the channels....not the same as doing scaling for each kernel!!!
    # helper selecting activation
    # negative_activation_slope: for leakyrelu and init of prelu
    # prelu_number_of_parameters: for p_relu num_parameters

    activation_function = activation_function.lower()
    if activation_function == 'relu':
        # layer = nn.ReLU(inplace)
        layer = nn.ReLU(False)  #changed
    elif activation_function == 'leakyrelu':
        layer = nn.LeakyReLU(negative_activation_slope, False)
    elif activation_function == 'prelu':
        layer = nn.PReLU(num_parameters=prelu_number_of_parameters, init=negative_activation_slope)
    elif activation_function == 'prelu_per_channel':
        layer = prelu_per_channel()
    elif activation_function == 'double_relu':
        layer = double_relu(slope=0.05);
    elif activation_function == 'sigmoid':
        layer = nn.Sigmoid();
    elif activation_function == 'none':
        layer = Identity_Layer();
    elif activation_function == 'swish_torch':
        layer = swish_torch();
    elif activation_function == 'diode_torch':
        layer = diode_torch();
    elif activation_function == 'diode_torch_no_learning':
        layer = diode_torch_no_learning();
    elif activation_function == 'abs_activation':
        layer = abs_activation();
    elif activation_function == 'parabola_activation':
        layer = parabola_activation();
    elif activation_function == 'normal_torch':
        layer = normal_torch();
    elif activation_function == 'normal_derivative_torch':
        layer = normal_derivative_torch();
    elif activation_function == 'normal_derivative_modified_torch':
        layer = normal_derivative_modified_torch();
    elif activation_function == 'normal_modified_torch':
        layer = normal_modified_torch();
    elif activation_function == 'diode_torch_positivitybyexp':
        layer = diode_torch_PositivityByExp()
    elif activation_function == 'diode_torch_positivitybyabs':
        layer = diode_torch_PositivityByAbs()
    elif activation_function == 'diode_torch_onlycenterslearned':
        layer = diode_torch_OnlyCenteredLearned()
    elif activation_function == 'learnable_amplitude':
        layer = learnable_amplitude_torch()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_function)

    ### Learnable Input/Output Scales: ###
    #(*). Note! - diode_torch_PositivityByAbs is basically that.....you've got channel-uniform input scaling effectively using learnable sigma's and you've got learnable output scaling using learnable Amplitudes.....
    #     However... since in diode_torch_PositivityByAbs there are 2(!!!!) amplitudes learned that means that even though it's easier to learn 0 it STILL (probably) suffers from the behaviors i've seen....so maybe even then it's preferable to use learnable scaling
    if flag_learnable_input_scale:
        layer = nn.Sequential(learnable_amplitude_torch(), layer)
    if flag_learnable_output_scale:
        layer = nn.Sequential(layer, learnable_amplitude_torch())

    return layer


















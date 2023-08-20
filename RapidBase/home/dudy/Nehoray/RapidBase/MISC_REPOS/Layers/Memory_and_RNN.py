import RapidBase.Basic_Import_Libs
from RapidBase.Basic_Import_Libs import *

# Deformable Convolution (probably delete):
# from pytorch_deform_conv.torch_deform_conv.layers import *


### Set Up Decices: ###
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
device_cpu = torch.device('cpu')


# #(5). Layers:
# import RapidBase.MISC_REPOS.Layers.Activations
# import RapidBase.MISC_REPOS.Layers.Basic_Layers
# import RapidBase.MISC_REPOS.Layers.Conv_Blocks
# import RapidBase.MISC_REPOS.Layers.DownSample_UpSample
# # import RapidBase.MISC_REPOS.Layers.Memory_and_RNN
# import RapidBase.MISC_REPOS.Layers.Refinement_Modules
# import RapidBase.MISC_REPOS.Layers.Special_Layers
# import RapidBase.MISC_REPOS.Layers.Unet_UpDown_Blocks
# import RapidBase.MISC_REPOS.Layers.Warp_Layers
# import RapidBase.MISC_REPOS.Layers.Wrappers
# from RapidBase.MISC_REPOS.Layers.Activations import *
# from RapidBase.MISC_REPOS.Layers.Basic_Layers import *
# from RapidBase.MISC_REPOS.Layers.Conv_Blocks import *
# from RapidBase.MISC_REPOS.Layers.DownSample_UpSample import *
# # from RapidBase.MISC_REPOS.Layers.Memory_and_RNN import *
# from RapidBase.MISC_REPOS.Layers.Refinement_Modules import *
# from RapidBase.MISC_REPOS.Layers.Special_Layers import *
# from RapidBase.MISC_REPOS.Layers.Unet_UpDown_Blocks import *
# from RapidBase.MISC_REPOS.Layers.Warp_Layers import *
# from RapidBase.MISC_REPOS.Layers.Wrappers import *



# # Denoising Basic Layers (replace by RapidBase):
# import Denoising_Basic_Layers
# from Denoising_Basic_Layers import *
# from Denoising_Basic_Layers import Conv_Block
####################################################################################################################################################################################################################################










##############################################################################################################################################################################################################################################################
class GradientHighway(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size, dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        self.conv_P = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                                               out_channels=self.hidden_size,
                                               kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))
        self.conv_S = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                                               out_channels=self.hidden_size,
                                               kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))

        self.initial_conv1 = nn.Sequential(self.Padding, nn.Conv2d(input_size, hidden_size * 2, self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))


    def forward(self, input_, z_prev, reset_flags_list):
        # Get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                z_prev = Variable(torch.zeros(state_size)).to(self.conv_P.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                z_prev = Variable(torch.zeros(state_size)).to(self.conv_P.parameters().__next__().device)
                z_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])  ### Maybe Initialize ALL gates with input?. In any case...this doesn't make sense if i want to be able to generalize to any hidden_size
            elif reset_flag == 3:  # Detach:
                z_prev = z_prev.detach();
            elif reset_flag == 4:  # Special Conv
                z_prev = self.initial_conv1(input_)

        # print(input_.shape)
        # print(z_prev.shape)
        stacked_inputs = torch.cat([input_,z_prev],1)
        P = torch.tanh(self.conv_P(stacked_inputs))
        S = torch.sigmoid(self.conv_S(stacked_inputs))
        return S*P + (1-S)*z_prev






class GradientHighway2(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size, dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        self.conv_P = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                                               out_channels=self.hidden_size,
                                               kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))
        self.conv_S = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                                               out_channels=self.hidden_size,
                                               kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))

        self.initial_conv1 = nn.Sequential(self.Padding, nn.Conv2d(input_size, hidden_size, self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))
        self.flag_initial_conv = (input_size!=hidden_size)

    def forward(self, input_, z_prev, reset_flags_list):
        # Get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        stacked_inputs = torch.cat([input_,z_prev],1)
        S = torch.sigmoid(self.conv_S(stacked_inputs))

        if self.flag_initial_conv:
            input_ = self.initial_conv1(input_)
        return S*input_ + (1-S)*z_prev




class GradientHighway3(nn.Module):
    #NOT RECURRENT, NO HIDDEN STATE!
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size, dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        self.conv_P = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                                               out_channels=self.hidden_size,
                                               kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))
        self.conv_S = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                                               out_channels=self.hidden_size,
                                               kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))

        self.initial_conv1 = nn.Sequential(self.Padding, nn.Conv2d(input_size, hidden_size * 2, self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))


    def forward(self, input_, z_prev, reset_flags_list):
        # Get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        stacked_inputs = torch.cat([input_,z_prev],1)
        P = torch.tanh(self.conv_P(stacked_inputs))
        S = torch.sigmoid(self.conv_S(stacked_inputs))
        return S*P + (1-S)*z_prev



class GradientHighway4(nn.Module):
    #NOT RECURRENT, NO HIDDEN STATE!
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size, dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        self.conv_P = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                                               out_channels=self.hidden_size,
                                               kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))
        self.conv_S = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                                               out_channels=self.hidden_size,
                                               kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))

        self.initial_conv1 = nn.Sequential(self.Padding, nn.Conv2d(input_size, hidden_size * 2, self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))


    def forward(self, input_, z_prev, reset_flags_list):
        # Get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        stacked_inputs = torch.cat([input_,z_prev],1)
        P = (self.conv_P(stacked_inputs))
        S = torch.sigmoid(self.conv_S(stacked_inputs))
        return S*P + (1-S)*z_prev
##############################################################################################################################################################################################################################################################









####################################################################################################################################################################################################################################################################
############# ConvGRU2D: #############
class ConvGRU2DCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        self.reset_gate = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=-2)

        self.update_gate = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)

        self.out_gate = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)


        ### Parameter Or Conv To Initialize Hidden State: ###
        self.initial_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=0)



    def forward(self, input_, prev_state, reset_flags_list=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.reset_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.reset_gate.parameters().__next__().device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))  #TODO: notice the large bias towards output zero!!!
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state




class ConvGRU2D(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super(ConvGRU2D, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvGRU2DCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'ConvGRU2DCell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]

            # pass through layer:
            current_cell_output = cell(input_, cell_hidden, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################










#############################################################################################################################################################################################################################
class ConvLSTM2DCell(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 kernel_size, stride=1, dilation=1, groups=1,
                 normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='dirac',
                 flag_cell_residual=True):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.flag_cell_residual = flag_cell_residual

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        self.reset_gate = Conv_Block(input_size + hidden_size, 4*hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                      normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method,
                                     flag_deformable_convolution=flag_deformable_convolution,
                                     flag_deformable_convolution_version=flag_deformable_convolution_version,
                                     flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        self.update_gate = Conv_Block(hidden_size, 4 * hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method,
                                      flag_deformable_convolution=flag_deformable_convolution,
                                      flag_deformable_convolution_version=flag_deformable_convolution_version,
                                      flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                      flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)

        # self.reset_gate = nn.Conv2d(input_size+hidden_size, hidden_size*4, kernel_size, padding=0)
        # self.update_gate = nn.Conv2d(hidden_size, hidden_size*4, kernel_size, padding=0)


        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size//2)
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, C_prev, reset_flags_list=None):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)  #TODO: Assuming hx,cx are of same size (maybe generalize in the future)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(input_.device)
                C_prev = Variable(torch.zeros(state_size)).to(input_.device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(input_.device)
                C_prev = Variable(torch.zeros(state_size)).to(input_.device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, 0:self.input_size, :, :])
                C_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, 0:self.input_size, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
                C_prev = C_prev.detach();


        ### Make Sure These are in the correct order: ###
        hx = prev_state
        cx = C_prev

        ### Forward: ###
        # gates = self.reset_gate(self.Padding(torch.cat([hx,input_],1)))  #TODO: Maybe involve C_prev here???
        gates = self.reset_gate(torch.cat([hx,input_],1))  #TODO: Maybe involve C_prev here???
        # gates = self.reset_gate(torch.cat([hx,input_],1))  #TODO: Maybe involve C_prev here???   # THIS IS THE PROPER / CORRECT VERSION!!!!
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate+1)  ###TODO: Notice!!! i put a large bias (+1) in the forget gate because supposedly it allows better learning
        cellgate = F.tanh(cellgate)   #TODO: in the PredRNN paper there is a vesion of ConvLSTM2D where the ingate and forgetgate involve C_prev!!!!
                                      #TODO: Maybe give up on the F.tanh here in order to be able to send input pixels through
        outgate = F.sigmoid(outgate)

        # ### Temp!!! DELETE LATER!!!: ###
        # # print(gates.shape)
        # # print(input_.shape)
        # if gates.shape[2] > input_.shape[2]:
        #     # gates = gates[:,:,0:gates.shape[2]-4, 0:gates.shape[3]-4]
        #     input_ = F.pad(input_,[2,2,2,2])
        #     cx = F.pad(cx,[2,2,2,2])
        # # print('second Time: ' +str(gates.shape))
        # # print('second Time: ' +str(input_.shape))
        # # print(forgetgate.shape)
        # # print()

        cy = (forgetgate * cx) + (ingate * cellgate)  # element-wise multiplication. TODO: Some put Sigmoid here. maybe change this to something like: gate1*cx + gate2*hx + gate3*input_

        if self.flag_input_conv:
            input_ = self.input_conv(input_)

        if self.flag_cell_residual == 0:
            hy = outgate * torch.tanh(cy) #TODO: DELETE +input_!!! JUST CHECKING
        elif self.flag_cell_residual == 1:
            # print('input_ shape: ' + str(input_.shape))
            # print('cy shape: ' + str(cy.shape))
            hy = outgate * torch.tanh(cy) + input_
        elif self.flag_cell_residual == 2:
            hy = outgate * torch.tanh(cy) + hx
        elif self.flag_cell_residual == 3:
            hy = outgate * torch.tanh(cy) + hx + input_

        return hy,cy




# input_tensor = torch.Tensor(randn(1,3,100,100))
# bla = ConvLSTM2D_According_To_Flags_String(3, [30,30,3], n_layers=3, kernel_sizes=5)
# output_tensor = bla(input_tensor,[1])


class ConvLSTM2DCell2(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 kernel_size, stride=1, dilation=1, groups=1,
                 normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False, flag_deformable_same_on_all_channels=False,
                 initialization_method='dirac',
                 flag_cell_residual=0):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)
        self.flag_cell_residual = flag_cell_residual
        ### GRU Gates: ###
        self.update_gate = Conv_Block(input_size+hidden_size, 4 * hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                      normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                      flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        self.reset_gate = Conv_Block(input_size+hidden_size, 4 * hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                      normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                      flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)

        # self.reset_gate = nn.Conv2d(input_size+hidden_size, hidden_size*4, kernel_size, padding=0)
        # self.update_gate = nn.Conv2d(input_size+hidden_size, hidden_size*4, kernel_size, padding=0)


        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, C_prev, reset_flags_list=None):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)  #TODO: Assuming hx,cx are of same size (maybe generalize in the future)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(input_.device)
                C_prev = Variable(torch.zeros(state_size)).to(input_.device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(input_.device)
                C_prev = Variable(torch.zeros(state_size)).to(input_.device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, 0:self.input_size, :, :])
                C_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, 0:self.input_size, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
                C_prev = C_prev.detach();


        ### Make Sure These are in the correct order: ###
        hx = prev_state
        cx = C_prev

        ### Forward: ###
        gates = self.update_gate(torch.cat([hx,input_],1))  #TODO: Maybe involve C_prev here???
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate+1)
        cellgate = cellgate   #CHANGE - deleted F.tanh here!
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)  # element-wise multiplication. TODO: Some put Sigmoid here. maybe change this to something like: gate1*cx + gate2*hx + gate3*input_

        if self.flag_input_conv:
            input_ = self.input_conv(input_)

        if self.flag_cell_residual == 0:
            hy = outgate * torch.tanh(cy)
        elif self.flag_cell_residual == 1:
            hy = outgate * torch.tanh(cy) + input_  #TODO: maybe delete tanh here??!?!?!?
        elif self.flag_cell_residual == 2:
            hy = outgate * torch.tanh(cy) + hx
        elif self.flag_cell_residual == 3:
            hy = outgate * torch.tanh(cy) + hx + input_

        return hy,cy


class ConvLSTM2DCell3(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False, flag_deformable_same_on_all_channels=False,
                 initialization_method='dirac',
                 flag_cell_residual=0):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size, dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)
        self.flag_cell_residual = flag_cell_residual
        ### GRU Gates: ###
        self.update_gate = Conv_Block(input_size + hidden_size, 4 * hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                      normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                      flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        self.reset_gate = Conv_Block(input_size + hidden_size, 4 * hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        # self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size * 4, kernel_size, padding=0)
        # self.update_gate = nn.Conv2d(input_size+hidden_size, hidden_size * 4, kernel_size, padding=0)


        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, C_prev, reset_flags_list=None):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)  # TODO: Assuming hx,cx are of same size (maybe generalize in the future)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(input_.device)
                C_prev = Variable(torch.zeros(state_size)).to(input_.device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(input_.device)
                C_prev = Variable(torch.zeros(state_size)).to(input_.device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, 0:self.input_size, :, :])
                C_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, 0:self.input_size, :, :])
            elif reset_flag == 3:  # Detach:
                prev_state = prev_state.detach();
                C_prev = C_prev.detach();

        ### Make Sure These are in the correct order: ###
        hx = prev_state
        cx = C_prev

        ### Forward: ###
        gates = self.update_gate(torch.cat([hx, input_],1))  # TODO: Maybe involve C_prev here???
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate + 1)
        cellgate = cellgate  # CHANGE - deleted F.tanh here!
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)  # element-wise multiplication. TODO: Some put Sigmoid here. maybe change this to something like: gate1*cx + gate2*hx + gate3*input_

        if self.flag_input_conv:
            input_ = self.input_conv(input_)

        if self.flag_cell_residual == 0:
            hy = outgate * cy
        elif self.flag_cell_residual == 1:
            hy = outgate * cy + input_  # TODO: maybe delete tanh here??!?!?!?
        elif self.flag_cell_residual == 2:
            hy = outgate * cy + hx
        elif self.flag_cell_residual == 3:
            hy = outgate * cy + hx + input_

        return hy, cy







class ConvLSTM2DCell4(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False, flag_deformable_same_on_all_channels=False,
                 initialization_method='dirac',
                 flag_cell_residual=0):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.flag_cell_residual = flag_cell_residual
        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size, dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        self.update_gate = Conv_Block(input_size + hidden_size, 4 * hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                      normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                      flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        self.reset_gate = Conv_Block(input_size + hidden_size, 4 * hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        # self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size * 4, kernel_size, padding=0)
        # self.update_gate = nn.Conv2d(input_size+hidden_size, hidden_size * 4, kernel_size, padding=0)

        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, C_prev, reset_flags_list=None):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)  # TODO: Assuming hx,cx are of same size (maybe generalize in the future)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(input_.device)
                C_prev = Variable(torch.zeros(state_size)).to(input_.device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(input_.device)
                C_prev = Variable(torch.zeros(state_size)).to(input_.device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, 0:self.input_size, :, :])
                C_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, 0:self.input_size, :, :])
            elif reset_flag == 3:  # Detach:
                prev_state = prev_state.detach();
                C_prev = C_prev.detach();

        ### Make Sure These are in the correct order: ###
        hx = prev_state
        cx = C_prev

        ### Forward: ###
        gates = self.update_gate(torch.cat([hx, input_],1))  # TODO: Maybe involve C_prev here???
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate + 1)
        cellgate = cellgate  # CHANGE - deleted F.tanh here!
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)  # element-wise multiplication. TODO: Some put Sigmoid here. maybe change this to something like: gate1*cx + gate2*hx + gate3*input_

        if self.flag_input_conv:
            input_ = self.input_conv(input_)


        if self.flag_cell_residual == 0:
            hy = outgate * cy + (1-outgate) * hx
        elif self.flag_cell_residual == 1:
            hy = outgate * cy + (1-outgate) * hx + input_  # TODO: maybe delete tanh here??!?!?!?
        elif self.flag_cell_residual == 2:
            hy = outgate * cy + (1-outgate) * hx + hx
        elif self.flag_cell_residual == 3:
            hy = outgate * cy + (1-outgate) * hx + hx + input_

        return hy, cy







class ConvLSTM2DCell5(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None, flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False, flag_deformable_same_on_all_channels=False,
                 initialization_method='dirac',
                 flag_cell_residual=0):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.flag_cell_residual = flag_cell_residual
        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size, dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        self.update_gate = Conv_Block(input_size + hidden_size, 4 * hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                      normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                      flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        self.reset_gate = Conv_Block(input_size + hidden_size, 4 * hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        # self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size * 4, kernel_size, padding=0)
        # self.update_gate = nn.Conv2d(input_size+hidden_size, hidden_size * 4, kernel_size, padding=0)

        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, C_prev, reset_flags_list=None):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)  # TODO: Assuming hx,cx are of same size (maybe generalize in the future)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(input_.device)
                C_prev = Variable(torch.zeros(state_size)).to(input_.device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(input_.device)
                C_prev = Variable(torch.zeros(state_size)).to(input_.device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, 0:self.input_size, :, :])
                C_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, 0:self.input_size, :, :])
            elif reset_flag == 3:  # Detach:
                prev_state = prev_state.detach();
                C_prev = C_prev.detach();

        ### Make Sure These are in the correct order: ###
        hx = prev_state
        cx = C_prev

        ### Forward: ###
        gates = self.update_gate(torch.cat([hx, input_],1))  # TODO: Maybe involve C_prev here???
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate + 1)
        cellgate = cellgate  # CHANGE - deleted F.tanh here!
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)  # element-wise multiplication. TODO: Some put Sigmoid here. maybe change this to something like: gate1*cx + gate2*hx + gate3*input_

        if self.flag_input_conv:
            input_ = self.input_conv(input_)


        if self.flag_cell_residual == 0:
            hy = outgate * cy + (1-outgate) * input_
        elif self.flag_cell_residual == 1:
            hy = outgate * cy + (1-outgate) * input_ + input_  # TODO: maybe delete tanh here??!?!?!?
        elif self.flag_cell_residual == 2:
            hy = outgate * cy + (1-outgate) * input_ + hx
        elif self.flag_cell_residual == 3:
            hy = outgate * cy + (1-outgate) * input_ + hx + input_

        return hy, cy







class ConvLSTM2D(nn.Module):
    def __init__(self, input_size,
                 hidden_sizes,
                 n_layers,
                 kernel_sizes, strides=1, dilations=1, groups=1,
                 normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='dirac',
                 flag_cell_residual=0,
                 flag_stack_residual=0,
                 flag_gradient_highway_type = 0,
                 flag_gradient_highway_strategy=0,
                 flag_gradient_highway_none_only_first_or_all=0,
                 flag_ConvLSTMCell_type=0):
        super(ConvLSTM2D, self).__init__()
        self.input_size = input_size
        self.n_layers = n_layers

        self.flag_stack_residual = flag_stack_residual
        self.flag_gradient_highway_type = flag_gradient_highway_type
        self.flag_gradient_highway_strategy = flag_gradient_highway_strategy

        ### Choose gradient highway function: ###
        if self.flag_gradient_highway_type == 0:
            Gradient_Highway_Function = GradientHighway
        elif self.flag_gradient_highway_type == 1:
            Gradient_Highway_Function = GradientHighway2
        elif self.flag_gradient_highway_type == 2:
            Gradient_Highway_Function = GradientHighway3

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups,\
        self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, \
        self.initialization_methods, self.flag_cell_residual =\
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups,
                                      flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels,
                                      initialization_method, flag_cell_residual)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        GradientHighways_list = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]
            ### Choose LSTM Cell type: ###
            if flag_ConvLSTMCell_type==0:
                cell = ConvLSTM2DCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i], self.flag_cell_residual[i])
            if flag_ConvLSTMCell_type==1:
                cell = ConvLSTM2DCell2(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i], self.flag_cell_residual[i])
            if flag_ConvLSTMCell_type==2:
                cell = ConvLSTM2DCell3(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i], self.flag_cell_residual[i])
            if flag_ConvLSTMCell_type==3:
                cell = ConvLSTM2DCell4(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i], self.flag_cell_residual[i])
            if flag_ConvLSTMCell_type==4:
                cell = ConvLSTM2DCell5(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i], self.flag_cell_residual[i])
            name = 'ConvLSTM2DCell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
            # (2). Get GradientHighway Cell:
            if flag_gradient_highway_strategy == 0:
                GradientHighway_current = Gradient_Highway_Function(self.hidden_sizes[i], self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            if flag_gradient_highway_strategy == 1:
                GradientHighway_current = Gradient_Highway_Function(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            if flag_gradient_highway_strategy == 2:
                GradientHighway_current = Gradient_Highway_Function(input_size, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            if flag_gradient_highway_strategy == 3:
                GradientHighway_current = Gradient_Highway_Function(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            if flag_gradient_highway_strategy == 4:
                GradientHighway_current = Gradient_Highway_Function(input_size, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])

            name = 'GradientHighway_Cell' + str(i).zfill(2)
            setattr(self, name, GradientHighway_current)
            GradientHighways_list.append(getattr(self, name))
        #############################

        self.cells = cells
        self.GradientHighways = GradientHighways_list;

        ### Initialize Hidden States: ###
        self.hidden_H = [None] * self.n_layers;
        self.hidden_C = [None] * self.n_layers;
        self.hidden_Z = [None] * self.n_layers;

        ### Gradient Highway Layers: ####
        if flag_gradient_highway_none_only_first_or_all == 0:
            self.flags_use_GradientHighway = [False] * self.n_layers;
        elif flag_gradient_highway_none_only_first_or_all == 1:
            self.flags_use_GradientHighway = [False] * self.n_layers;
            self.flags_use_GradientHighway[0] = True;
        elif flag_gradient_highway_none_only_first_or_all == 2:
            self.flags_use_GradientHighway = [True] * self.n_layers;

        self.reset_flags_list = False;

        self.last_output = 0


        if self.flag_stack_residual:
            self.projection_block = nn.Sequential(nn.ReflectionPad2d(2) ,nn.Conv2d(input_size,hidden_sizes[-1],kernel_size=5))  #TODO: projection block is usually 1x1 conv!!!!.....



    def reset_hidden_states(self):
        self.hidden_H = [None]*self.n_layers;

    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;
        if type(self.hidden_H[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_H)):
                self.hidden_H[hidden_state_index] = self.hidden_H[hidden_state_index].detach();
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_H[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_H[hidden_state_index].device;
                        self.hidden_H[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_H = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_H:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list):
        input_ = x
        original_input = x;

        ### Save last final output in case i want to do something with it (maybe pass it as another initial input the next time step (in PredRNN there is a whole variable M dedicated to it): ###
        reset_flag = reset_flags_list[0]
        if reset_flag == 0:  # Do Nothing
            1;
        elif reset_flag == 1:  # Initialize with zeros
            self.last_output = Variable(torch.zeros(input_.shape)).to(input_.device)
        elif reset_flag == 2:  # Initialize with input_
            self.last_output = Variable(torch.zeros(input_.shape)).to(input_.device)
            self.last_output[:, 0:self.input_size, :, :].copy_(input_[:, 0:self.input_size, :, :])
        elif reset_flag == 3:  # Detach:
            self.last_output = self.last_output.detach();


        ### Loop Over Layers: ###
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]

            # pass through layer:
            current_H, current_C = cell(input_, self.hidden_H[layer_idx], self.hidden_C[layer_idx], reset_flags_list)
            # update current layer hidden state list:
            self.hidden_H[layer_idx] = current_H;
            self.hidden_C[layer_idx] = current_C;
            # use GradientHighway if wanted:
            #TODO: understand the different possibilities here!!!
            if self.flags_use_GradientHighway[layer_idx]:
                if self.flag_gradient_highway_strategy == 0: #Recurrent with hidden state
                    current_Z = self.GradientHighways[layer_idx](current_H, self.hidden_Z[layer_idx], reset_flags_list)
                elif self.flag_gradient_highway_strategy == 1:
                    # print(input_.shape)
                    # print(current_H.shape)
                    current_Z = self.GradientHighways[layer_idx](input_, current_H, reset_flags_list) #Assumes sequential layers have same hidden_size
                elif self.flag_gradient_highway_strategy == 2:
                    current_Z = self.GradientHighways[layer_idx](original_input, current_H, reset_flags_list) #Assumes input_size = hidden_size for all layers!!
                elif self.flag_gradient_highway_strategy == 3:
                    current_Z = self.GradientHighways[layer_idx](input_, self.hidden_Z[layer_idx], reset_flags_list)
                elif self.flag_gradient_highway_strategy == 4:
                    current_Z = self.GradientHighways[layer_idx](original_input, self.hidden_Z[layer_idx], reset_flags_list)

                ### Assign Final: ###
                self.hidden_Z[layer_idx] = current_Z;
                input_ = current_Z;
            else:
                input_ = current_H;


        ### Get final output (maybe use residual network with original input to stack...maybe add another option to concat or maybe concat and conv....): ###
        if self.flag_stack_residual == 0:
            final_output = input_
        elif self.flag_stack_residual == 1:
            if self.input_size == self.hidden_sizes[-1]:
                final_output = input_ + original_input
            else:
                final_output = input_ + self.projection_block(original_input)

        # ### Accumulation by adding to this result previous time step result (There must be other tricks....): ###
        # final_output = input_ + self.last_output
        # self.last_output = final_output.detach()


        return final_output
#############################################################################################################################################################################################################################
























##############################################################################################################################################################################################################################################################
class PredRNN_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None, flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False, initialization_method='combined', M_size=3):

        super().__init__()
        padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.M_size = M_size;

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### PredRNN Gates: ###
        self.conv_IH = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                                               out_channels=self.hidden_size*3,
                                               kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))
        self.conv_IM = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.M_size,
                                         out_channels=self.M_size * 3,
                                         kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))

        self.conv_o = nn.Sequential(self.Padding,
                                    nn.Conv2d(in_channels=self.input_size + 2 * self.hidden_size + self.M_size,
                                        out_channels=self.hidden_size,
                                        kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups,  padding=0))

        self.conv_c = nn.Sequential(self.Padding,
                                    nn.Conv2d(in_channels=self.hidden_size + self.M_size,
                                        out_channels=self.hidden_size,
                                        kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups,  padding=0))

        ### Parameter Or Conv To Initialize Hidden State: ###
        self.initial_conv1 = nn.Sequential(self.Padding,nn.Conv2d(input_size, hidden_size*2, self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))
        self.initial_conv2 = nn.Sequential(self.Padding,nn.Conv2d(input_size, hidden_size * 2, self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))

        ### Initialization Method: ###
        initialize_weights(initialization_method, self.conv_IH[1], self.conv_IM[1], self.conv_o[1], self.conv_c[1])

        ### Instance Norm (i don't want to deal with BN for now): ###
        if normalization_type:
            self.conv_IH = nn.Sequential(self.conv_IH, Normalization_Function(normalization_type, number_of_input_channels=hidden_size))
            self.conv_IM = nn.Sequential(self.conv_IM, Normalization_Function(normalization_type, number_of_input_channels=hidden_size))
            self.conv_o = nn.Sequential(self.conv_o, Normalization_Function(normalization_type, number_of_input_channels=hidden_size))
            self.conv_c = nn.Sequential(self.conv_c, Normalization_Function(normalization_type, number_of_input_channels=hidden_size))

        ### Deformable Convolution: ###
        if flag_deformable_convolution:
            self.conv_IH = nn.Sequential(self.conv_IH, ConvOffset2D(hidden_size))
            self.conv_IM = nn.Sequential(self.conv_IM, ConvOffset2D(hidden_size))
            self.conv_o = nn.Sequential(self.conv_o, ConvOffset2D(hidden_size))
            self.conv_c = nn.Sequential(self.conv_c, ConvOffset2D(hidden_size))



    def forward(self, input_, H_prev, C_prev, M_prev, reset_flags_list=None):
        # Get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                H_prev = Variable(torch.zeros(state_size)).to(self.conv_IH.parameters().__next__().device)
                C_prev = Variable(torch.zeros(state_size)).to(self.conv_IH.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                H_prev = Variable(torch.zeros(state_size)).to(self.conv_IH.parameters().__next__().device)
                C_prev = Variable(torch.zeros(state_size)).to(self.conv_IH.parameters().__next__().device)
                H_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])  ### Maybe Initialize ALL gates with input?. In any case...this doesn't make sense if i want to be able to generalize to any hidden_size
                C_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])  ### Maybe Initialize ALL gates with input?. In any case...this doesn't make sense if i want to be able to generalize to any hidden_size
            elif reset_flag == 3:  # Detach:
                H_prev = H_prev.detach();
                C_prev = C_prev.detach();
            elif reset_flag == 4:  # Special Conv
                H_prev = self.initial_conv1(input_)
                C_prev = self.initial_conv2(input_)


        ### ~Classic LSTM: ###
        #TODO: if i want to use LayerNorm then i will probably need to define much more (smaller) nn.Conv2d becaue i think i need to normalize each conv output seperately.....
        combined_conv_xh = self.conv_IH(torch.cat([input_, H_prev], dim=1))
        i,f,g = combined_conv_xh.chunk(3,1)
        g = torch.tanh(g)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        C_new = f * C_prev + i * g

        ### HighWay: ###
        combined_conv_xM = self.conv_IM(torch.cat([input_, M_prev], dim=1))
        i_,f_,g_ = combined_conv_xM.chunk(3,1)
        g_ = torch.tanh(g_)
        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_)
        M_new = f_ * M_prev + i_ * g_   #TODO: in the PredRNN++ paper this equation is supposed to be:  M_new = f_*tanh(conv2D(M_prev) + i_*g_  .....

        ### Final Outputs: ###
        o = torch.sigmoid(self.conv_o(torch.cat([input_, M_new, C_new, H_prev], dim=1)))
        H_new = o * torch.tanh(self.conv_c(torch.cat([C_new, M_new], dim=1)))

        return H_new,C_new,M_new





class PredRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False, initialization_method='combined'):
        super(PredRNN, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers
        self.M_size = input_size; #for now i assume all M's are of the same number of channels..
        # TODO: understand if this is a necessity or i can Generalize....i think i can generalize but am limited by the fact that the FINAL M should equal the beginning becaue it reconnects on itself

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation,self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        GradientHighways = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = PredRNN_Cell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i], self.M_size)
            name = 'PredRNN_Cell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
            # (2). Get GradientHighway Cell:
            GradientHighway_current = GradientHighway(self.hidden_sizes[i], self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'GradientHighway_Cell' + str(i).zfill(2)
            setattr(self, name, GradientHighway_current)
            GradientHighways.append(getattr(self, name))
        #############################

        self.cells = cells
        self.GradientHighways = GradientHighways;
        self.hidden_states = [None]*self.n_layers;

        ### Initialize Hidden States: ###
        self.hidden_H = [None] * self.n_layers;
        self.hidden_C = [None] * self.n_layers;
        self.hidden_Z = [None] * self.n_layers;
        self.flags_use_GradientHighway = [True] * self.n_layers;
        self.M_prev = [None];

        ### Initialize Reset Flags: ###
        self.reset_flags_list = False;



    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;

    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, input_, reset_flags_list=[2]):
        # Get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.M_size] + list(spatial_size)

        # Initialize M State by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                self.M_prev = Variable(torch.zeros(state_size)).to(input_.device)
            elif reset_flag == 2:  # Initialize with input_
                self.M_prev = Variable(torch.zeros(state_size)).to(input_.device)
                self.M_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])  ### Maybe Initialize ALL gates with input?. In any case...this doesn't make sense if i want to be able to generalize to any hidden_size
            elif reset_flag == 3:  # Detach:
                self.M_prev = self.M_prev.detach();
            elif reset_flag == 4:  # Special Conv
                self.M_prev = self.initial_conv(input_)


        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]

            # pass through layer:
            current_H, current_C, current_M = cell(input_, self.hidden_H[layer_idx], self.hidden_C[layer_idx], self.M_prev, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_H[layer_idx] = current_H;
            self.hidden_C[layer_idx] = current_C;
            self.M_prev = current_M;
            # use GradientHighway if wanted:
            if self.flags_use_GradientHighway[layer_idx]:
                current_Z = self.GradientHighways[layer_idx](current_H, self.hidden_Z[layer_idx], reset_flags_list)   #TODO: I Changed the first argument from current_H and input_!!!@
                self.hidden_Z[layer_idx] = current_Z;
                input_ = current_Z;
            else:
                input_ = current_H;

        return input_
#############################################################################################################################################################################################################################















#############################################################################################################################################################################################################################
class PredRNNPP_Cell(nn.Module):
    ### So Called: Causal LSTM in the PredRNN++ paper: ###
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False, initialization_method='combined',M_size=3):

        super().__init__()
        padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.M_size = M_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### PredRNN Gates: ###
        self.conv_IH = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.hidden_size + self.hidden_size,
                                               out_channels=self.hidden_size*3,
                                               kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))
        self.conv_IM = nn.Sequential(self.Padding,
                                     nn.Conv2d(in_channels=self.input_size + self.hidden_size + self.M_size,
                                         out_channels=self.M_size * 3,
                                         kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))

        self.conv_o = nn.Sequential(self.Padding,
                                    nn.Conv2d(in_channels=self.input_size + self.hidden_size + self.M_size,
                                        out_channels=self.hidden_size,
                                        kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups,  padding=0))

        self.conv_c = nn.Sequential(self.Padding,
                                    nn.Conv2d(in_channels=self.hidden_size + self.M_size,
                                        out_channels=self.hidden_size,
                                        kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups,  padding=0))

        self.M_conv = nn.Sequential(self.Padding,
                                    nn.Conv2d(in_channels = self.M_size,
                                              out_channels = self.M_size,
                                              kernel_size=self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))

        ### Parameter Or Conv To Initialize Hidden State: ###
        self.initial_conv1 = nn.Sequential(self.Padding, nn.Conv2d(input_size, hidden_size * 2, self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))
        self.initial_conv2 = nn.Sequential(self.Padding, nn.Conv2d(input_size, hidden_size * 2, self.kernel_size, stride=stride, dilation=dilation, groups=groups, padding=0))

        ### Initialization Method: ###
        initialize_weights(initialization_method, self.conv_IH[1], self.conv_IM[1], self.conv_o[1], self.conv_c[1])

        ### Instance Norm (i don't want to deal with BN for now): ###
        if normalization_type:
            self.conv_IH = nn.Sequential(self.conv_IH, Normalization_Function(normalization_type, number_of_input_channels=hidden_size))
            self.conv_IM = nn.Sequential(self.conv_IM, Normalization_Function(normalization_type, number_of_input_channels=hidden_size))
            self.conv_o = nn.Sequential(self.conv_o, Normalization_Function(normalization_type, number_of_input_channels=hidden_size))
            self.conv_c = nn.Sequential(self.conv_c, Normalization_Function(normalization_type, number_of_input_channels=hidden_size))
            self.M_conv = nn.Sequential(self.M_conv, Normalization_Function(normalization_type, number_of_input_channels=hidden_size))

        ### Deformable Convolution: ###
        if flag_deformable_convolution:
            self.conv_IH = nn.Sequential(self.conv_IH, ConvOffset2D(hidden_size))
            self.conv_IM = nn.Sequential(self.conv_IM, ConvOffset2D(hidden_size))
            self.conv_o = nn.Sequential(self.conv_o, ConvOffset2D(hidden_size))
            self.conv_c = nn.Sequential(self.conv_c, ConvOffset2D(hidden_size))
            self.M_conv = nn.Sequential(self.M_conv, ConvOffset2D(hidden_size))



    def forward(self, input_, H_prev, C_prev, M_prev, reset_flags_list=None):
        # Get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                H_prev = Variable(torch.zeros(state_size)).to(self.conv_IH.parameters().__next__().device)
                C_prev = Variable(torch.zeros(state_size)).to(self.conv_IH.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                H_prev = Variable(torch.zeros(state_size)).to(self.conv_IH.parameters().__next__().device)
                C_prev = Variable(torch.zeros(state_size)).to(self.conv_IH.parameters().__next__().device)
                H_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])  ### Maybe Initialize ALL gates with input?. In any case...this doesn't make sense if i want to be able to generalize to any hidden_size
                C_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])  ### Maybe Initialize ALL gates with input?. In any case...this doesn't make sense if i want to be able to generalize to any hidden_size
            elif reset_flag == 3:  # Detach:
                H_prev = H_prev.detach();
                C_prev = C_prev.detach();
            elif reset_flag == 4:  # Special Conv
                H_prev = self.initial_conv1(input_)
                C_prev = self.initial_conv2(input_)


        ### ~Classic LSTM: ###
        #TODO: if i want to use LayerNorm then i will probably need to define much more (smaller) nn.Conv2d becaue i think i need to normalize each conv output seperately.....
        combined_conv_xh = self.conv_IH(torch.cat([input_, H_prev, C_prev], dim=1))
        i,f,g = combined_conv_xh.chunk(3,1)
        g = torch.tanh(g)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        C_new = f * C_prev + i * g

        ### HighWay: ###
        combined_conv_xM = self.conv_IM(torch.cat([input_, C_prev, M_prev], dim=1))
        i_,f_,g_ = combined_conv_xM.chunk(3,1)
        g_ = torch.tanh(g_)
        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_)
        M_prev_conved = torch.tanh(self.M_conv(M_prev))
        M_new = f_ * M_prev_conved + i_ * g_

        ### Final Outputs: ###
        o = torch.tanh(self.conv_o(torch.cat([input_, C_prev, M_new], dim=1)))
        H_new = o * torch.tanh(self.conv_c(torch.cat([C_new, M_new], dim=1)))

        return H_new,C_new,M_new





class PredRNNPP(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False, initialization_method='combined'):
        super(PredRNNPP, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers
        self.M_size = input_size; #for now i assume all M's are of the same number of channels..
        # TODO: understand if this is a necessity or i can Generalize....i think i can generalize but am limited by the fact that the FINAL M should equal the beginning becaue it reconnects on itself

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        GradientHighways = [];
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            #(1). Get PredRNNPP Cell:
            cell = PredRNNPP_Cell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i],self.M_size )
            name = 'PredRNNPP_Cell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
            #(2). Get GradientHighway Cell:
            GradientHighway_current = GradientHighway(self.hidden_sizes[i],self.hidden_sizes[i],self.kernel_sizes[i],self.strides[i],self.dilations[i],self.groups[i],normalization_type,self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i],self.initialization_methods[i])
            name = 'GradientHighway_Cell' + str(i).zfill(2)
            setattr(self, name, GradientHighway_current)
            GradientHighways.append(getattr(self, name))
        #############################

        self.cells = cells
        self.GradientHighways = GradientHighways;
        self.hidden_states = [None]*self.n_layers;

        ### Initialize Hidden States: ###
        self.hidden_H = [None] * self.n_layers;
        self.hidden_C = [None] * self.n_layers;
        self.hidden_Z = [None] * self.n_layers;
        self.flags_use_GradientHighway = [True]*self.n_layers;
        self.M_prev = [None];

        ### Initialize Reset Flags: ###
        self.reset_flags_list = False;



    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;

    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, input_, reset_flags_list=[2]):
        # Get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.M_size] + list(spatial_size)

        # Initialize M State by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                self.M_prev = Variable(torch.zeros(state_size)).to(input_.device)
            elif reset_flag == 2:  # Initialize with input_
                self.M_prev = Variable(torch.zeros(state_size)).to(input_.device)
                self.M_prev[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])  ### Maybe Initialize ALL gates with input?. In any case...this doesn't make sense if i want to be able to generalize to any hidden_size
            elif reset_flag == 3:  # Detach:
                self.M_prev = self.M_prev.detach();
            elif reset_flag == 4:  # Special Conv
                self.M_prev = self.initial_conv(input_)


        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]

            # pass through layer:
            current_H, current_C, current_M = cell(input_, self.hidden_H[layer_idx], self.hidden_C[layer_idx], self.M_prev, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_H[layer_idx] = current_H;
            self.hidden_C[layer_idx] = current_C;
            self.M_prev = current_M;
            # use GradientHighway if wanted:
            if self.flags_use_GradientHighway[layer_idx]:
                current_Z = self.GradientHighways[layer_idx](current_H, self.hidden_Z[layer_idx], reset_flags_list) #TODO: I Changed the first argument from current_H and input_!!!@
                self.hidden_Z[layer_idx] = current_Z;
                input_ = current_Z;
            else:
                input_ = current_H

        return input_
#############################################################################################################################################################################################################################



















####################################################################################################################################################################################################################################################################
############# ConvGRU2D_LinearCombination: #############
class ConvGRU2DCell_LinearCombination(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        self.update_gate = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                      normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                      flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        self.reset_gate = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        self.out_gate = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                      normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                      flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        # self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)
        # self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)
        # self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)


    def forward(self, input_, prev_state, reset_flags_list=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.reset_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.reset_gate.parameters().__next__().device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(self.Padding(stacked_inputs)))
        reset = F.sigmoid(self.reset_gate(self.Padding(stacked_inputs)))
        out_inputs = (self.out_gate(self.Padding(torch.cat([input_, prev_state * reset], dim=1)))) #removed non linearity
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state




class ConvGRU2D_LinearCombination(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super(ConvGRU2D_LinearCombination, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvGRU2DCell_LinearCombination(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'ConvGRU2DCell_LinearCombination_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]

            # pass through layer:
            current_cell_output = cell(input_, cell_hidden, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################










####################################################################################################################################################################################################################################################################
############# ConvGRU2D_AlphaEstimator: #############
class ConvGRU2DCell_AlphaEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        #TODO: Generalize this block which calculates alpha!...
        #TODO: Check whether adding a GradNorm or GradientPenalty Solution to RNNs improves something

        #(*). TODO: maybe change from deep sequential_conv_block to something else because if i concatenate layers it can become very deep
        self.update_gate = Sequential_Conv_Block(input_size+hidden_size,  #TODO: before it was input_size+hidden_size
                              [2*hidden_size,2*hidden_size,1],
                              [kernel_size,kernel_size,kernel_size],
                              strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None, activation_type=['leakyrelu','leakyrelu','none'], mode='CNA', flag_resnet=False, flag_concat=False)

        self.reset_gate = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                   normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                   flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        # self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)

        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, reset_flags_list=None):
        #TODO: the problem (or loss of generality) here is that hidden_size needs to equal input_size..... this puts some restriction....maybe use projection layers or something else...

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = self.update_gate(stacked_inputs);
        update = torch.sigmoid(update)
        #TODO: maybe instead of Summing the contributions Concatenate them and then use Conv layers: Sequential_Conv_Block([prev_state*(1-update) , input_*update])

        if self.flag_input_conv:
            input_ = self.input_conv(input_)

        new_state = (1-update)*prev_state + (update)*input_;
        return new_state



class ConvGRU2D_AlphaEstimator(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super(ConvGRU2D_AlphaEstimator, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvGRU2DCell_AlphaEstimator(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'ConvGRU2DCell_AlphaEstimator_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list=[2]):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]

            # pass through layer:
            current_cell_output = cell(input_, cell_hidden, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################







# #### Special Information Layers: ###
# input_tensor = read_image_torch_default();
# average_tensor = 0;
# counter = 1;
# for i in arange(5):
#     input_tensor_current = input_tensor + 0.1*torch.randn(input_tensor.shape)
#     alpha_t = 1/counter;
#     average_tensor = average_tensor*(1-alpha_t) + input_tensor_current*(alpha_t)
#     counter += 1
#
#
# ### Cross Correlation: ###
# output_tensor1 = Pseudo_PixelWise_CrossCorrelation_Layer()(input_tensor_current,average_tensor)
# output_tensor2 = Pseudo_PixelWise_CrossCorrelation_Layer()(input_tensor,average_tensor)
# figure(1)
# imshow_torch(input_tensor)
# figure(2)
# imshow_torch(output_tensor1)
# figure(3)
# imshow_torch(output_tensor2)
#
#
# ### Residual Energy: ###
# alpha = 5;
# output_tensor1 = Residual_Energy_Layer()(input_tensor_current,average_tensor)
# output_tensor2 = Residual_Energy_Layer()(input_tensor,average_tensor)
# figure(1)
# imshow_torch(input_tensor*alpha)
# figure(2)
# imshow_torch(output_tensor1*alpha)
# figure(3)
# imshow_torch(output_tensor2*alpha)
#
#
# figure(1)
# imshow_torch(input_tensor)
# figure(2)
# imshow_torch(input_tensor_current)
# figure(3)
# imshow_torch(average_tensor)






####################################################################################################################################################################################################################################################################
############# ConvAlphaEstimatorV2: #############
class ConvAlphaEstimatorV2_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        # ### Special Specific Information Layers: ###
        # Regularized_Inverse_Abs_Layer
        # Regularized_Inverse_Layer
        # Residual_Energy_Layer
        # Energy_Residual_Layer
        # Pseudo_PixelWise_CrossCorrelation_Layer
        # Multiplication_Layer

        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### Special Information Layers: ###
        self.pseudo_cross_correlation_layer = Pseudo_PixelWise_CrossCorrelation_Layer()

        ### GRU Gates: ###

        #(*). TODO: maybe change from deep sequential_conv_block to something else because if i concatenate layers it can become very deep
        self.update_gate = Sequential_Conv_Block(2*input_size+hidden_size,  #TODO: before it was input_size+hidden_size
                              [2*hidden_size,2*hidden_size,1],
                              [kernel_size,kernel_size,kernel_size],
                              strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None, activation_type=['leakyrelu','leakyrelu','none'], mode='CNA', flag_resnet=False, flag_concat=False)

        self.reset_gate = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        # self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)

        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, reset_flags_list=None):
        #TODO: the problem (or loss of generality) here is that hidden_size needs to equal input_size..... this puts some restriction....maybe use projection layers or something else...

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)

        # data size is [batch, channel, height, width]
        if self.flag_input_conv:
            input_ = self.input_conv(input_)


        pseudo_cross_correlation_tensor = self.pseudo_cross_correlation_layer(input_, prev_state)
        stacked_inputs = torch.cat([pseudo_cross_correlation_tensor, input_, prev_state], dim=1)
        update = self.update_gate(stacked_inputs);
        update = torch.sigmoid(update)
        #TODO: maybe instead of Summing the contributions Concatenate them and then use Conv layers: Sequential_Conv_Block([prev_state*(1-update) , input_*update])



        new_state = (1-update)*prev_state + (update)*input_;
        return new_state



class ConvAlphaEstimatorV2(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super(ConvAlphaEstimatorV2, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvAlphaEstimatorV2_Cell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'ConvAlphaEstimatorV2_Cell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list=[2]):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]

            # pass through layer:
            current_cell_output = cell(input_, cell_hidden, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################















####################################################################################################################################################################################################################################################################
############# ConvAlphaEstimatorV3: #############
class ConvAlphaEstimatorV3_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        # ### Special Specific Information Layers: ###
        # Regularized_Inverse_Abs_Layer
        # Regularized_Inverse_Layer
        # Residual_Energy_Layer
        # Energy_Residual_Layer
        # Pseudo_PixelWise_CrossCorrelation_Layer
        # Multiplication_Layer

        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### Special Information Layers: ###
        self.residual_energy_layer = Residual_Energy_Layer()

        ### GRU Gates: ###

        #(*). TODO: maybe change from deep sequential_conv_block to something else because if i concatenate layers it can become very deep
        self.update_gate = Sequential_Conv_Block(3*hidden_size,  #TODO: before it was input_size+hidden_size
                              [2*hidden_size,2*hidden_size,1],
                              [kernel_size,kernel_size,kernel_size],
                              strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None, activation_type=['leakyrelu','leakyrelu','none'], mode='CNA', flag_resnet=False, flag_concat=False)

        self.reset_gate = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        # self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)

        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, reset_flags_list=None):
        #TODO: the problem (or loss of generality) here is that hidden_size needs to equal input_size..... this puts some restriction....maybe use projection layers or something else...

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)

        if self.flag_input_conv:
            input_ = self.input_conv(input_)

        # data size is [batch, channel, height, width]
        residual_energy_tensor = self.residual_energy_layer(input_, prev_state)
        stacked_inputs = torch.cat([residual_energy_tensor, input_, prev_state], dim=1)
        update = self.update_gate(stacked_inputs);
        update = torch.sigmoid(update)
        #TODO: maybe instead of Summing the contributions Concatenate them and then use Conv layers: Sequential_Conv_Block([prev_state*(1-update) , input_*update])



        new_state = (1-update)*prev_state + (update)*input_;
        return new_state



class ConvAlphaEstimatorV3(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super(ConvAlphaEstimatorV3, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvAlphaEstimatorV3_Cell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'ConvAlphaEstimatorV3_Cell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list=[2]):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]

            # pass through layer:
            current_cell_output = cell(input_, cell_hidden, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################



















####################################################################################################################################################################################################################################################################
############# ConvMean2D: #############
class ConvMean2D_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        #TODO: Generalize this block which calculates alpha!...
        #TODO: Check whether adding a GradNorm or GradientPenalty Solution to RNNs improves something
        self.update_gate = Sequential_Conv_Block(input_size+hidden_size,
                              [2*hidden_size,2*hidden_size,1],
                              [kernel_size,kernel_size,kernel_size],
                              strides=1, dilations=1, groups=1, padding_type='reflect', normalization_function=None, activation_function=['leakyrelu','leakyrelu','none'], mode='CNA', flag_resnet=False, flag_concat=False)

        self.T_epsilon = 0.
        self.max_T = 50;

        self.reset_gate = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method,
                                     flag_deformable_convolution=flag_deformable_convolution,
                                     flag_deformable_convolution_version=flag_deformable_convolution_version,
                                     flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation,
                                     flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        # self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)

        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, prev_T_state, reset_flags_list=None):
        #TODO: the problem (or loss of generality) here is that hidden_size needs to equal input_size..... this puts some restriction....maybe use projection layers or something else...

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_T_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_T_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)  #T_state ONLY starts from zero
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
                prev_T_state = prev_T_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)


        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)


        ### Activate update gate (original was sigmoid but i think something like SoftPlus is possibly better because i doesn't saturate above 0 for x>0 but does saturate at 0 for x<0): ###
        ### TODO: add flag which allows the cell to a non learning simple running average cell
        #(1). Use Sigmoid:
        update = self.update_gate(stacked_inputs + 2);  ### TODO: Generalize to be able to choose bias!!!!
        update = torch.sigmoid(update)
        # #(2). Use Softplus:
        # update = self.update_gate(stacked_inputs+1)
        # update = torch.nn.functional.softplus(update)


        prev_T_state = update*prev_T_state;  #Deep running average
        # prev_T_state =  (1 + 0*update)*prev_T_state;  #Simple running average by design
        # prev_T_state = (prev_T_state + 1).clamp(0, self.max_T);  #zero is meaningless here because the minimum is actually 1;
        prev_T_state = abs(prev_T_state + 1)
        alpha_T = 1/prev_T_state;

        if self.flag_input_conv:
            input_ = self.input_conv(input_)

        new_state = prev_state*(1-alpha_T) + input_*alpha_T;
        return new_state, prev_T_state



class ConvMean2D(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super(ConvMean2D, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvMean2D_Cell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'ConvMean2D_Cell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.T_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list=[2]):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]
            T_states = self.T_states[layer_idx]

            # pass through layer:
            current_cell_output, T_states = cell(input_, cell_hidden, T_states, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            self.T_states[layer_idx] = T_states;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################











####################################################################################################################################################################################################################################################################
############# ConvMean2D: #############
class ConvMean2D_PureRunningAverage_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        #TODO: Generalize this block which calculates alpha!...
        #TODO: Check whether adding a GradNorm or GradientPenalty Solution to RNNs improves something
        self.update_gate = Sequential_Conv_Block(input_size+hidden_size,
                              [2*hidden_size,2*hidden_size,1],
                              [kernel_size,kernel_size,kernel_size],
                              strides=1, dilations=1, groups=1, padding_type='reflect', normalization_function=None, activation_function=['leakyrelu','leakyrelu','none'], mode='CNA', flag_resnet=False, flag_concat=False)

        self.T_epsilon = 0.
        self.max_T = 50;

        # self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)

        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, prev_T_state, reset_flags_list=None):
        #TODO: the problem (or loss of generality) here is that hidden_size needs to equal input_size..... this puts some restriction....maybe use projection layers or something else...

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_T_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_T_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)  #T_state ONLY starts from zero
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
                prev_T_state = prev_T_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)


        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        # update = self.update_gate(stacked_inputs+2);  ### TODO: Generalize to be able to choose bias!!!!
        # update = torch.sigmoid(update) ### TODO: add flag which allows the cell to a non learning simple running average cell
        update = 1;

        prev_T_state = update*prev_T_state;  #Deep running average
        # prev_T_state =  (1 + 0*update)*prev_T_state;  #Simple running average by design
        # prev_T_state = (prev_T_state + 1).clamp(0, self.max_T);  #zero is meaningless here because the minimum is actually 1;
        prev_T_state = abs(prev_T_state + 1)
        alpha_T = 1/prev_T_state;

        if self.flag_input_conv:
            input_ = self.input_conv(input_)

        new_state = prev_state*(1-alpha_T) + input_*alpha_T;
        return new_state, prev_T_state



class ConvMean2D_PureRunningAverage(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super(ConvMean2D_PureRunningAverage, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvMean2D_PureRunningAverage_Cell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'ConvMean2D_PureRunningAverage_Cell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.T_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list=[2]):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]
            T_states = self.T_states[layer_idx]

            # pass through layer:
            current_cell_output, T_states = cell(input_, cell_hidden, T_states, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            self.T_states[layer_idx] = T_states;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################














####################################################################################################################################################################################################################################################################
############# ConvMean2DExtra: #############
class ConvMean2DExtra_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined', extra_information_version='v1'):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        #TODO: Generalize this block which calculates alpha!...
        #TODO: Check whether adding a GradNorm or GradientPenalty Solution to RNNs improves something
        self.update_gate = Sequential_Conv_Block(3*hidden_size,
                                                  [2*hidden_size,2*hidden_size,1],
                                                  [kernel_size,kernel_size,kernel_size],
                                                  strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None,
                                                  activation_type=['leakyrelu','leakyrelu','sigmoid'],
                                                  mode='CNA',
                                                  flag_resnet=False, flag_concat=False)

        if extra_information_version == 'v1':
            self.extra_information_layer = nn.Conv2d(input_size,input_size,3,1,1,1,1,10)
        elif extra_information_version == 'v2':
            self.extra_information_layer = Pseudo_PixelWise_CrossCorrelation_Layer()
        elif extra_information_version == 'v3':
            # supposedly an easy solution would be something like - is the residual energy is large output a large negative numebr and after sigmoid it would be zero...
            # however we have leakyrelu's which can perhapse fuck it up.... probably the answer would be playing with biases .....
            self.extra_information_layer = Residual_Energy_Layer()
        elif extra_information_version == 'v4':
            self.extra_information_layer = nn.Sequential(Residual_Energy_Layer(), Regularized_Inverse_Layer())


        self.T_epsilon = 0.
        self.max_T = 50;

        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)

        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2) #TODO: add reflective padding
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, prev_T_state, reset_flags_list=None):
        #TODO: the problem (or loss of generality) here is that hidden_size needs to equal input_size..... this puts some restriction....maybe use projection layers or something else...

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_T_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_T_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)  #T_state ONLY starts from zero
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
                prev_T_state = prev_T_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)

        # NOTE: the input_=self.input_conv(input_) is before the extra information layer because for the extra information layer input_ and prev_state must be exactly the same size
        if self.flag_input_conv:
            input_ = self.input_conv(input_)

        extra_information_tensor = self.extra_information_layer(input_, prev_state)
        stacked_inputs = torch.cat([extra_information_tensor, input_, prev_state], dim=1)

        # data size is [batch, channel, height, width]
        update = self.update_gate(stacked_inputs);
        prev_T_state = update*prev_T_state;  #Deep running average
        # prev_T_state =  (1 + 0*update)*prev_T_state;  #Simple running average by design
        # prev_T_state = (prev_T_state + 1).clamp(0, self.max_T);  #zero is meaningless here because the minimum is actually 1;
        prev_T_state = abs(prev_T_state + 1)
        alpha_T = 1/prev_T_state;



        new_state = prev_state*(1-alpha_T) + input_*alpha_T;
        return new_state, prev_T_state



class ConvMean2DExtra(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined',extra_information_version='v1'):
        super(ConvMean2DExtra, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods, self.extra_information_version = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method, extra_information_version)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvMean2DExtra_Cell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i], self.extra_information_version[i])
            name = 'ConvMean2DExtra_Cell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.T_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list=[2]):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]
            T_states = self.T_states[layer_idx]

            # pass through layer:
            current_cell_output, T_states = cell(input_, cell_hidden, T_states, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            self.T_states[layer_idx] = T_states;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################












#############################################################################################################################################################################################################################
############# ConvSum2D: #############
class ConvSum2D_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        #TODO: Generalize this block which calculates alpha!...
        #TODO: Check whether adding a GradNorm or GradientPenalty Solution to RNNs improves something
        self.update_gate = Sequential_Conv_Block(input_size+hidden_size,
                              [2*hidden_size,2*hidden_size,1],
                              [kernel_size,kernel_size,kernel_size],
                              strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None, activation_type=['leakyrelu','leakyrelu','none'], mode='CNA', flag_resnet=False, flag_concat=False)

        self.T_epsilon = 0.
        self.max_T = 50;

        self.reset_gate = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        # self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)

        self.flag_input_conv = (input_size != hidden_size)
        self.input_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        nn.init.dirac_(self.input_conv.weight)


    def forward(self, input_, prev_state, prev_T_state, reset_flags_list=None):
        #TODO: the problem (or loss of generality) here is that hidden_size needs to equal input_size..... this puts some restriction....maybe use projection layers or something else...

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_T_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_T_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)  #T_state ONLY starts from zero
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
                prev_T_state = prev_T_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)


        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)


        ### Activate update gate (original was sigmoid but i think something like SoftPlus is possibly better because i doesn't saturate above 0 for x>0 but does saturate at 0 for x<0): ###
        ### TODO: add flag which allows the cell to a non learning simple running average cell
        #(1). Use Sigmoid:
        update = self.update_gate(stacked_inputs + 2);  ### TODO: Generalize to be able to choose bias!!!!
        update = torch.sigmoid(update)
        # #(2). Use Softplus:
        # update = self.update_gate(stacked_inputs+1)
        # update = torch.nn.functional.softplus(update)

        if self.flag_input_conv:
            input_ = self.input_conv(input_)

        ### Running Resetable Sum (RRS) (as opposed to runnig resetable average (RRA)): ###
        new_state = prev_state*update + input_

        ### Limit new_state value to avoid explosion: ###
        abs_max_value = 2
        new_state = new_state.clamp(-abs_max_value,abs_max_value)

        ### I think we need a sigmoid here....but maybe if i simply limit the output value than it's enough?: ###
        new_state = torch.tanh(new_state)

        return new_state, prev_T_state



class ConvSum2D(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super(ConvSum2D, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvSum2D_Cell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'ConvSum2D_Cell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.T_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list=[2]):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]
            T_states = self.T_states[layer_idx]

            # pass through layer:
            current_cell_output, T_states = cell(input_, cell_hidden, T_states, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            self.T_states[layer_idx] = T_states;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################






















####################################################################################################################################################################################################################################################################
############# ConvSRU2D: #############
#TODO: impelement!. also, as far as i understand, the differnece between this and FRU (fourier recurrent unit) is the instead of using a few predetermined alpha, we use a few predetermined frequencies
#      HOWEVER, using FRU with indefinite time is not straightforward...one thing i can do is to use CYCLIC time...choose T (cycle time) and t=[0,...,T] because in any case t only comes into play
#      inside a cosine function which is cyclic in T. also maybe i can add an exponental like exp(-alpha*t) with a predetermined alpha to now allow incorporating time from the indefinite past (if that only has my problem)...
class ConvSRU2D_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Decide on alphas: ###
        alpha_vec = [0,0.25,0.5,0.9,0.99];

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        self.update_gate = Sequential_Conv_Block(input_size+hidden_size,
                              [2*hidden_size,2*hidden_size,1],
                              [kernel_size,kernel_size,kernel_size],
                              strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None, activation_type=['leakyrelu','leakyrelu','sigmoid'], mode='CNA', flag_resnet=False, flag_concat=False)

        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)


    def forward(self, input_, prev_state, reset_flags_list=None):
        #TODO: the problem (or loss of generality) here is that hidden_size needs to equal input_size..... this puts some restriction....maybe use projection layers or something else...

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = self.update_gate(stacked_inputs);
        #TODO: maybe instead of Summing the contributions Concatenate them and then use Conv layers: Sequential_Conv_Block([prev_state*(1-update) , input_*update])
        new_state = prev_state*(1-update) + input_*update;
        return new_state



class ConvSRU2D(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super(ConvSRU2D, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvSRU2D_Cell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'ConvSRU2D_Cell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list=[2]):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]

            # pass through layer:
            current_cell_output = cell(input_, cell_hidden, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################














####################################################################################################################################################################################################################################################################
############# ConvGRU2D_AlphaEstimator: #############
class ConvGRU2DCell_SeperateAlphaEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)
        self.final_activation = 'none'
        self.final_activation_function = Activation_Function(self.final_activation)

        ### GRU Gates: ###
        #TODO: Generalize this block which calculates alpha!...
        #TODO: Check whether adding a GradNorm or GradientPenalty Solution to RNNs improves something
        self.update_gate = Sequential_Conv_Block(input_size+hidden_size,
                              [2*hidden_size,2*hidden_size,1],
                              [kernel_size,kernel_size,kernel_size],
                              strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None, activation_type=['relu','relu','sigmoid'], mode='CNA', flag_resnet=False, flag_concat=False)

        self.input_gate = Sequential_Conv_Block(input_size,
                                                 [input_size],
                                                 [kernel_size],
                                                 strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None, activation_type=['none'], mode='CNA', flag_resnet=False, flag_concat=False)

        self.hidden_gate = Sequential_Conv_Block(input_size,
                                                [hidden_size],
                                                [kernel_size],
                                                strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None, activation_type=['none'], mode='CNA', flag_resnet=False, flag_concat=False)


    def forward(self, input_, prev_state, reset_flags_list=None):
        #TODO: the problem (or loss of generality) here is that hidden_size needs to equal input_size..... this puts some restriction....maybe use projection layers or something else...

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = self.update_gate(stacked_inputs)
        prev_state = self.hidden_gate(prev_state);
        input_ = self.input_gate(input_);
        new_state = prev_state*(1-update) + input_*update;
        new_state = self.final_activation_function(new_state)
        return new_state



class ConvGRU2D_SeperateAlphaEstimator(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super(ConvGRU2D_SeperateAlphaEstimator, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvGRU2DCell_SeperateAlphaEstimator(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'ConvGRU2DCell_SeperateAlphaEstimator_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list=[2]):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]

            # pass through layer:
            current_cell_output = cell(input_, cell_hidden, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################









####################################################################################################################################################################################################################################################################
############# ConvGRU2D_AlphaEstimator: #############
class ConvGRU2DCell_GeneralAlphaEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)
        self.final_activation = 'none'
        self.final_activation_function = Activation_Function(self.final_activation)

        ### GRU Gates: ###
        #TODO: Generalize this block which calculates alpha!...
        #TODO: Check whether adding a GradNorm or GradientPenalty Solution to RNNs improves something
        self.update_gate = Sequential_Conv_Block(input_size+hidden_size,
                              [2*hidden_size,2*hidden_size,1],
                              [kernel_size,kernel_size,kernel_size],
                              strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None, activation_type=['leakyrelu','leakyrelu','none'], mode='CNA', initialization_method='xavier',
                                                 flag_resnet=False, flag_concat=False)

        # self.update_gate = nn.Sequential(
        #     ConvOffset2D(input_size + hidden_size), self.Padding, nn.Conv2d(input_size+hidden_size,2*hidden_size,kernel_size), nn.LeakyReLU(),
        #     ConvOffset2D(input_size + hidden_size), self.Padding, nn.Conv2d(2*hidden_size,2*hidden_size,kernel_size), nn.LeakyReLU(),
        #     ConvOffset2D(input_size + hidden_size), self.Padding, nn.Conv2d(2*hidden_size, 1, kernel_size)
        # )

        self.input_gate = Sequential_Conv_Block(input_size + hidden_size,
                                                 [input_size],
                                                 [kernel_size],
                                                 strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None, activation_type=['none'], mode='CNA', initialization_method='dirac',
                                                flag_resnet=False, flag_concat=False)

        self.hidden_gate = Sequential_Conv_Block(input_size + hidden_size,
                                                [hidden_size],
                                                [kernel_size],
                                                strides=1, dilations=1, groups=1, padding_type='reflect', normalization_type=None, activation_type=['none'], mode='CNA', initialization_method='dirac',
                                                 flag_resnet=False, flag_concat=False)


        # self.input_gate = freeze_gradients(self.input_gate)
        # self.hidden_gate = freeze_gradients(self.hidden_gate)


    def forward(self, input_, prev_state, reset_flags_list=None):
        #TODO: the problem (or loss of generality) here is that hidden_size needs to equal input_size..... this puts some restriction....maybe use projection layers or something else...

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.update_gate.parameters().__next__().device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();
            elif reset_flag == 4: #Special Conv
                prev_state = self.initial_conv(input_)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs) + 2)  #TODO: NOTICE THE LARGE BIAS!!!!
        # prev_state = self.hidden_gate(stacked_inputs);
        # input_ = self.input_gate(stacked_inputs);
        new_state = prev_state*(1-update) + input_*update;


        new_state = self.final_activation_function(new_state)
        return new_state



class ConvGRU2D_GeneralAlphaEstimator(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 initialization_method='combined'):
        super(ConvGRU2D_GeneralAlphaEstimator, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.flag_deformable_same_on_all_channels, self.initialization_methods = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels, initialization_method)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvGRU2DCell_GeneralAlphaEstimator(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.flag_deformable_same_on_all_channels[i], self.initialization_methods[i])
            name = 'ConvGRU2DCell_GeneralAlphaEstimator_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        self.reset_flags_list = reset_flags_list;  # Send to Cell to initialize hidden states with input_ instead of zero!!@#$$!@#%
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                #(1). Continue previous stream - just detach hidden states to save on GPU memory:
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();

                #(2). Reset hidden states where necessary (according to reset_flags_list):
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list=[2]):
        input_ = x
        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]

            # pass through layer:
            current_cell_output = cell(input_, cell_hidden, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################





















#############################################################################################################################################################################################################################
class ConvMem2D_Cell(nn.Module):
    #TODO: create a ConvMem2D_Dense module to accomodate possible RDB style Recurrent Networks
    #TODO: maybe wrapp the entire ConvMem2D with a Concate or residual connection from the very input...why not?!
    def __init__(self, input_size, hidden_size,
                 kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 activation_function='relu', initialization_method='dirac',
                 flag_sequential_resnet=False,
                 flag_sequential_concat=False,
                 flag_Sequential_or_RDB='sequential',
                 number_of_intermediate_channels=3):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.flag_Sequential_or_RDB = flag_Sequential_or_RDB;
        self.flag_sequential_concat = flag_sequential_concat;
        self.flag_sequential_resnet = flag_sequential_resnet
        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)
        self.number_of_intermediate_channels = number_of_intermediate_channels
        ### Parameter Or Conv To Initialize Hidden State: ###

        if number_of_intermediate_channels!=0:
            number_of_intermediate_channels = [number_of_intermediate_channels, hidden_size]
        else:
            number_of_intermediate_channels = [hidden_size]

        if flag_sequential_resnet:
            number_of_intermediate_channels[-1] = input_size + hidden_size; #(*). To allow residual connection without projection block i need number of channels to be the same

        if flag_Sequential_or_RDB == 'sequential':
            #TODO: can't really use a residual block here because input is (input_size+hidden_size) and output is supposed to be hidden_size.... so maybe seperate convolutions but that way it's less expressive....
            #TODO: maybe Wrap cell with a Concat to output [new_state,input] or something like that
            self.initial_conv = Sequential_Conv_Block(number_of_input_channels=input_size + hidden_size,
                                               number_of_output_channels=number_of_intermediate_channels,
                                               kernel_sizes=kernel_size,
                                               strides=stride,
                                               dilations=dilation,
                                               groups=groups,
                                               padding_type='reflect',
                                               normalization_type=normalization_type,
                                               activation_type=activation_function,
                                               mode='CNA',
                                               initialization_method=initialization_method,
                                               flag_resnet=flag_sequential_resnet,
                                               flag_concat=flag_sequential_concat)
        elif flag_Sequential_or_RDB == 'rdb':
            #Problematic for now because RDB accepts (input_size+hidden_size) channels but it's a residual block and therefore also OUTPUTS (input_size+hidden_size)...but it should output (hidden_size).
            #So either we take only the first hidden_size channels or something else....
            number_of_layers = 2;
            self.initial_conv = RDB(number_of_input_channels=input_size + hidden_size,
                             number_of_output_channels_for_each_conv_block=self.number_of_intermediate_channels,
                             number_of_conv_blocks=number_of_layers,
                             kernel_size=kernel_size, stride=1, bias=True, padding_type='reflect', normalization_type=None, activation_type='leakyrelu', mode='CNA',
                             final_residual_branch_scale_factor=1/number_of_layers)



    def forward(self, input_, prev_state, reset_flags_list=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0: #Do Nothing
                1;
            elif reset_flag == 1: #Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.initial_conv.parameters().__next__().device)
            elif reset_flag == 2:  #Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.initial_conv.parameters().__next__().device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();


        # Data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        new_state = self.initial_conv(stacked_inputs)

        ### TODO: think maybe of something else...maybe another conv or something: ###
        if self.flag_Sequential_or_RDB or self.flag_resnet or self.flag_sequential_concat:
            new_state = new_state[:,0:self.hidden_size,:,:]

        return new_state



class ConvMem2D(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers,
                 kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 activation_function = 'leakyrelu',
                 initialization_method='dirac',
                 flag_sequential_resnet=False,
                 flag_sequential_concat=False,
                 flag_Sequential_or_RDB='sequential',
                 number_of_intermediate_channels=3):
        super(ConvMem2D, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.activation_functions, self.number_of_intermediate_channels, self.initialization_method, self.flag_sequential_resnet, self.flag_sequential_concat, self.flag_Sequential_or_RDB = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, activation_function, number_of_intermediate_channels, initialization_method, flag_sequential_resnet, flag_sequential_concat, flag_Sequential_or_RDB)
        #############################

        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvMem2D_Cell(input_size=input_dim,
                                  hidden_size=self.hidden_sizes[i],
                                  kernel_size=self.kernel_sizes[i], stride=self.strides[i], dilation=self.dilations[i], groups=self.groups[i], normalization_type=None,
                                  flag_deformable_convolution=self.flag_deformable_convolution[i],
                                  activation_function=self.activation_functions[i],
                                  initialization_method=self.initialization_method[i],
                                  flag_sequential_resnet=self.flag_sequential_resnet[i],
                                  flag_sequential_concat=self.flag_sequential_concat[i],
                                  flag_Sequential_or_RDB=self.flag_Sequential_or_RDB[i],
                                  number_of_intermediate_channels=self.number_of_intermediate_channels[i])

            name = 'ConvMem2D_Cell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.reset_flags_list = False;
        self.identity = Identity_Layer();
    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;

    def reset_or_detach_hidden_states(self, reset_flags_list=[2]):
        self.reset_flags_list = reset_flags_list;
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;


    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list):
        input_ = self.identity(x)

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]

            # pass through layer:
            current_cell_output = cell(input_, cell_hidden, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################











#############################################################################################################################################################################################################################
class ConvIRNN2D_Cell(nn.Module):
    #TODO: create a ConvMem2D_Dense module to accomodate possible RDB style Recurrent Networks
    #TODO: maybe wrapp the entire ConvMem2D with a Concate or residual connection from the very input...why not?!
    def __init__(self, input_size, hidden_size,
                 kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 activation_function='relu',
                 initialization_method='dirac',
                 flag_sequential_resnet=False,
                 flag_sequential_concat=False,
                 flag_Sequential_or_RDB='sequential',
                 number_of_intermediate_channels=3):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_function = activation_function

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        self.Activation = Activation_Function(activation_function)


        ### Parameter Or Conv To Initialize Hidden State: ###
        if number_of_intermediate_channels!=0:
            number_of_intermediate_channels = [number_of_intermediate_channels, hidden_size]
        else:
            number_of_intermediate_channels = hidden_size
        if flag_Sequential_or_RDB == 'sequential':
            #TODO: can't really use a residual block here because input is (input_size+hidden_size) and output is supposed to be hidden_size.... so maybe seperate convolutions but that way it's less expressive....
            #TODO: maybe Wrap cell with a Concat to output [new_state,input] or something like that
            self.input_conv = Sequential_Conv_Block(number_of_input_channels=input_size,
                                               number_of_output_channels=number_of_intermediate_channels,
                                               kernel_sizes=kernel_size,
                                               strides=stride,
                                               dilations=dilation,
                                               groups=groups,
                                               padding_type='reflect',
                                               normalization_type=normalization_type,
                                               activation_type=None,
                                               mode='CNA',
                                               initialization_method='xavier',
                                               flag_resnet=flag_sequential_resnet,
                                               flag_concat=flag_sequential_concat)
            self.hidden_conv = Sequential_Conv_Block(number_of_input_channels=hidden_size,
                                                    number_of_output_channels=number_of_intermediate_channels,
                                                    kernel_sizes=kernel_size,
                                                    strides=stride,
                                                    dilations=dilation,
                                                    groups=groups,
                                                    padding_type='reflect',
                                                    normalization_type=normalization_type,
                                                    activation_type=None,
                                                    mode='CNA',
                                                    initialization_method='dirac',
                                                    flag_resnet=flag_sequential_resnet,
                                                    flag_concat=flag_sequential_concat)
        elif flag_Sequential_or_RDB == 'rdb':
            #Problematic for now because RDB accepts (input_size+hidden_size) channels but it's a residual block and therefore also OUTPUTS (input_size+hidden_size)...but it should output (hidden_size).
            #So either we take only the first hidden_size channels or something else....
            number_of_layers = len(kernel_size);
            self.input_conv = RDB(number_of_input_channels=input_size + hidden_size,
                                  number_of_output_channels_for_each_conv_block=number_of_intermediate_channels,
                                  number_of_conv_blocks=number_of_layers,
                                  kernel_size=kernel_size, stride=1, bias=True, padding_type='reflect', normalization_type=None, activation_type='leakyrelu', mode='CNA',
                                  final_residual_branch_scale_factor=1/number_of_layers)
            self.hidden_conv = RDB(number_of_input_channels=input_size + hidden_size,
                                   number_of_output_channels_for_each_conv_block=number_of_intermediate_channels,
                                   number_of_conv_blocks=number_of_layers,
                                   kernel_size=kernel_size, stride=1, bias=True, padding_type='reflect', normalization_type=None, activation_type='leakyrelu', mode='CNA',
                                   final_residual_branch_scale_factor=1 / number_of_layers)



    def forward(self, input_, prev_state, reset_flags_list=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0: #Do Nothing
                1;
            elif reset_flag == 1: #Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.input_conv.parameters().__next__().device)
            elif reset_flag == 2:  #Initialize with input_
                prev_state = Variable(torch.zeros(state_size)).to(self.input_conv.parameters().__next__().device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();


        # Data size is [batch, channel, height, width]
        new_state = self.Activation( self.input_conv(input_) + self.hidden_conv(prev_state) )
        return new_state



class ConvIRNN2D(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_layers,
                 kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 activation_function = 'leakyrelu',
                 initialization_method='dirac',
                 flag_sequential_resnet=False,
                 flag_sequential_concat=False,
                 flag_Sequential_or_RDB='sequential',
                 number_of_intermediate_channels=3):
        super(ConvIRNN2D, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation, self.activation_functions, self.number_of_intermediate_channels, self.initialization_method, self.flag_sequential_resnet, self.flag_sequential_concat, self.flag_Sequential_or_RDB = \
            make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups, flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution, flag_deformable_convolution_modulation, activation_function, number_of_intermediate_channels, initialization_method, flag_sequential_resnet, flag_sequential_concat, flag_Sequential_or_RDB)
        #############################

        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvIRNN2D_Cell(input_size=input_dim,
                                  hidden_size=self.hidden_sizes[i],
                                  kernel_size=self.kernel_sizes[i], stride=self.strides[i], dilation=self.dilations[i], groups=self.groups[i], normalization_type=None,
                                  flag_deformable_convolution=self.flag_deformable_convolution[i],
                                  activation_function=self.activation_functions[i],
                                  initialization_method=self.initialization_method[i],
                                  flag_sequential_resnet=self.flag_sequential_resnet[i],
                                  flag_sequential_concat=self.flag_sequential_concat[i],
                                  flag_Sequential_or_RDB=self.flag_Sequential_or_RDB[i],
                                  number_of_intermediate_channels=self.number_of_intermediate_channels[i])

            name = 'ConvMem2D_Cell' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################

        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.reset_flags_list = False;
        self.identity = Identity_Layer();
    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;

    def reset_or_detach_hidden_states(self, reset_flags_list=[2]):
        self.reset_flags_list = reset_flags_list;
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)

        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;


    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list):
        input_ = self.identity(x)

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]

            # pass through layer:
            current_cell_output = cell(input_, cell_hidden, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################














#############################################################################################################################################################################################################################
class ConvReLU2D_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 final_activation_function='None', initialization_method='dirac'):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        ### Get Reflection Padding Size: ###
        padding_size = get_valid_padding(kernel_size,dilation)
        self.Padding = nn.ReflectionPad2d(padding_size)

        ### GRU Gates: ###
        self.input_gate = Conv_Block(input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        self.combined_gate1 = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        self.combined_gate2 = Conv_Block(input_size + hidden_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',  # don't do padding .... this means the output size will be different from input size right?
                                     normalization_function=None, activation_function=None, mode='CNA', initialization_method=initialization_method, flag_deformable_convolution=flag_deformable_convolution, flag_deformable_convolution_version=flag_deformable_convolution_version, flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,  # 'before' / 'after'
                                     flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels, bias_value=None)
        # self.input_gate = nn.Conv2d(input_size, hidden_size, kernel_size, padding=0)
        # self.combined_gate1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=0)
        # self.combined_gate2 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=0)

        ### Parameter Or Conv To Initialize Hidden State: ###
        self.initial_conv = nn.Conv2d(input_size, hidden_size, kernel_size, padding=0)

        self.final_activation_function = final_activation_function


    def forward(self, input_, prev_state, reset_flags_list=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # Initialize Hidden States by Looping over reset_flags_list and where you need to "reset" initialize with current _input:
        for batch_example_index, reset_flag in enumerate(reset_flags_list):
            if reset_flag == 0:  # Do Nothing
                1;
            elif reset_flag == 1:  # Initialize with zeros
                prev_state = Variable(torch.zeros(state_size)).to(self.combined_gate1.parameters().__next__().device)
            elif reset_flag == 2:  # Initialize with input_
                # Assuming channels(prev_state) > channels(input_)
                prev_state = Variable(torch.zeros(state_size)).to(self.combined_gate1.parameters().__next__().device)
                prev_state[batch_example_index, 0:self.input_size, :, :].copy_(input_[batch_example_index, :, :, :])
            elif reset_flag == 3: #Detach:
                prev_state = prev_state.detach();

        # data size is [batch, channel, height, width]
        input_ = F.leaky_relu(self.input_gate(self.Padding(input_)));
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        stacked_inputs = F.leaky_relu(self.combined_gate1(self.Padding(stacked_inputs)));
        stacked_inputs = self.combined_gate2(self.Padding(stacked_inputs));

        if self.final_activation_function=='None':
            new_state = stacked_inputs;
        else:
            #Generalize to different activation functions
            new_state = F.leaky_relu(stacked_inputs);

        return new_state



class ConvReLU2D(nn.Module):  #the same as the recurrent autoencoder paper
    def __init__(self, input_size, hidden_sizes, n_layers, kernel_sizes, strides=1, dilations=1, groups=1, normalization_type=None,
                 flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 flag_deformable_same_on_all_channels=False,
                 final_activation_function='leaky_relu', initialization_method='dirac'):
        super(ConvReLU2D, self).__init__()

        self.input_size = input_size
        self.n_layers = n_layers
        self.final_activation_function = final_activation_function

        ##############################
        ### Check Inputs Validity: ###
        self.hidden_sizes, self.kernel_sizes, self.strides, self.dilations, self.groups, self.flag_deformable_convolution, self.flag_deformable_convolution_version, self.flag_deformable_convolution_before_or_after_main_convolution, self.flag_deformable_convolution_modulation,\
                                                                                                                            = make_list_of_certain_size(n_layers, hidden_sizes, kernel_sizes, strides, dilations, groups,
                                                                                                                                                       flag_deformable_convolution,
                                                                                                                                                       flag_deformable_convolution_version,
                                                                                                                                                       flag_deformable_convolution_before_or_after_main_convolution,
                                                                                                                                                       flag_deformable_convolution_modulation)
        #############################


        #############################
        ### Build Cells: ###
        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]
            cell = ConvReLU2D_Cell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i], self.strides[i], self.dilations[i], self.groups[i], normalization_type, self.flag_deformable_convolution[i],self.flag_deformable_convolution_version[i], self.flag_deformable_convolution_before_or_after_main_convolution[i], self.flag_deformable_convolution_modulation[i], self.final_activation_function[i])
            name = 'ConvGRU2DCell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))
        #############################


        self.cells = cells
        self.hidden_states = [None]*self.n_layers;
        self.reset_flags_list = False;

    def reset_hidden_states(self):
        self.hidden_states = [None]*self.n_layers;


    def reset_or_detach_hidden_states(self, reset_flags_list):
        if type(self.hidden_states[0]) == torch.Tensor:
            for hidden_state_index in arange(len(self.hidden_states)):
                self.hidden_states[hidden_state_index] = self.hidden_states[hidden_state_index].detach();
                for batch_example_index, reset_flag in enumerate(reset_flags_list):
                    if reset_flag:
                        current_hidden_state_shape = self.hidden_states[hidden_state_index].shape;
                        # current_hidden_state_shape[0] = 1;
                        current_hidden_state_device = self.hidden_states[hidden_state_index].device;
                        self.hidden_states[hidden_state_index][batch_example_index,:,:,:] = Variable(torch.zeros(current_hidden_state_shape[1:])).to(current_hidden_state_device)
                        self.reset_flags_list = reset_flags_list;
        else:
            #Creating hidden states for the first time:
            self.hidden_states = [None] * self.n_layers;



    def hidden_states_to_device(self,device):
        for hidden_state in self.hidden_states:
            if type(hidden_state) == torch.Tensor:
               hidden_state = hidden_state.to(device);


    def forward(self, x, reset_flags_list):
        input_ = x

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = self.hidden_states[layer_idx]

            # pass through layer:
            current_cell_output = cell(input_, cell_hidden, reset_flags_list)
            # update current layer hidden state list:
            self.hidden_states[layer_idx] = current_cell_output;
            # update input_ to the last updated hidden layer for next pass:
            input_ = current_cell_output

        return input_
#############################################################################################################################################################################################################################






#####################################################################################################################################################################################################################################
class Kalman_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride=1, dilation=1, groups=1, normalization_type=None, flag_deformable_convolution=False,
                 flag_deformable_convolution_version='v1',
                 flag_deformable_convolution_before_or_after_main_convolution='before',
                 flag_deformable_convolution_modulation=False,
                 initialization_method='dirac'):
        super().__init__()

        self.n_layers = 3;
        self.input_size = input_size
        self.hidden_size = hidden_size;
        self.kernel_size = kernel_size;

        # Transition LSTM"
        self.Transition_LSTM = ConvLSTM2D(input_size=3, hidden_sizes=[10, 3], n_layers=2, kernel_sizes=5, strides=1, dilations=1, groups=1, normalization_type=None, flag_deformable_convolution=False, initialization_method='xavier', flag_cell_residual=0, flag_stack_residual=0, flag_gradient_highway_type=0, flag_gradient_highway_strategy=0, flag_gradient_highway_none_only_first_or_all=0)
        # Q LSTM:
        self.Q_LSTM = ConvLSTM2D(input_size=3, hidden_sizes=[10, 3], n_layers=2, kernel_sizes=5, strides=1, dilations=1, groups=1, normalization_type=None, flag_deformable_convolution=False, initialization_method='xavier', flag_cell_residual=0, flag_stack_residual=0, flag_gradient_highway_type=0, flag_gradient_highway_strategy=0, flag_gradient_highway_none_only_first_or_all=0)
        # R LSTM"
        self.R_LSTM = ConvLSTM2D(input_size=3, hidden_sizes=[10, 3], n_layers=2, kernel_sizes=5, strides=1, dilations=1, groups=1, normalization_type=None, flag_deformable_convolution=False, initialization_method='xavier', flag_cell_residual=0, flag_stack_residual=0, flag_gradient_highway_type=0, flag_gradient_highway_strategy=0, flag_gradient_highway_none_only_first_or_all=0)
        # F LSTM:
        self.F_LSTM = ConvLSTM2D(input_size=3, hidden_sizes=[10, 3], n_layers=2, kernel_sizes=5, strides=1, dilations=1, groups=1, normalization_type=None, flag_deformable_convolution=False, initialization_method='xavier', flag_cell_residual=0, flag_stack_residual=0, flag_gradient_highway_type=0, flag_gradient_highway_strategy=0, flag_gradient_highway_none_only_first_or_all=0)

        self.enforce_positivity = True;
        self.flag_clip_K = True;

    def hidden_states_to_device(self,device):
        self.Transition_LSTM.hidden_states_to_device(device);
        self.Q_LSTM.hidden_states_to_device(device);
        self.R_LSTM.hidden_states_to_device(device);
        self.F_LSTM.hidden_states_to_device(device);

    def reset_hidden_states(self):
        self.Transition_LSTM.reset_hidden_states();
        self.Q_LSTM.reset_hidden_states();
        self.R_LSTM.reset_hidden_states();
        self.F_LSTM.reset_hidden_states();


    def forward(self, z, reset_flags_list):

        if reset_flags_list[0] == 1 or reset_flags_list[0] == 2:
            self._x = z
            self._P = Variable(torch.Tensor(torch.randn(z.shape))).to(z.device)  # there should be some initialiation of P ...
        elif reset_flags_list[0] == 3:
            self._x = self._x.detach()
            self._P = self._P.detach()
            self._y = self._y.detach()

        self._P = torch.clamp(self._P, 0)


        ### Apriori X: ###
        self._x = self.Transition_LSTM(self._x,reset_flags_list)  # Notice!: the self._x inputed into the following Q_LSTM, F_LSTM are AFTER the Transition_LSTM(self._x) !!!!
        ### Q Mat: ###
        Q = self.Q_LSTM(self._x,reset_flags_list)
        ### F Mat: ###
        F = self.F_LSTM(self._x,reset_flags_list);
        ### R Mat: ###
        R = self.R_LSTM(z,reset_flags_list);

        ### Enforce Positivity For Q,R Matrices: ###
        if self.enforce_positivity:
            Q = torch.clamp(Q, 0)  # WAIT!: why not simply ReLU at the end of the LSTMs?!?!?!?!
            R = torch.clamp(R, 0)
            F = torch.clamp(F, 0)

        ### Predict Step: ###
        P = self._P
        P = F * P * F + Q  # P_new = F*P_old*F' + Q.    #TODO: maybe simply use a convolution or something, maybe with relu, to build P?????
        if self.enforce_positivity:
            P = torch.clamp(P, 0)

        ### Update Step: ###
        self._y = z - self._x  # Error = z-x

        ### Kalman Gain: ###
        S = P + R
        K = P / (S+0.1);  # probably should add bias here! (or maybe just use a convolution with sigmoid?) (maybe clip to between [0,1]?!!!?!?!) (maybe enforce positivity on P,R,Q matrices?!?!?!?!)
        if self.flag_clip_K:
            K = torch.clamp(K, 0, 1);

        ### Update x_estimate with Kalman gain of error: ###
        self._x = self._x + K * self._y;

        ### Update P Mat: ###
        I_KH = 1 - K
        self._P = I_KH * P * I_KH + K * R * K

        return self._x

#####################################################################################################################################################################################################################################






def ConvLSTM2D_According_To_Flags_String(input_size=3,
                                         hidden_sizes=[30,30,3],
                                         n_layers=3,
                                         kernel_sizes=5,
                                         strides=1,dilations=1,groups=1,
                                         flag_deformable_convolution=False,
                                         flag_deformable_convolution_version='v1',
                                         flag_deformable_convolution_before_or_after_main_convolution='before',
                                         flag_deformable_convolution_modulation=False,
                                         flag_deformable_same_on_all_channels=False,
                                         initialization_method='xavier',
                                         normalization_type = 'none',
                                         flags_string='112010'):
    ### Good Options: ###
    # '112010'
    # '112010' \
    # '110020' \
    # '110120' \
    # '112020' \

    flag_cell_residual, flag_stack_residual, flag_gradient_highway_non_only_first_or_all, flag_gradient_highway_type, flag_gradient_highway_strategy, flag_ConvLSTMCell_type = list(flags_string)
    flag_cell_residual = int(flag_cell_residual)
    flag_stack_residual = int(flag_stack_residual)
    flag_gradient_highway_non_only_first_or_all = int(flag_gradient_highway_non_only_first_or_all)
    flag_gradient_highway_type = int(flag_gradient_highway_type)
    flag_gradient_highway_strategy = int(flag_gradient_highway_strategy)
    flag_ConvLSTMCell_type = int(flag_ConvLSTMCell_type)
    #TODO: insert the rest of the deformable convolution flags into the ConvLSTM2D !!@@%^$
    LSTM_Network = ConvLSTM2D(input_size=input_size,
                                   hidden_sizes=hidden_sizes,
                                   n_layers=n_layers,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides, dilations=dilations, groups=groups,
                                   normalization_type=normalization_type,
                                   flag_deformable_convolution=flag_deformable_convolution,
                                   initialization_method=initialization_method,
                                   flag_cell_residual=flag_cell_residual,
                                   flag_stack_residual=flag_stack_residual,
                                   flag_gradient_highway_type=flag_gradient_highway_type,
                                   flag_gradient_highway_strategy=flag_gradient_highway_strategy,
                                   flag_gradient_highway_none_only_first_or_all=flag_gradient_highway_non_only_first_or_all,
                                   flag_ConvLSTMCell_type=flag_ConvLSTMCell_type)


    return LSTM_Network












def Get_Memory_Unit(memory_unit_name='gru',
                    input_size = 3,
                    hidden_sizes= [3],
                    n_layers=1,
                    kernel_sizes=3,
                    strides=1,
                    dilations=1,
                    groups=1, normalization_function='none',
                    activation_function = 'none',
                    initialization_method = 'xavier',
                    flag_deformable_convolution=False,
                    flag_deformable_convolution_version='v1',
                    flag_deformable_convolution_before_or_after_main_convolution='before',
                    flag_deformable_convolution_modulation=False,
                    flag_deformable_same_on_all_channels=False,):


    if str.lower(memory_unit_name) == 'gru':
        Memory_Unit = ConvGRU2D(input_size=input_size,
                                hidden_sizes=hidden_sizes,
                                n_layers=n_layers,
                                kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                normalization_type=normalization_function,
                                flag_deformable_convolution=flag_deformable_convolution,
                                flag_deformable_convolution_version=flag_deformable_convolution_version,
                                flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                initialization_method=initialization_method)

    if str.lower(memory_unit_name) == 'lstm':
        Memory_Unit = ConvLSTM2D(input_size=input_size,
                                hidden_sizes=hidden_sizes,
                                n_layers=n_layers,
                                kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                normalization_type=normalization_function,
                                flag_deformable_convolution=flag_deformable_convolution,
                                flag_deformable_convolution_version=flag_deformable_convolution_version,
                                flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                initialization_method=initialization_method)

    if str.lower(memory_unit_name) == 'convmem2d':
        Memory_Unit = ConvMem2D(input_size=input_size,
                                hidden_sizes=hidden_sizes,
                                n_layers=n_layers,
                                kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                normalization_type=normalization_function,
                                flag_deformable_convolution=flag_deformable_convolution,
                                flag_deformable_convolution_version=flag_deformable_convolution_version,
                                flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                initialization_method=initialization_method)

    if str.lower(memory_unit_name) == 'alpha_estimator':
        Memory_Unit = ConvGRU2D_AlphaEstimator(input_size=input_size,
                                hidden_sizes=hidden_sizes,
                                n_layers=n_layers,
                                kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                normalization_type=normalization_function,
                                flag_deformable_convolution=flag_deformable_convolution,
                                flag_deformable_convolution_version=flag_deformable_convolution_version,
                                flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                initialization_method=initialization_method)

    if str.lower(memory_unit_name) == 'alpha_estimator_v2':
        Memory_Unit = ConvAlphaEstimatorV2(input_size=input_size,
                                hidden_sizes=hidden_sizes,
                                n_layers=n_layers,
                                kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                normalization_type=normalization_function,
                                flag_deformable_convolution=flag_deformable_convolution,
                                flag_deformable_convolution_version=flag_deformable_convolution_version,
                                flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                initialization_method=initialization_method)

    if str.lower(memory_unit_name) == 'alpha_estimator_v3':
        Memory_Unit = ConvAlphaEstimatorV3(input_size=input_size,
                                hidden_sizes=hidden_sizes,
                                n_layers=n_layers,
                                kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                normalization_type=normalization_function,
                                flag_deformable_convolution=flag_deformable_convolution,
                                flag_deformable_convolution_version=flag_deformable_convolution_version,
                                flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                initialization_method=initialization_method)

    if str.lower(memory_unit_name) == 'general_alpha_estimator':
        Memory_Unit = ConvGRU2D_GeneralAlphaEstimator(input_size=input_size,
                                               hidden_sizes=hidden_sizes,
                                               n_layers=n_layers,
                                               kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                               normalization_type=normalization_function,
                                               flag_deformable_convolution=flag_deformable_convolution,
                                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                               initialization_method=initialization_method)

    if str.lower(memory_unit_name) == 'predrnn':
        Memory_Unit = PredRNN(input_size=input_size,
                                               hidden_sizes=hidden_sizes,
                                               n_layers=n_layers,
                                               kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                               normalization_type=normalization_function,
                                               flag_deformable_convolution=flag_deformable_convolution,
                                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                               initialization_method=initialization_method)

    if str.lower(memory_unit_name) == 'predrnnpp':
        Memory_Unit = PredRNNPP(input_size=input_size,
                                               hidden_sizes=hidden_sizes,
                                               n_layers=n_layers,
                                               kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                               normalization_type=normalization_function,
                                               flag_deformable_convolution=flag_deformable_convolution,
                                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                               initialization_method=initialization_method)

    if str.lower(memory_unit_name) == 'convmean2d':
        Memory_Unit = ConvMean2D(input_size=input_size,
                                               hidden_sizes=hidden_sizes,
                                               n_layers=n_layers,
                                               kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                               normalization_type=normalization_function,
                                               flag_deformable_convolution=flag_deformable_convolution,
                                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                               initialization_method=initialization_method)

    if str.lower(memory_unit_name) == 'convmean2dextra_v1':
        Memory_Unit = ConvMean2DExtra(input_size=input_size,
                                               hidden_sizes=hidden_sizes,
                                               n_layers=n_layers,
                                               kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                               normalization_type=normalization_function,
                                               flag_deformable_convolution=flag_deformable_convolution,
                                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                               initialization_method=initialization_method, extra_information_version='v1')

    if str.lower(memory_unit_name) == 'convmean2dextra_v2':
        Memory_Unit = ConvMean2DExtra(input_size=input_size,
                                               hidden_sizes=hidden_sizes,
                                               n_layers=n_layers,
                                               kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                               normalization_type=normalization_function,
                                               flag_deformable_convolution=flag_deformable_convolution,
                                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                               initialization_method=initialization_method, extra_information_version='v2')

    if str.lower(memory_unit_name) == 'convmean2dextra_v3':
        Memory_Unit = ConvMean2DExtra(input_size=input_size,
                                               hidden_sizes=hidden_sizes,
                                               n_layers=n_layers,
                                               kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                               normalization_type=normalization_function,
                                               flag_deformable_convolution=flag_deformable_convolution,
                                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                               initialization_method=initialization_method, extra_information_version='v3')

    if str.lower(memory_unit_name) == 'convmean2dextra_v4':
        Memory_Unit = ConvMean2DExtra(input_size=input_size,
                                               hidden_sizes=hidden_sizes,
                                               n_layers=n_layers,
                                               kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                               normalization_type=normalization_function,
                                               flag_deformable_convolution=flag_deformable_convolution,
                                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                               initialization_method=initialization_method, extra_information_version='v4')

    if str.lower(memory_unit_name) == 'convmean2d_running_average':
        Memory_Unit = ConvMean2D_PureRunningAverage(input_size=input_size,
                                               hidden_sizes=hidden_sizes,
                                               n_layers=n_layers,
                                               kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                                               normalization_type=normalization_function,
                                               flag_deformable_convolution=flag_deformable_convolution,
                                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                               initialization_method=initialization_method)


    if str.lower(memory_unit_name) == 'convsum2d':
        Memory_Unit = ConvSum2D(input_size=input_size,
                               hidden_sizes=hidden_sizes,
                               n_layers=n_layers,
                               kernel_sizes=kernel_sizes, strides=strides, dilations=dilations, groups=groups,
                               normalization_type=normalization_function,
                               flag_deformable_convolution=flag_deformable_convolution,
                               flag_deformable_convolution_version=flag_deformable_convolution_version,
                               flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                               flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                               initialization_method=initialization_method)


    if str.lower(memory_unit_name) == 'lstm_flags':
        Memory_Unit = ConvLSTM2D_According_To_Flags_String(input_size=input_size,
                                                           hidden_sizes=hidden_sizes,
                                                           n_layers=n_layers,
                                                           kernel_sizes=kernel_sizes,
                                                           strides=strides, dilations=dilations, groups=groups,
                                                           flag_deformable_convolution=flag_deformable_convolution,
                                                           flag_deformable_convolution_version=flag_deformable_convolution_version,
                                                           flag_deformable_convolution_before_or_after_main_convolution=flag_deformable_convolution_before_or_after_main_convolution,
                                                           flag_deformable_convolution_modulation=flag_deformable_convolution_modulation, flag_deformable_same_on_all_channels=flag_deformable_same_on_all_channels,
                                                           initialization_method='initialization_method',
                                                           normalization_type=normalization_function,
                                                           flags_string='112010')


    # print(str.lower(memory_unit_name))

    return Memory_Unit








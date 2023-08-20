import RapidBase.Basic_Import_Libs
from RapidBase.Basic_Import_Libs import *


# #(5). Layers:
# import RapidBase.MISC_REPOS.Layers.Activations
# import RapidBase.MISC_REPOS.Layers.Basic_Layers
# import RapidBase.MISC_REPOS.Layers.Conv_Blocks
# import RapidBase.MISC_REPOS.Layers.DownSample_UpSample
# import RapidBase.MISC_REPOS.Layers.Memory_and_RNN
# import RapidBase.MISC_REPOS.Layers.Refinement_Modules
# import RapidBase.MISC_REPOS.Layers.Special_Layers
# import RapidBase.MISC_REPOS.Layers.Unet_UpDown_Blocks
# import RapidBase.MISC_REPOS.Layers.Warp_Layers
# # import RapidBase.MISC_REPOS.Layers.Wrappers
# from RapidBase.MISC_REPOS.Layers.Activations import *
# from RapidBase.MISC_REPOS.Layers.Basic_Layers import *
# from RapidBase.MISC_REPOS.Layers.Conv_Blocks import *
# from RapidBase.MISC_REPOS.Layers.DownSample_UpSample import *
# from RapidBase.MISC_REPOS.Layers.Memory_and_RNN import *
# from RapidBase.MISC_REPOS.Layers.Refinement_Modules import *
# from RapidBase.MISC_REPOS.Layers.Special_Layers import *
# from RapidBase.MISC_REPOS.Layers.Unet_UpDown_Blocks import *
# from RapidBase.MISC_REPOS.Layers.Warp_Layers import *
# # from RapidBase.MISC_REPOS.Layers.Wrappers import *







############################################################################################################################################################################################################################################################
####################
# Special/Custom Layers:
####################
#TODO: see how these are saved in pytorch to disk!
#(1). Custom Layers Skelaton/Examples:
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)

#(*). Note: an example of a stupid custom layer
class MyModel(nn.Module):
    def forward(self, input):
        return input ** 2 + 1

class Identity_Layer(nn.Module):
    def __init__(self, *args):
        super(Identity_Layer, self).__init__()
    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self, *args):
        super(Flatten, self).__init__()
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Concat_Block(nn.Module):
    def __init__(self, module_list):
        super(Concat_Block, self).__init__()
        self.module_list = module_list
    def forward(self, x):
        output = [];
        for i, module in enumerate(self.module_list):
            output.append(module(x));
        return torch.cat(output,1)

# outputs = [branch3x3, branch3x3dbl, branch_pool]
#         return torch.cat(outputs, 1)

#(2). Lambda Layers:
class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))
########################################################################################################################################################################################################################################################################################





# class Input_Output_Concat_Block(nn.Module):
#     # Initialize this with a module
#     def __init__(self, submodule):
#         super(Input_Output_Concat_Block, self).__init__()
#         self.sub = submodule
#
#         self.flag_first = False;
#     # Feed the module an input and output the module output concatenated with the input itself.
#     def forward(self, x, y_cell_conditional=None, y_deformable_conditional=None):
#         if self.flag_first == False:
#             self.sub = self.sub.to(x.device)
#             self.flag_first = True;
#
#         ### To make this wrapper compatible with the most general configuration which includes modules receiving y_cell_conditional and y_deformable_conditional, check what parameters sub_module accepts: ###
#         if 'y_cell_conditional' in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' in signature(self.sub.forward).parameters.keys():
#             output = torch.cat([x, self.sub(x, y_cell_conditional, y_deformable_conditional)], dim=1)
#         elif 'y_cell_conditional' in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' not in signature(self.sub.forward).parameters.keys():
#             output = torch.cat([x, self.sub(x, y_cell_conditional)], dim=1)
#         elif 'y_cell_conditional' not in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' in signature(self.sub.forward).parameters.keys():
#             output = torch.cat([x, self.sub(x, y_deformable_conditional)], dim=1)
#         else:
#             output = torch.cat([x, self.sub(x)], dim=1)
#         ########################################################################################################################################################################################################
#
#         return output
#
#     def __repr__(self):
#         tmpstr = 'Input_Output_Concat .. \n|'
#         modstr = self.sub.__repr__().replace('\n', '\n|')
#         tmpstr = tmpstr + modstr
#         return tmpstr
#
#
# class Input_Output_Sum_Block(nn.Module):
#     # Initialize this with a module
#     def __init__(self, submodule, residual_scale=1):
#         super(Input_Output_Sum_Block, self).__init__()
#         self.sub = submodule
#         self.residual_scale = residual_scale;
#         self.flag_first = False;
#     # Elementwise sum the output of a submodule to its input
#     def forward(self, x):
#         if self.flag_first == False:
#             self.sub = self.sub.to(x.device)
#             self.flag_first = True;
#         output = x + self.sub(x)*self.residual_scale
#         return output
#
#     def __repr__(self):
#         tmpstr = 'Input_Output_Sum + \n|'
#         modstr = self.sub.__repr__().replace('\n', '\n|')
#         tmpstr = tmpstr + modstr
#         return tmpstr
#
#
#
# class Input_Output_Sum_Block_With_Projection_Block(nn.Module):
#     #TODO: implement!@@!#%$%*(^
#     # Initialize this with a module
#     def __init__(self, submodule, residual_scale=1):
#         super(Input_Output_Sum_Block, self).__init__()
#         self.sub = submodule
#         self.residual_scale = residual_scale;
#         self.flag_first = False;
#     # Elementwise sum the output of a submodule to its input
#     def forward(self, x):
#         if self.flag_first == False:
#             self.sub = self.sub.to(x.device)
#             self.flag_first = True;
#         output = x + self.sub(x)*self.residual_scale
#         return output
#
#     def __repr__(self):
#         tmpstr = 'Input_Output_Sum_With_Projection + \n|'
#         modstr = self.sub.__repr__().replace('\n', '\n|')
#         tmpstr = tmpstr + modstr
#         return tmpstr
#
#
#
# class Input_Output_Sum_Block_With_StraightThrough_Projection_Block(nn.Module):
#     # Initialize this with a module
#     def __init__(self, submodule, number_of_output_channels, residual_scale=1):
#         super(Input_Output_Sum_Block_With_StraightThrough_Projection_Block, self).__init__()
#         self.sub = submodule
#         self.residual_scale = residual_scale;
#         self.number_of_output_channels = number_of_output_channels;
#         self.projection_block = None;
#         self.flag_first = False;
#
#     # Elementwise sum the output of a submodule to its input
#     def forward(self, x, y_cell_conditional=None, y_deformable_conditional=None):
#         if self.projection_block == None:
#             self.number_of_input_channels = x.shape[1]
#             if self.number_of_output_channels != x.shape[1]:
#                 if self.number_of_output_channels > x.shape[1]:
#                     self.projection_block = Conv_Block(self.number_of_input_channels,self.number_of_output_channels-x.shape[1], kernel_size=3,stride=1,dilation=1,groups=1,bias=True,padding_type='reflect',
#                                                        normalization_function='none', activation_function='none', mode='CNA', initialization_method='xavier')
#                     self.projection_block = self.projection_block.to(x.device)
#                     self.projection_block = Input_Output_Concat_Block(self.projection_block)  #Concat original input with conv_block output to produce needed number of output channels
#                     self.projection_block = self.projection_block.to(x.device)
#                 elif self.number_of_output_channels < x.shape[1]:
#                     self.projection_block = Conv_Block(self.number_of_input_channels,  self.number_of_output_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=True, padding_type='reflect',
#                                                        normalization_function='none', activation_function='none', mode='CNA', initialization_method='dirac') #use conv block with dirac initialization to bias towards pass through
#                     self.projection_block = self.projection_block.to(x.device)
#             elif self.number_of_output_channels == x.shape[1]:
#                 self.projection_block = Identity_Layer();
#             self.projection_block = self.projection_block.to(x.device)
#
#         if self.flag_first == False:
#             self.sub = self.sub.to(x.device)
#             self.flag_first = True;
#
#
#         ### To make this wrapper compatible with the most general configuration which includes modules receiving y_cell_conditional and y_deformable_conditional, check what parameters sub_module accepts: ###
#         # print('input output sum block ' + str(y_cell_conditional.shape))
#         if 'y_cell_conditional' in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' in signature(self.sub.forward).parameters.keys():
#             output = self.projection_block(x) + self.sub(x,y_cell_conditional,y_deformable_conditional) * self.residual_scale
#         elif 'y_cell_conditional' in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' not in signature(self.sub.forward).parameters.keys():
#             output = self.projection_block(x) + self.sub(x,y_cell_conditional) * self.residual_scale
#         elif 'y_cell_conditional' not in signature(self.sub.forward).parameters.keys() and 'y_deformable_conditional' in signature(self.sub.forward).parameters.keys():
#             output = self.projection_block(x) + self.sub(x,y_deformable_conditional) * self.residual_scale
#         else:
#             output = self.projection_block(x) + self.sub(x)*self.residual_scale
#         ########################################################################################################################################################################################################
#
#         return output
#
#     def __repr__(self):
#         tmpstr = 'Input_Output_Sum_With_Dirac_Projection + \n|'
#         modstr = self.sub.__repr__().replace('\n', '\n|')
#         tmpstr = tmpstr + modstr
#         return tmpstr
#
#
#
#
#
# # Concatenate a list of input modules on top of each other to make on long module!:
# def Pile_Modules_On_Top_Of_Each_Other(*args):
#     # Flatten Sequential. It unwraps nn.Sequential.
#
#     # If i only got one module (assuming it's a module) just return it:
#     if len(args) == 1:
#         if isinstance(args[0], OrderedDict):
#             raise NotImplementedError('sequential does not support OrderedDict input.')
#         return args[0]  # No sequential is needed.
#
#     # Take all the modules i got in the args input and simply concatenate them / pile them on top of each other and return a single "long" module.
#     modules = []
#     for module in args:
#         if isinstance(module, nn.Sequential):
#             for submodule in module.children():
#                 modules.append(submodule)
#         elif isinstance(module, nn.Module):
#             modules.append(module)
#     return nn.Sequential(*modules)
#
#
#
#
# class Pile_Modules_On_Top_Of_Each_Other_Layer(nn.Module):
#     def __init__(self, module_list):
#         super(Pile_Modules_On_Top_Of_Each_Other_Layer, self).__init__()
#         self.module_list = module_list
#
#     def forward(self, x, y_cell_conditional=None, y_deformable_conditional=None):
#         output = x.clone();
#
#         ### Check whether i received list (which means a different conditional input for each module) or a single tensor (the same conditional input for all modules): ###
#         if type(y_cell_conditional) == list:
#             flag_y_cell_conditional_list = True;
#         else:
#             flag_y_cell_conditional_list = False
#         if type(y_deformable_conditional) == list:
#             flag_y_deformable_conditional_list = True;
#         else:
#             flag_y_deformable_conditional_list = False
#
#
#         # ### Pass Through Networks: ###
#         # #TODO: maybe using signature(module).parameters().keys() is slow and i should only use it once and save them as internal variables......why not?!
#         # for i, module in enumerate(self.module_list):
#         #     if i == 0:
#         #         output = module(x)
#         #     else:
#         #         output = module(output)
#
#         for i, module in enumerate(self.module_list):
#             ### To make this wrapper compatible with the most general configuration which includes modules receiving y_cell_conditional and y_deformable_conditional, check what parameters sub_module accepts: ###
#             if 'y_cell_conditional' in signature(module.forward).parameters.keys() and 'y_deformable_conditional' in signature(module.forward).parameters.keys():
#                 if flag_y_cell_conditional_list and flag_y_deformable_conditional_list:
#                     output = module(output, y_cell_conditional[i], y_deformable_conditional[i])
#                 elif flag_y_cell_conditional_list:
#                     output = module(output, y_cell_conditional[i], y_deformable_conditional)
#                 elif flag_y_deformable_conditional_list:
#                     output = module(output, y_cell_conditional, y_deformable_conditional[i])
#                 else:
#                     output = module(output, y_cell_conditional, y_deformable_conditional)
#             elif 'y_cell_conditional' in signature(module.forward).parameters.keys() and 'y_deformable_conditional' not in signature(module.forward).parameters.keys():
#                 if flag_y_cell_conditional_list:
#                     output = module(output, y_cell_conditional[i])
#                 else:
#                     output = module(output, y_cell_conditional)
#             elif 'y_cell_conditional' not in signature(module.forward).parameters.keys() and 'y_deformable_conditional' in signature(module.forward).parameters.keys():
#                 output = module(output, y_deformable_conditional)
#                 if flag_y_deformable_conditional_list:
#                     output = module(output, y_deformable_conditional[i])
#                 else:
#                     output = module(output, y_deformable_conditional)
#             else:
#                 output = module(output)
#
#         return output


############################################################################################################################################################################################################################################################




















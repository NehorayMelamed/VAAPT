from easydict import EasyDict
import  torch.nn as nn
import  torch


def update_dict(main_dict, input_dict):
    default_dict = EasyDict(main_dict)
    if input_dict is None:
        input_dict = EasyDict()
    default_dict.update(input_dict)
    return default_dict

class EMA(nn.Module):
    #Exponential moving average
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.flag_first_time_passed = False
    def forward(self, x, last_average):
        if self.flag_first_time_passed==False:
            new_average = x
            self.flag_first_time_passed = True
        else:
            new_average = self.mu * x + (1 - self.mu) * last_average
        return new_average



############################################################################################################################################################################################
### Clip Gradient: ###
from RapidBase.Utils.Pytorch_Numpy_Utils import get_model_gradient_norm
from RapidBase.Utils.tic_toc import *
class Clip_Gradient_Base(object):
    ### Clip Gradient: ###
    def __init__(self, input_dict=None):
        ### Default Parameters: ###
        self.flag_use_gradient_clipping = input_dict.flag_use_gradient_clipping
        self.flag_clip_grad_value_or_norm = input_dict.flag_clip_grad_value_or_norm
        self.gradient_clip_value_factor = input_dict.gradient_clip_value_factor
        self.total_norm = input_dict.gradient_clip_norm
        self.gradient_clip_value = input_dict.gradient_clip_norm * input_dict.gradient_clip_value_factor
        self.EMA_alpha = 0.05

        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)
        self.final_dict = self.__dict__

        ### Initialize EMA object: ###
        self.ema_object = EMA(self.final_dict.EMA_alpha)


    def clip_gradient(self, Network, Train_dict):
        optimization_dict = Train_dict.optimization_dict

        ### if this is not an anomaly, use regular gradient clipping if so wanted: ###
        if self.flag_use_gradient_clipping:
            if self.flag_clip_grad_value_or_norm == 'norm':
                # tic() #TODO: gradient clipping takes a long time...think of possible replacements? maybe some way of keeping parameters in memory? parallel loop?
                total_norm_before_clipping = torch.nn.utils.clip_grad_norm_(Network.parameters(), Train_dict.gradient_clip_norm * 1.2)
                total_norm_after_clipping = torch.nn.utils.clip_grad_norm_(Network.parameters(), Train_dict.gradient_clip_norm * 1.2)
                # toc('pytorch clip grad norm')
            elif self.flag_clip_grad_value_or_norm == 'value':
                torch.nn.utils.clip_grad_value_(Network.parameters(), Train_dict.gradient_clip_norm * Train_dict.gradient_clip_value_factor * 1.2)

            # (*). Update norm moving average: (*)#
            if self.flag_clip_grad_value_or_norm == 'norm':
                Train_dict.gradient_clip_value = self.ema_object(total_norm_after_clipping, Train_dict.gradient_clip_value).item()
                Train_dict.total_norm = self.ema_object(total_norm_after_clipping, total_norm_after_clipping).item()
            elif self.flag_clip_grad_value_or_norm == 'value':
                Train_dict.gradient_clip_value = self.ema_object(grad_max_value_after_clipping, Train_dict.gradient_clip_value).item()
                Train_dict.total_norm = self.ema_object(total_norm_after_clipping, total_norm_after_clipping).item()

        return Network, Train_dict


    # def clip_gradient(self, Network, Train_dict):
    #     optimization_dict = Train_dict.optimization_dict
    #
    #     ### check gradient norm to see if this is an anomaly: ###
    #     tic()
    #     total_norm_current, grad_average_value, grad_max_value = get_model_gradient_norm(Network)
    #     toc('get gradient norm')
    #     print('grad total norm: ' + str(total_norm_current))
    #     print('grad average value: ' + str(grad_average_value))
    #     print('grad max value: ' + str(grad_max_value))
    #
    #     ### Decide what to do according to gradient norm: ###
    #     if total_norm_current > Train_dict.total_norm * Train_dict.anomaly_factor_threshold:
    #         ### if this is an anomaly, disregard it: ###
    #         total_norm_after_clipping = torch.nn.utils.clip_grad_norm_(Network.parameters(), 0)
    #     else:
    #         ### if this is not an anomaly, use regular gradient clipping if so wanted: ###
    #         if optimization_dict.flag_use_gradient_clipping:
    #             if optimization_dict.flag_clip_grad_value_or_norm == 'norm':
    #                 tic()
    #                 total_norm_before_clipping = torch.nn.utils.clip_grad_norm_(Network.parameters(), Train_dict.gradient_clip_value * 1.2)
    #                 toc('pytorch clip grad norm')
    #             elif optimization_dict.flag_clip_grad_value_or_norm == 'value':
    #                 torch.nn.utils.clip_grad_value_(Network.parameters(), Train_dict.gradient_clip_value * 1.2)
    #
    #         ### Get gradient norm after clipping: ###
    #         total_norm_after_clipping, grad_average_value_after_clipping, grad_max_value_after_clipping = get_model_gradient_norm(Network)
    #
    #         # (*). Update norm moving average: (*)#
    #         if optimization_dict.flag_clip_grad_value_or_norm == 'norm':
    #             Train_dict.gradient_clip_value = self.ema_object(total_norm_after_clipping, Train_dict.gradient_clip_value).item()
    #             Train_dict.total_norm = self.ema_object(total_norm_after_clipping, total_norm_after_clipping).item()
    #         elif optimization_dict.flag_clip_grad_value_or_norm == 'value':
    #             Train_dict.gradient_clip_value = self.ema_object(grad_max_value_after_clipping, Train_dict.gradient_clip_value).item()
    #             Train_dict.total_norm = self.ema_object(total_norm_after_clipping, total_norm_after_clipping).item()

        return Network, Train_dict



############################################################################################################################################################################################











from easydict import EasyDict
import torch


def update_dict(main_dict, input_dict):
    default_dict = EasyDict(main_dict)
    if input_dict is None:
        input_dict = EasyDict()
    default_dict.update(input_dict)
    return default_dict


############################################################################################################################################################################################
### Optimizer: ###
class Optimizer_Base(object):
    def __init__(self, Network, intial_lr=1e-2, hypergrad_lr=0, optimizer_type='adam'):
        ### Default Parameters: ###
        self.init_learning_rate = intial_lr
        self.hypergrad_lr = hypergrad_lr

        input_dict = None
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)
        self.final_dict = self.__dict__

        ### Get Parameters: ###
        parameters_to_optimize = [{'params': Network.parameters()}, ]

        ### Get Optimizer: ###
        # Network_Optimizer = AdamHD(parameters_to_optimize, lr=Train_dict.init_learning_rate, hypergrad_lr=Train_dict.hypergrad_lr)
        # Network_Optimizer = Adam(parameters_to_optimize, lr=Train_dict.init_learning_rate)
        # Network_Optimizer = Adam(parameters_to_optimize, lr=Train_dict.init_learning_rate, amsgrad=True)
        if str.lower(optimizer_type) == 'sgd':
            self.Network_Optimizer = torch.optim.SGD(parameters_to_optimize, lr=self.final_dict.init_learning_rate)
        if str.lower(optimizer_type) == 'adam':
            self.Network_Optimizer = torch.optim.Adam(parameters_to_optimize,
                                                      lr=self.final_dict.init_learning_rate, amsgrad=False, weight_decay=0)

        self.Network_Optimizer = torch.optim.Adam(parameters_to_optimize,
                                                  lr=self.final_dict.init_learning_rate)#,betas=[0.9,0.999])


    def Step(self):
        return self.Network_Optimizer.step()
############################################################################################################################################################################################

### Optimizer: ###
class Optimizer_LearnableLeakyRelu(Optimizer_Base):
    def __init__(self, Network, intial_lr=1e-2, hypergrad_lr=0):
        ### Default Parameters: ###
        self.init_learning_rate = intial_lr
        self.hypergrad_lr = hypergrad_lr

        input_dict = None
        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)
        self.final_dict = self.__dict__

        ### Get Parameters: ###
        parameters_to_optimize = [{'params': Network.parameters()}, ]

        my_list = ['leak_param']
        leak_param = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, Network.named_parameters()))))
        base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, Network.named_parameters()))))

        ### Get Optimizer: ###
        # Network_Optimizer = AdamHD(parameters_to_optimize, lr=Train_dict.init_learning_rate, hypergrad_lr=Train_dict.hypergrad_lr)
        # Network_Optimizer = Adam(parameters_to_optimize, lr=Train_dict.init_learning_rate)
        # Network_Optimizer = Adam(parameters_to_optimize, lr=Train_dict.init_learning_rate, amsgrad=True)
        # Network_Optimizer = SGD(parameters_to_optimize, lr=init_learning_rate)
        self.Network_Optimizer = torch.optim.Adam([{'params': base_params}, {'params': leak_param, 'lr': '{}'.format(self.final_dict.init_learning_rate)}],
                                      lr=self.final_dict.init_learning_rate, amsgrad=True, weight_decay=2e-5)



from easydict import EasyDict
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def update_dict(main_dict, input_dict):
    default_dict = EasyDict(main_dict)
    if input_dict is None:
        input_dict = EasyDict()
    default_dict.update(input_dict)
    return default_dict


def set_optimizer_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


############################################################################################################################################################################################
### Learning Rate: ###
class LR_Scheduler_Base_Wrapper(object):
    def __init__(self, network_optimizer, lr_scheduler):
        self.lr_scheduler = lr_scheduler
        self.network_optimizer = network_optimizer

    def step(self, Network_optimizer=None, total_loss=None, Train_dict=None):
        #(*). i want a universal scheduler calling using the .step() function calling. however, my custom scheduler objects also include loss and train_dict so i wrap:
        if Network_optimizer is None and total_loss is None and Train_dict is None:
            self.lr_scheduler.step()
        elif type(self.lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
            ### reduce on plateau pytorch learning rate scheduler: ###
            self.lr_scheduler.step(total_loss)
        elif 'torch.optim' in str(type(self.lr_scheduler)):
            ### "normal" pytorch lr_scheduler
            self.lr_scheduler.step(Train_dict.global_step)   #TODO: maybe use this instead?: self.lr_scheduler.step(Train_dict.global_step)
        else:
            ### one of my custom learning rate schedulers: ###
            self.lr_scheduler.step(Network_optimizer, total_loss, Train_dict)

        return Network_optimizer, Train_dict


class LR_Scheduler_Base(object):
    def __init__(self):
        1

    def get_Optimizer_lr(self, optimizer_G): #TODO: use input scheduler get_lr? or base class get_lr
        if type(optimizer_G.param_groups[0]['lr']) == torch.Tensor:
            Network_lr = optimizer_G.param_groups[0]['lr'].cpu().item()
        elif type(optimizer_G.param_groups[0]['lr']) == float:
            Network_lr = optimizer_G.param_groups[0]['lr']
        elif type(optimizer_G.param_groups[0]['lr']) == np.float64:
            Network_lr = optimizer_G.param_groups[0]['lr']
        return Network_lr

    def set_Optimizer_lr(self, optimizer_G, new_lr):
        set_optimizer_learning_rate(optimizer_G, new_lr)


class LR_Scheduler_ReduceLROnPlateauCustom(LR_Scheduler_Base):
    def __init__(self, Network_optimizer,
                 flag_change_optimizer_betas=False,
                 flag_use_learning_rate_scheduler=True,
                 minimum_learning_rate=1e-6,
                 learning_rate_lowering_factor=0.8,
                 learning_rate_lowering_patience_before_lowering=200,
                 learning_rate_lowering_tolerance_factor_from_best_so_far=0.03,
                 initial_spike_learning_rate_reduction_multiple=0.8,
                 minimum_learning_rate_counter=0,
                 input_dict=None):

        ### Default Parameters: ###
        self.flag_momentum_cycling = input_dict.flag_momentum_cycling
        self.momentum_bounds = input_dict.momentum_bounds
        self.flag_change_optimizer_betas = flag_change_optimizer_betas
        self.flag_use_learning_rate_scheduler = flag_use_learning_rate_scheduler
        self.minimum_learning_rate = minimum_learning_rate
        self.learning_rate_lowering_factor = learning_rate_lowering_factor
        self.learning_rate_lowering_patience_before_lowering = learning_rate_lowering_patience_before_lowering
        #self.learning_rate_lowering_patience_before_lowering = 10
        self.learning_rate_lowering_tolerance_factor_from_best_so_far = learning_rate_lowering_tolerance_factor_from_best_so_far
        self.initial_spike_learning_rate_reduction_multiple = initial_spike_learning_rate_reduction_multiple
        self.minimum_learning_rate_counter = minimum_learning_rate_counter
        self.flag_uptick_learning_rate_after_falling_below_minimum = input_dict.flag_uptick_learning_rate_after_falling_below_minimum

        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)
        self.final_dict = self.__dict__

        ### Initialize Learning Rate Scheduler: ###
        Network_lr_previous = self.init_learning_rate
        if self.flag_use_learning_rate_scheduler == False:
            learning_rate_lowering_patience_before_lowering = 1e9
        learning_rate = self.init_learning_rate * \
            (self.initial_spike_learning_rate_reduction_multiple ** self.minimum_learning_rate_counter)
        set_optimizer_learning_rate(Network_optimizer, learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Network_optimizer,
                                                                       mode='min',
                                                                       factor=self.learning_rate_lowering_factor,
                                                                       patience=self.learning_rate_lowering_patience_before_lowering,
                                                                       threshold=self.learning_rate_lowering_tolerance_factor_from_best_so_far,
                                                                       threshold_mode='rel',
                                                                       min_lr=minimum_learning_rate)

        self.learning_rate = learning_rate
        self.Network_lr_previous = Network_lr_previous

    def step(self, Network_optimizer, total_loss, Train_dict):
        ### Get Network Learning Rate: ###
        Network_lr_previous = self.get_Optimizer_lr(Network_optimizer)

        ### Update Learning-Rate Scheduler: ###
        self.lr_scheduler.step(total_loss)

        ### Get Network Learning Rate: ###
        Network_lr = self.get_Optimizer_lr(Network_optimizer)

        ### If Learning Rate changed do some stuff: ###
        if Network_lr < Network_lr_previous:
            # TODO: check out what is happening to betas/momentum throughout training....i don't think this is very effective. i need to effectively
            # TODO: lower the momentum when grad_norm raises and perhapse use cyclic learning rates and lower/raise momentum as learning rate lowers/raises
            print('CHANGED LEEARNING RATE!!!!')
            if Train_dict.flag_change_optimizer_betas:
                for i, param_group in enumerate(Network_optimizer.param_groups):
                    # beta_new = 0.1*beta_old + 0.9
                    param_group['betas'] = tuple([1 - (1 - b) * 0.1 for b in param_group['betas']])
        Network_lr_previous = Network_lr

        ##############################################################
        ### If Learning Rate falls below minimum then spike it up: ###
        if Network_lr < Train_dict.minimum_learning_rate and self.flag_uptick_learning_rate_after_falling_below_minimum:
            Train_dict.minimum_learning_rate_counter += 1
            # If we're below the minimum learning rate than it's time to jump the LR up
            # to try and get to a lower local minima:

            ### Initialize Learning Rate (later to be subject to lr scheduling): ###
            learning_rate = Train_dict.init_learning_rate * \
                (Train_dict.initial_spike_learning_rate_reduction_multiple ** Train_dict.minimum_learning_rate_counter)
            set_optimizer_learning_rate(Network_optimizer, learning_rate)

            ### Initialize Learning Rate Scheduler: ###
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Network_optimizer,
                                                                           mode='min',
                                                                           factor=Train_dict.learning_rate_lowering_factor,
                                                                           patience=Train_dict.learning_rate_lowering_patience_before_lowering,
                                                                           threshold=Train_dict.learning_rate_lowering_tolerance_factor_from_best_so_far,
                                                                           threshold_mode='rel',
                                                                           min_lr=self.minimum_learning_rate)
        ###################################################################

        ### Assign: ###
        self.Network_lr = Network_lr
        self.Network_lr_previous = Network_lr_previous
        Train_dict.Network_lr = Network_lr
        return Network_optimizer, Train_dict






#TODO: make the following scheduling_rate objects: WarmpUp, Cyclic
class GradualWarmupScheduler(LR_Scheduler_Base):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        number_of_warmup_epochs: target learning rate is reached at number_of_warmup_epochs, gradually
        scheduler_after_warmup: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, initial_lr, final_lr, number_of_warmup_epochs, scheduler_after_warmup=None, input_dict=None):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.number_of_warmup_epochs = number_of_warmup_epochs
        self.scheduler_after_warmup = scheduler_after_warmup
        self.flag_finished_warmup = False

        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)
        self.final_dict = self.__dict__

        super(GradualWarmupScheduler, self).__init__()

    def step(self, Network_optimizer, total_loss, Train_dict):
        ### If we're in warmup stage -> update LR accordingly: ###
        if Train_dict.global_step <= self.number_of_warmup_epochs:
            warmup_lr = self.initial_lr + (self.final_lr - self.initial_lr) * (Train_dict.global_step / self.number_of_warmup_epochs)
            set_optimizer_learning_rate(Network_optimizer, warmup_lr)
        else:
            ### If we're after warumup -> Use external scheduler: ###
            # self.scheduler_after_warmup.step(Network_optimizer, total_loss, Train_dict.global_step - self.number_of_warmup_epochs)
            self.scheduler_after_warmup.step(Network_optimizer, total_loss, Train_dict)

        ### Assign: ###
        self.Network_lr = self.get_Optimizer_lr(Network_optimizer)
        Train_dict.Network_lr = self.Network_lr
        return Network_optimizer, Train_dict

import numpy as np
class TwoStage_Scheduler(LR_Scheduler_Base):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        number_of_warmup_epochs: target learning rate is reached at number_of_warmup_epochs, gradually
        scheduler_after_warmup: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, number_of_warmup_epochs, first_stage_scheduler=None, second_stage_scheduler=None, input_dict=None):
        self.number_of_warmup_epochs = number_of_warmup_epochs
        self.first_stage_scheduler = first_stage_scheduler
        self.second_stage_scheduler = second_stage_scheduler
        self.flag_finished_warmup = False

        ### Get Final Internal Dictionary: ###
        self.__dict__ = update_dict(self.__dict__, input_dict)
        self.final_dict = self.__dict__

        super(TwoStage_Scheduler, self).__init__()

    def step(self, Network_optimizer=None, total_loss=None, Train_dict=None):
        ### If we're in warmup stage -> update LR accordingly: ###
        if Train_dict.global_step <= self.number_of_warmup_epochs:
            self.first_stage_scheduler.step(Network_optimizer, total_loss, Train_dict)
        else:
            self.second_stage_scheduler.step(Network_optimizer, total_loss, Train_dict)

        ### Assign: ###
        self.Network_lr = self.get_Optimizer_lr(Network_optimizer)
        Train_dict.Network_lr = self.Network_lr
        return Network_optimizer, Train_dict

# class GradualWarmupScheduler(_LRScheduler):
#     """ Gradually warm-up(increasing) learning rate in optimizer.
#     Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
#
#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
#         number_of_warmup_epochs: target learning rate is reached at number_of_warmup_epochs, gradually
#         scheduler_after_warmup: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
#     """
#
#     def __init__(self, optimizer, multiplier, number_of_warmup_epochs, scheduler_after_warmup=None):
#         self.multiplier = multiplier
#         if self.multiplier < 1.:
#             raise ValueError('multiplier should be greater thant or equal to 1.')
#         self.number_of_warmup_epochs = number_of_warmup_epochs
#         self.scheduler_after_warmup = scheduler_after_warmup
#         self.flag_finished_warmup = False
#         super(GradualWarmupScheduler, self).__init__(optimizer)
#
#     def get_lr(self):
#         ### If we are done with the warmup -> update flag: ###
#         if self.last_epoch > self.number_of_warmup_epochs and (not self.flag_finished_warmup):
#             self.scheduler_after_warmup.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
#             self.flag_finished_warmup = True
#
#         ### If we are done with the warmpup -> use the external scheduler: ###
#         if self.last_epoch > self.number_of_warmup_epochs:
#             return self.scheduler_after_warmup.get_lr()
#
#         ### If we are still in warmup: ###
#         if self.multiplier == 1.0:
#             return [base_lr * (float(self.last_epoch) / self.number_of_warmup_epochs) for base_lr in self.base_lrs]
#         else:
#             return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.number_of_warmup_epochs + 1.) for base_lr
#                     in self.base_lrs]
#
#     def step(self, metrics, epoch=None):
#         ### Update current epoch: ###
#         if epoch is None:
#             epoch = self.last_epoch + 1
#         self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
#
#         ### If we're in warmup stage -> update LR accordingly: ###
#         if self.last_epoch <= self.number_of_warmup_epochs:
#             warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.number_of_warmup_epochs + 1.) for
#                          base_lr in self.base_lrs]
#             for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
#                 param_group['lr'] = lr
#         else:
#             ### If we're after warumup -> Use external scheduler: ###
#             self.scheduler_after_warmup.step(metrics, epoch - self.number_of_warmup_epochs)


############################################################################################################################################################################################

### Learning Rate: ###
class LR_Scheduler_LearnableLeakyRelu(LR_Scheduler_Base):

    def set_optimizer_learning_rate(self, Network_optimizer, learning_rate):
        # for param_group in Network_optimizer.param_groups:
        #     param_group['lr'] = learning_rate
        Network_optimizer.param_groups[0]['lr'] = learning_rate
        Network_optimizer.param_groups[1]['lr'] = 1e-3

    def step(self, Network_optimizer, total_loss, Train_dict):
        ### Get Needed Dictionaries: ###
        optimization_dict = Train_dict.optimization_dict
        dynamic_optimization_dict = Train_dict.dynamic_optimization_dict

        ### Get Network Learning Rate: ###
        Network_lr_previous = self.get_Optimizer_lr(Network_optimizer)

        ### Update Learning-Rate Scheduler: ###
        self.lr_scheduler.step(total_loss)

        ### Get Network Learning Rate: ###
        Network_lr = self.get_Optimizer_lr(Network_optimizer)

        ### If Learning Rate changed do some stuff: ###
        if Network_lr < Network_lr_previous:
            print('CHANGED LEEARNING RATE!!!!')
            if Train_dict.flag_change_optimizer_betas:
                for i, param_group in enumerate(Network_optimizer.param_groups):
                    # beta_new = 0.1*beta_old + 0.9
                    param_group['betas'] = tuple([1 - (1 - b) * 0.1 for b in param_group['betas']])
        Network_lr_previous = Network_lr

        ### If Learning Rate falls below minimum then spike it up: ###
        if Network_lr < Train_dict.minimum_learning_rate:
            dynamic_optimization_dict.minimum_learning_rate_counter += 1
            # If we're below the minimum learning rate than it's time to jump the LR up
            # to try and get to a lower local minima:

            ### Initialize Learning Rate (later to be subject to lr scheduling): ###
            learning_rate = Train_dict.init_learning_rate * \
                (Train_dict.initial_spike_learning_rate_reduction_multiple ** Train_dict.minimum_learning_rate_counter)
            self.set_optimizer_learning_rate(Network_optimizer, learning_rate)

            ### Initialize Learning Rate Scheduler: ###
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Network_optimizer,
                                                               mode='min',
                                                               factor=Train_dict.learning_rate_lowering_factor,
                                                               patience=Train_dict.learning_rate_lowering_patience_before_lowering,
                                                               threshold=Train_dict.learning_rate_lowering_tolerance_factor_from_best_so_far,
                                                               threshold_mode='rel')

        ### Assign: ###
        self.Network_lr = Network_lr
        self.Network_lr_previous = Network_lr_previous
        Train_dict.Network_lr = Network_lr
        return Network_optimizer, Train_dict





def LR_scheduler_examples():
    Train_dict.flag_LR_Scheduler_batch_or_epoch = 'batch'  # 'batch', 'epoch'  - decide on which basis to use it, batch based or epoch based
    number_of_steps_per_epoch = len(train_dataset) // IO_dict.number_of_epochs

    ### Warmup with cosine annealing: ###
    warmup_iterations = 3000
    total_number_of_iterations = 20e3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(Network_Optimizer_object.Network_Optimizer,
                                                            T_max=total_number_of_iterations, eta_min=1e-6)
    scheduler_cosine_wrapper = LR_Scheduler_Base_Wrapper(Network_Optimizer_object.Network_Optimizer, scheduler_cosine)
    LR_scheduler_object = GradualWarmupScheduler(Network_Optimizer_object.Network_Optimizer, initial_lr=1e-6,
                                                 final_lr=1e-4,
                                                 number_of_warmup_epochs=warmup_iterations,
                                                 scheduler_after_warmup=scheduler_cosine_wrapper,
                                                 input_dict=optimization_dict)

    ### OneCycle Policy LR Scheduler: ###
    steps_per_epoch = (len(train_dataset) // IO_dict.batch_size)
    number_of_epochs_for_cycle = 2
    initial_learning_rate = 1e-4
    max_learning_rate = 1e-3
    final_learning_rate = 1e-6
    div_factor = max_learning_rate / initial_learning_rate
    final_div_factor = initial_learning_rate / final_learning_rate
    LR_scheduler_object = torch.optim.lr_scheduler.OneCycleLR(Network_Optimizer_object.Network_Optimizer,
                                                              max_learning_rate,
                                                              # to be determined, maybe from lr_finder
                                                              total_steps=None,
                                                              epochs=number_of_epochs_for_cycle,
                                                              steps_per_epoch=steps_per_epoch,
                                                              pct_start=0.3,
                                                              anneal_strategy='linear',
                                                              cycle_momentum=True,
                                                              base_momentum=0.85, max_momentum=0.95,
                                                              div_factor=div_factor, final_div_factor=final_div_factor)
    LR_scheduler_object = LR_Scheduler_Base_Wrapper(Network_Optimizer_object.Network_Optimizer, LR_scheduler_object)
    ### Test one_cycle policy learning rate scheduler object: ###
    # learning_rate_array = np.ones((10**5,1))
    # for counter in np.arange(10**5):
    #     current_learning_rate = get_optimizer_learning_rate(Network_Optimizer_object.Network_Optimizer)
    #     LR_scheduler_object.step()
    #     learning_rate_array[counter] = current_learning_rate
    # plot(learning_rate_array[0:counter])

    ### Reduce On Plateau: ###
    optimization_dict.minimum_learning_rate = 1e-6
    optimization_dict.learning_rate_lowering_factor = 0.8
    optimization_dict.learning_rate_lowering_patience_before_lowering = 1500
    optimization_dict.learning_rate_lowering_tolerance_factor_from_best_so_far = 0.03
    LR_scheduler_object = torch.optim.lr_scheduler.ReduceLROnPlateau(Network_Optimizer_object.Network_Optimizer,
                                                                     mode='min',
                                                                     factor=optimization_dict.learning_rate_lowering_factor,
                                                                     patience=optimization_dict.learning_rate_lowering_patience_before_lowering,
                                                                     threshold=optimization_dict.learning_rate_lowering_tolerance_factor_from_best_so_far,
                                                                     threshold_mode='rel',
                                                                     min_lr=optimization_dict.minimum_learning_rate)
    LR_scheduler_object = LR_Scheduler_Base_Wrapper(Network_Optimizer_object.Network_Optimizer, LR_scheduler_object)

    ### MultiStep LR Scheduler: ###
    LR_scheduler_object = torch.optim.lr_scheduler.MultiStepLR(Network_Optimizer_object.Network_Optimizer,
                                                               milestones=[5000, 8000],
                                                               gamma=0.1)
    LR_scheduler_object = LR_Scheduler_Base_Wrapper(Network_Optimizer_object.Network_Optimizer, LR_scheduler_object)

    ### Custom: Lower on plato with logic: ###
    optimization_dict.minimum_learning_rate = 1e-6
    optimization_dict.learning_rate_lowering_factor = 0.8
    optimization_dict.learning_rate_lowering_patience_before_lowering = 1500
    optimization_dict.flag_momentum_cycling = False
    optimization_dict.momentum_bounds = [0.1, 0.9]
    optimization_dict.flag_uptick_learning_rate_after_falling_below_minimum = False
    LR_scheduler_object = LR_Scheduler_Base(Network_Optimizer_object.Network_Optimizer, input_dict=optimization_dict)
    optimization_dict = LR_scheduler_object.final_dict

    ### Custom: GradualWarmup + reduce on plateau: ###
    # steps_per_epoch = (len(train_dataset)//IO_dict.batch_size)
    # number_of_epochs_for_cycle = 2
    # (1). GradualWarmup:
    number_of_iterations_per_cycle = 2000
    initial_learning_rate = 1e-4
    max_learning_rate = 1e-3
    final_learning_rate = 1e-6
    div_factor = max_learning_rate / initial_learning_rate
    final_div_factor = initial_learning_rate / final_learning_rate
    LR_scheduler_object = torch.optim.lr_scheduler.OneCycleLR(Network_Optimizer_object.Network_Optimizer,
                                                              max_learning_rate,
                                                              # to be determined, maybe from lr_finder
                                                              total_steps=None,
                                                              epochs=1,
                                                              steps_per_epoch=number_of_iterations_per_cycle,
                                                              pct_start=0.3,
                                                              anneal_strategy='linear',
                                                              cycle_momentum=True,
                                                              base_momentum=0.85, max_momentum=0.95,
                                                              div_factor=div_factor, final_div_factor=final_div_factor)
    LR_scheduler_object_1 = LR_Scheduler_Base_Wrapper(Network_Optimizer_object.Network_Optimizer, LR_scheduler_object)
    # (2). Warmup with cosine annealing:
    LR_scheduler_object_2 = torch.optim.lr_scheduler.MultiStepLR(Network_Optimizer_object.Network_Optimizer,
                                                                 milestones=[5000, 8000],
                                                                 gamma=0.1)
    LR_scheduler_object_2 = LR_Scheduler_Base_Wrapper(Network_Optimizer_object.Network_Optimizer, LR_scheduler_object_2)
    ### Combine Them: ###
    warmup_iterations = number_of_iterations_per_cycle
    LR_scheduler_object = TwoStage_Scheduler(warmup_iterations, LR_scheduler_object_1, LR_scheduler_object_2)


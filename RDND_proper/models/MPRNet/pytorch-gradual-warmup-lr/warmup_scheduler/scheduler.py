from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        number_of_warmup_epochs: target learning rate is reached at number_of_warmup_epochs, gradually
        scheduler_after_warmup: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, number_of_warmup_epochs, scheduler_after_warmup=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.number_of_warmup_epochs = number_of_warmup_epochs
        self.scheduler_after_warmup = scheduler_after_warmup
        self.flag_finished_warmup = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        ### If we are done with the warmup -> update flag: ###
        if self.last_epoch > self.number_of_warmup_epochs and (not self.flag_finished_warmup):
            self.scheduler_after_warmup.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
            self.flag_finished_warmup = True

        ### If we are done with the warmpup -> use the external scheduler: ###
        if self.last_epoch > self.number_of_warmup_epochs:
            return self.scheduler_after_warmup.get_lr()

        ### If we are still in warmup: ###
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.number_of_warmup_epochs) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.number_of_warmup_epochs + 1.) for base_lr in self.base_lrs]

    def step(self, metrics, epoch=None):
        ### Update current epoch: ###
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning

        ### If we're in warmup stage -> update LR accordingly: ###
        if self.last_epoch <= self.number_of_warmup_epochs:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.number_of_warmup_epochs + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            ### If we're after warumup -> Use external scheduler: ###
            self.scheduler_after_warmup.step(metrics, epoch - self.number_of_warmup_epochs)




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
#         if self.last_epoch > self.number_of_warmup_epochs:
#             if self.scheduler_after_warmup is not None:
#                 if not self.flag_finished_warmup:
#                     self.scheduler_after_warmup.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
#                     self.flag_finished_warmup = True
#                 return self.scheduler_after_warmup.get_lr()
#             return [base_lr * self.multiplier for base_lr in self.base_lrs]
#
#         if self.multiplier == 1.0:
#             return [base_lr * (float(self.last_epoch) / self.number_of_warmup_epochs) for base_lr in self.base_lrs]
#         else:
#             return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.number_of_warmup_epochs + 1.) for base_lr in self.base_lrs]
#
#     def step_ReduceLROnPlateau(self, metrics, epoch=None):
#         if epoch is None:
#             epoch = self.last_epoch + 1
#         self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
#         if self.last_epoch <= self.number_of_warmup_epochs:
#             warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.number_of_warmup_epochs + 1.) for base_lr in self.base_lrs]
#             for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
#                 param_group['lr'] = lr
#         else:
#             if epoch is None:
#                 self.scheduler_after_warmup.step(metrics, None)
#             else:
#                 self.scheduler_after_warmup.step(metrics, epoch - self.number_of_warmup_epochs)
#
#     def step(self, epoch=None, metrics=None):
#         # If my sheduler_after_warmup is something external then decide what to do
#         if type(self.scheduler_after_warmup) != ReduceLROnPlateau:
#             if self.flag_finished_warmup and (self.scheduler_after_warmup is not None):
#                 if epoch is None:
#                     self.scheduler_after_warmup.step(None)
#                 else:
#                     self.scheduler_after_warmup.step(epoch - self.number_of_warmup_epochs)
#             else:
#                 return super(GradualWarmupScheduler, self).step(epoch)
#         else:
#             # If my scheduler after
#             self.step_ReduceLROnPlateau(metrics, epoch)

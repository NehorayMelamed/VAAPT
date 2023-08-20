# import RapidBase.import_all
# from RapidBase.import_all import *

import RapidBase.Basic_Import_Libs
from RapidBase.Basic_Import_Libs import *

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import ESRGAN_utils
# from ESRGAN_utils import *





####################################################################################################################################################################################################################################################################################################################################################################################################################################
##### Loss Functions and Optimizers:
##############################################
#Optimizer Learning Rate get/set:
def get_optimizer_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def set_optimizer_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def multiply_optimizer_learning_rate(optimizer, lr_factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_factor

def update_learning_rate_using_schadualer(schedualers):
    for scheduler in schedualers:
        scheduler.step()
########################################################################################################################################################################################################################################################################################






########################################################################################################################################################################################################################################################################################
### Basic Optimizers: ###
#TODO: this is perhapse to tied down to the OPT paradigm, which makes it not very accesible as a stand alone function and in any case i think i should send in the OPT.variables_dictionary and not the OPT itself....
def Optimizer(parameters, OPT):
    if str.lower(OPT.optimizer_name) == str.lower('Adadelta'):
        kw_arguments = {'lr': 1, 'rho': 0.9, 'eps': 1e-6, 'weight_decay': 0};  # Adadelta
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments, OPT.variables_dictionary)
        return torch.optim.Adadelta(params=parameters,**kw_arguments);
    if str.lower(OPT.optimizer_name) == str.lower('Adagrad'):
        kw_arguments = {'lr': 1e-2, 'lr_decay': 0, 'weight_decay': 0, 'initial_accumulator_value': 0};  # Adagrad
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments, OPT.variables_dictionary)
        return torch.optim.Adagrad(params=parameters,**kw_arguments);
    if str.lower(OPT.optimizer_name) == str.lower('Adamax'):
        kw_arguments = {'lr': 2e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0};  # Adamax
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments, OPT.variables_dictionary)
        return torch.optim.Adamax(params=parameters,**kw_arguments);
    if str.lower(OPT.optimizer_name) == str.lower('Adam'):
        kw_arguments = {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False};  # Adam
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments, OPT.variables_dictionary)
        return torch.optim.Adam(params=parameters,**kw_arguments);
    if str.lower(OPT.optimizer_name) == str.lower('SparseAdam'):
        kw_arguments = {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8};  # SparseAdam
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments, OPT.variables_dictionary)
        return torch.optim.SparseAdam(params=parameters,**kw_arguments);
    if str.lower(OPT.optimizer_name) == str.lower('SGD'):
        kw_arguments = {'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False};  # SGD
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments, OPT.variables_dictionary)
        return torch.optim.SGD(params=parameters,**kw_arguments);
    if str.lower(OPT.optimizer_name) == str.lower('ASGD'):
        kw_arguments = {'lr': 1e-2, 'lambd': 1e-4, 'alpha': 0.75, 't0': 1e6, 'weight_decay': 0};  # ASGD
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments, OPT.variables_dictionary)
        return torch.optim.ASGD(params=parameters,**kw_arguments);
    if str.lower(OPT.optimizer_name) == str.lower('RMSprop'):
        kw_arguments = {'lr': 1e-2, 'alpha': 0.99, 'eps': 1e-8, 'weight_decay': 0, 'momentum': 0, 'centered': False};  # RMSprop
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments, OPT.variables_dictionary)
        return torch.optim.RMSprop(params=parameters,**kw_arguments);
    if str.lower(OPT.optimizer_name) == str.lower('Rprop'):
        kw_arguments = {'lr': 1e-2, 'etas': (0.5, 1.2), 'step': (1e-6, 50)};  # Rprop
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments, OPT.variables_dictionary)
        return torch.optim.Rprop(params=parametersr,**kw_arguments);
    if str.lower(OPT.optimizer_name) == str.lower('LBFGS'):
        kw_arguments = {'lr': 1, 'max_iter': 20, 'max_eval': None, 'tolerance_grad': 1e-5, 'tolerance_change': 1e-9,'history_size': 100, 'line_search_fn': None};  # LBFGS
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments, OPT.variables_dictionary)
        return torch.optim.LBFGS(params=parameters,**kw_arguments);



class AdamHD(torch.optim.Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hypergrad_lr (float, optional): hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        `Online Learning Rate Adaptation with Hypergradient Descent`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Online Learning Rate Adaptation with Hypergradient Descent:
        https://openreview.net/forum?id=BkrsAzWAb
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, hypergrad_lr=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hypergrad_lr=hypergrad_lr)
        super(AdamHD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                if state['step'] > 1:
                    prev_bias_correction1 = 1 - beta1 ** (state['step'] - 1)
                    prev_bias_correction2 = 1 - beta2 ** (state['step'] - 1)
                    # Hypergradient for Adam:
                    h = torch.dot(grad.view(-1), torch.div(exp_avg, exp_avg_sq.sqrt().add_(group['eps'])).view(-1)) * math.sqrt(prev_bias_correction2) / prev_bias_correction1
                    # Hypergradient descent of the learning rate:
                    group['lr'] += group['hypergrad_lr'] * h

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss







########################################################################################################################################################################################################################################################################################
####################
# Learning-Rate Schedualer Wrapper:
####################
# (*). Auxiliary Loss function:
def Learning_Rate_Schedualer(optimizer,OPT):
    if str.lower(OPT.learning_rate_schedualer_scheme) == 'cosine_annealing':
        kw_arguments = {'eta_min':0,'last_epoch':-1,'T_max':None};  #Cosine Annealing
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments,OPT.variables_dictionary)
        return lr_scheduler.CosineAnnealingLR(optimizer=optimizer,**kw_arguments);
    if str.lower(OPT.learning_rate_schedualer_scheme) == 'exponential':
        kw_arguments = {'gamma':None,'last_epoch':-1};  # Exponential
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments,OPT.variables_dictionary)
        return lr_scheduler.ExponentialLR(optimizer=optimizer,**kw_arguments);
    if str.lower(OPT.learning_rate_schedualer_scheme) == 'multi_step':
        kw_arguments = {'gamma':0.1,'last_epoch':-1,'milestones':None};  # Multi-Step
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments,OPT.variables_dictionary)
        return lr_scheduler.MultiStepLR(optimizer=optimizer,**kw_arguments);
    if str.lower(OPT.learning_rate_schedualer_scheme) == 'reduce_lr_on_plateau':
        kw_arguments = {'mode':'min', 'factor':0.1, 'patience':10, 'verbose':False, 'threshold':1e-4, 'threshold_mode':'rel', 'cooldown':0, 'min_lr':0, 'eps':1e-8};  # Reduce On Plateau
        kw_arguments = merge_dictionaries_if_keys_exist(kw_arguments,OPT.variables_dictionary)
        return lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,**kw_arguments);




class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return [base_lr for base_lr in self.base_lrs]
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, scheduler, mode='linear', warmup_iters=100, gamma=0.2, last_epoch=-1):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cold_lrs = self.scheduler.get_lr()

        if self.last_epoch < self.warmup_iters:
            if self.mode == 'linear':
                alpha = self.last_epoch / float(self.warmup_iters)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == 'constant':
                factor = self.gamma
            else:
                raise KeyError('WarmUp type {} not implemented'.format(self.mode))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs

########################################################################################################################################################################################################################################################################################









class L4():
    """Implements L4: Practical loss-based stepsize adaptation for deep learning
    Proposed by Michal Rolinek & Georg Martius in
    `paper <https://arxiv.org/abs/1802.05074>`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        optimizer: an optimizer to wrap with L4
        alpha (float, optional): scale the step size, recommended value is in range (0.1, 0.3) (default: 0.15)
        gamma (float, optional): scale min Loss (default: 0.9)
        tau (float, optional): min Loss forget rate (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-12)
    """

    def __init__(self, optimizer, alpha=0.15, gamma=0.9, tau=1e-3, eps=1e-12):
        # TODO: save and load, state
        self.optimizer = optimizer
        self.state = dict(alpha=alpha, gamma=gamma, tau=tau, eps=eps)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, loss):
        if loss is None:
            raise RuntimeError('L4: loss is required to step')

        if loss.data < 0:
            raise RuntimeError('L4: loss must be non negative')

        if math.isnan(loss.data):
            return

            # copy original data for parameters
        originals = {}
        # grad estimate decay
        decay = 0.9

        state = self.state
        if 'step' not in state:
            state['step'] = 0

        state['step'] += 1
        # correction_term = 1 - math.exp(state['step']  * math.log(decay))
        correction_term = 1 - decay ** state['step']

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if p not in state:
                    state[p] = torch.zeros_like(p.grad.data)

                # grad running average momentum
                state[p].mul_(decay).add_(1 - decay, p.grad.data)

                if p not in originals:
                    originals[p] = torch.zeros_like(p.data)
                originals[p].copy_(p.data)
                p.data.zero_()

        if 'lmin' not in state:
            state['lmin'] = loss.data * 0.75

        lmin = min(state['lmin'], loss.data)

        gamma = state['gamma']
        tau = state['tau']
        alpha = state['alpha']
        eps = state['eps']

        self.optimizer.step()

        inner_prod = 0

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = state[p].div(correction_term)
                v = -p.data.clone()
                inner_prod += torch.dot(grad.view(-1), -p.data.view(-1))

        lr = alpha * (loss.data - lmin * gamma) / (inner_prod + eps)
        state['lr'] = lr
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = state[p].div(correction_term)
                v = -p.data.clone()
                p.data.copy_(originals[p])
                p.data.add_(-lr, v)

        state['lmin'] = (1 + tau) * lmin

        return loss





# #(2). L4:
# import L4
# from L4 import *
# parameters_to_optimize = [
#         {
#             'params': Generator_Network.parameters()
#         }
#                               ]
# optimizer_G = SGD(parameters_to_optimize,lr=0.001/50,momentum=0.5)
# optimizer_G_L4 = L4(optimizer_G)
# #(3). LR Finder:
# network = Generator_Network
# data_loader = train_data_loader;
# criterion_function = DirectPixels_Loss_Function;
# optimizer = optimizer_G;
def find_lr(network, data_loader, criterion_function, optimizer, initial_lr_value = 1e-8, final_lr_value=10., beta = 0.98):
    #Input Parameters:
    initial_lr_value = 1e-8
    final_lr_value = 50.
    beta = 0.98
    #Initial Parameters:
    number_of_epochs = 1000;
    number_of_examples = len(data_loader)-1
    number_of_examples = number_of_examples * number_of_epochs
    mult = (final_lr_value / initial_lr_value) ** (1/number_of_examples)
    lr = initial_lr_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 10.
    batch_num = 0
    losses = []
    log_lrs = []
    #Loop over data and get the wanted behavior graph:
    for i in arange(number_of_epochs):
        for current_data in data_loader:
            batch_num += 1
            #As before, get the loss for this mini-batch of inputs/outputs
            inputs, labels = current_data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            Generator_Network.reset_hidden_states();
            outputs = network(inputs)
            loss = criterion_function(outputs, labels)
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) * loss.data
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                # return log_lrs, losses
                break;
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss.item())
            log_lrs.append(math.log10(lr))
            #Do the SGD step
            loss.backward()
            optimizer.step()
            #Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr

    #Plot the Graph:
    plot(log_lrs, losses)
    return log_lrs, losses
# #(*). Call Function:
# logs, losses = find_lr(network, data_loader, criterion_function, optimizer)


import math
import torch
import torch.optim as optim
from torch.optim import lr_scheduler


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int = 1,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def cosineannealing(optimizer):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=6e-8)
    return scheduler


def lrdecay_scheduler_kim(optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every 10 epochs. Add the paper reference here."""
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    return scheduler


def lrdecay_scheduler_shu(optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every 10 epochs"""
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.98)
    return scheduler


def multisteplr_routine1(optimizer):  # multisteplr1
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5],
                                               gamma=0.1)  # 6 means 7. epochs starts with 0. So, from 7th epoch the model will move to another learning rate
    return scheduler


def select_lr_scheduler(config_params, optimizer):
    if config_params['trainingmethod'] == 'multisteplr1':
        scheduler = multisteplr_routine1(optimizer)
    elif config_params['trainingmethod'] == 'lrdecayshu':
        scheduler = lrdecay_scheduler_shu(optimizer)
    elif config_params['trainingmethod'] == 'lrdecaykim':
        scheduler = lrdecay_scheduler_kim(optimizer)
    elif config_params['trainingmethod'] == 'cosineannealing':
        scheduler = cosineannealing(optimizer)
    elif config_params['trainingmethod'] == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif config_params['trainingmethod'] == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, gamma=0.95, step_size=5)
    return scheduler


def optimizer_fn(config_params, model):
    if config_params['viewsinclusion'] == 'all' and config_params['learningtype'] == 'MIL':
        image_attention_group = []
        side_attention_group = []
        rest_group = []
        param_list = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'img.attention' in name:
                    image_attention_group.append(param)
                elif 'side.attention' in name:
                    side_attention_group.append(param)
                else:
                    rest_group.append(param)
        for item in [image_attention_group, side_attention_group, rest_group]:
            if item:
                if config_params['optimizer'] == 'Adam':
                    param_list.append(
                        {"params": item, "lr": config_params['lr'], "weight_decay": config_params['wtdecay']})
                elif config_params['optimizer'] == 'SGD':
                    param_list.append({"params": item, "lr": config_params['lr'], "momentum": 0.9,
                                       "weight_decay": config_params['wtdecay']})

        if config_params['optimizer'] == 'Adam':
            optimizer = optim.Adam(param_list)
        elif config_params['optimizer'] == 'SGD':
            optimizer = optim.SGD(param_list)

    else:  # config_params['viewsinclusion'] == 'standard':
        classifier = []
        rest_group = []
        if config_params['optimizer'] == 'Adam':
            if config_params['papertoreproduce'] == 'shu':
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if '.fc' in name:
                            classifier.append(param)
                        else:
                            rest_group.append(param)
                optimizer = optim.Adam([{'params': classifier, 'lr': 0.0001, "weight_decay": config_params['wtdecay']},
                                        {'params': rest_group, 'lr': config_params['lr'],
                                         "weight_decay": config_params['wtdecay']}])
            else:
                # parameters_with_grad = list(filter(lambda p: p.requires_grad, model.parameters()))
                # pa = []
                # pa_name = []
                # for k, v in model.named_parameters():
                #     if v.requires_grad:
                #         pa.append({k, v})
                #         pa_name.append(k)
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config_params['lr'],
                                       weight_decay=config_params['wtdecay'])

        elif config_params['optimizer'] == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config_params['lr'],
                                  momentum=0.9, weight_decay=config_params['wtdecay'])
    return optimizer

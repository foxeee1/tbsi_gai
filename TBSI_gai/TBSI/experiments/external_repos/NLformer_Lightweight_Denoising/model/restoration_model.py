import torch
import torch.nn as nn
import torch.nn.functional as F
from model.build_model import build_model
from utils.utils_io import log
import os
from torch.optim.lr_scheduler import _LRScheduler
import math
from collections import Counter
from collections import OrderedDict


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i

class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1, ),
                 eta_mins=(0, ),
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]
        

class restoration_model():
    def __init__(self, opt):
        super(restoration_model, self).__init__()
        self.opt = opt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.network = build_model(opt).to(self.device)
        if opt['load_model'] == True:
            self.load_net(opt['model_path'])

        self.criteron = nn.L1Loss(reduction='mean')
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=opt['lr'])
        # milestones = [int(opt['num_iters'] * 0.4), int(opt['num_iters'] * 0.8)]
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        self.scheduler = CosineAnnealingRestartCyclicLR(self.optimizer, [int(opt['num_iters'] * 0.3), opt['num_iters'] - int(opt['num_iters'] * 0.3)], restart_weights=[1, 1], eta_mins=[opt['lr'], 1e-6])
        self.iter = 0


    def train_step(self, data, sup_type='unsup'):
        data_L = data['L'].to(self.device)

        self.network.train()
        output = self.network(data_L)

        if sup_type == 'unsup':
            self.loss = self.criteron(output, data_L)
        elif sup_type == 'sup':
            data_H = data['H'].to(self.device)
            self.loss = self.criteron(output, data_H)
        else:
            raise ValueError('sup_type must be either "unsup" or "sup".')
            
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.iter += 1


    def validation_step(self, data, rs = 8):
        input_noisy = data['L'].to(self.device)
        b, c, h, w = input_noisy.shape
        pad_size = (0, (rs - w % rs) % rs, 0, (rs - h % rs) % rs)
        input_pad = F.pad(input_noisy, pad_size, mode='reflect')

        self.network.eval()
        with torch.no_grad():
            output = self.network(input_pad)

        denoised = output[:, :, :h, :w]

        return denoised.detach().clone()

    def save_net(self):
        torch.save(self.network.state_dict(), os.path.join(self.opt['save_dir'], 'checkpoints', 'iter_%07d.pth' % self.iter))

    def print_net(self):
        loss_message = 'iter: %d, loss: %f' % (self.iter, self.loss.item()) + '\n'
        train_log = os.path.join(self.opt['save_dir'], 'train_log.out')
        log(train_log, loss_message, also_print=False)

    def load_net(self, path):
        state_dict = torch.load(path, map_location='cpu')
        try:
            state_dict = state_dict["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if 'module.' in k else k
                new_state_dict[name] = v
            self.network.load_state_dict(new_state_dict)

        except:
            for key in list(state_dict.keys()):
                state_dict[key.replace('module.', '')] =  state_dict.pop(key)
            self.network.load_state_dict(state_dict)

    def data_parallel(self):
        self.device = torch.device('cuda')
        self.network = nn.DataParallel(self.network).to(self.device)
import torch
from torch.optim import Optimizer
import math

class LinearWarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 max_steps: int = None,
                 warmup_iters: int = 1500,
                 warmup_ratio: float = 1e-6,
                 min_lr=0.,
                 last_epoch=-1):
        self.max_updates = max_steps
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        # warmup phase
        if self.last_epoch < self.warmup_iters:
            k = (1 - self.last_epoch / self.warmup_iters) * \
                (1 - self.warmup_ratio)
            return [_lr * (1 - k) for _lr in self.base_lrs]

        # poly phase
        else:
            coeff = 1 - (self.last_epoch - self.warmup_iters) / \
                     float(self.max_updates - self.warmup_iters)
            return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]

class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 max_steps: int = None,
                 warmup_iters: int = 1500,
                 warmup_ratio: float = 1e-6,
                 min_lr=0.,
                 last_epoch=-1):
    # def __init__(self, optimizer, T_max=2, eta_min=0, last_epoch=-1, verbose=False):
    #     self.T_max = T_max
    #     self.eta_min = eta_min
    #     super().__init__(optimizer, last_epoch, verbose)
        self.max_updates = max_steps
        self.warmup_iters = warmup_iters
        self.T_max = 2
        self.eta_min = 0
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        # if self.last_epoch < self.warmup_iters:
        #     k = (1 - self.last_epoch / self.warmup_iters) * \
        #         (1 - self.warmup_ratio)
        #     return [_lr * (1 - k) for _lr in self.base_lrs]
        
        if self.last_epoch == 0:
            return [_lr for _lr in self.base_lrs]
        
        elif self.last_epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                    for base_lr, _lr in
                    zip(self.base_lrs, self.base_lrs)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [_lr + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, _lr in
                    zip(self.base_lrs, self.base_lrs)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (_lr - self.eta_min) + self.eta_min
                for _lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
    

# def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
#     warmup_schedule = np.array([])
#     warmup_iters = warmup_epochs * niter_per_ep
#     if warmup_epochs > 0:
#         warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

#     iters = np.arange(epochs * niter_per_ep - warmup_iters)
#     schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

#     schedule = np.concatenate((warmup_schedule, schedule))
#     assert len(schedule) == epochs * niter_per_ep
#     return schedule
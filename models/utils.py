import torch
import torch.nn as nn
from PIL import Image
import os
import sys
import time
import math
import json
import random
import datetime
import subprocess
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import transforms

from collections import defaultdict, deque
from pathlib import Path

PALETTE = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
for i in range(256 * 3 - len(PALETTE)):
    PALETTE.append(0)


def colorize_mask(mask):
    assert isinstance(mask, Image.Image)
    new_mask = mask.convert('P')
    new_mask.putpalette(PALETTE)
    return new_mask


def warp(x, flo, padding_mode='zeros', return_mask=False):
    """
    ---------------------------------------------------------------------------
    Copyright (c) Prune Truong. All rights reserved.
    
    This source code is licensed under the license found in the
    LICENSE file in https://github.com/PruneTruong/DenseMatching.
    ---------------------------------------------------------------------------
    
    warp an image/tensor (im2) back to im1, according to the optical flow
    Args:
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    if torch.all(flo == 0):
        if return_mask:
            return x, torch.ones((B, H, W), dtype=torch.bool, device=x.device)
        return x
    # mesh grid
    # import pdb; pdb.set_trace()
    xx = torch.arange(0, W, dtype=flo.dtype,
                      device=flo.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, dtype=flo.dtype,
                      device=flo.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1)
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / float(max(W-1, 1)) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / float(max(H-1, 1)) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    output = nn.functional.grid_sample(
        x, vgrid, align_corners=True, padding_mode=padding_mode)
    if return_mask:
        vgrid = vgrid.detach().clone().permute(0, 3, 1, 2)
        mask = (vgrid[:, 0] > -1) & (vgrid[:, 1] > -
                                     1) & (vgrid[:, 0] < 1) & (vgrid[:, 1] < 1)
        return output, mask
    return output


def estimate_probability_of_confidence_interval(uncert_output, R=1.0):
    assert uncert_output.shape[1] == 1
    var = torch.exp(uncert_output)
    p_r = 1.0 - torch.exp(-R ** 2 / (2 * var))
    return p_r


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def step_scheduler(step_size, epochs, niter_per_ep, gamma=0.1, base_value=1):
    schedule = np.ones(epochs * niter_per_ep) * base_value
    iters = np.arange(epochs * niter_per_ep)
    schedule = schedule * (gamma ** (iters // (step_size * niter_per_ep)))
    
    assert len(schedule) == epochs * niter_per_ep
    return schedule
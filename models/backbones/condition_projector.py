"""Condition projector (psi_proj) used by FREST.

A small MLP that maps the 128-d output of the projection head onto the condition
embedding space, where the condition-specific loss (Eq. 1) and the feature
restoration loss (Eq. 3) are computed.
"""
import torch
import torch.nn as nn


class ConditionProjector(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # three-layer bottleneck: in_dim -> in_dim/2 -> in_dim/4 -> in_dim/8
        self.d = self._param(in_dim, in_dim // 2)
        self.d_2 = self._param(in_dim // 2, in_dim // 4)
        self.d_3 = self._param(in_dim // 4, in_dim // 8)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(torch.einsum('ji,ib->jb', x, self.d))
        x = self.gelu(torch.einsum('jb,bk->jk', x, self.d_2))
        x = torch.einsum('jk,kl->jl', x, self.d_3)
        return x

    @staticmethod
    def _param(in_dim, out_dim):
        p = nn.Parameter(torch.empty(in_dim, out_dim), requires_grad=True)
        nn.init.kaiming_uniform_(p, a=1)
        return p

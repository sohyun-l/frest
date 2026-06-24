"""Condition strainer (psi_strainer) used by FREST.

A lightweight, parameter-efficient module attached to each transformer block
through residual connections on the MHSA and FFN outputs. It extracts
condition-specific information while leaving the frozen encoder intact, so that
the condition-infused feature is  c = phi_enc(x) + psi_strainer(x).
See Sec. 4 and Fig. 3 of the paper.
"""
import torch
import torch.nn as nn


def tensor_init(p, init, gain=1, std=1, a=1, b=1):
    if init == 'ortho':
        nn.init.orthogonal_(p)
    elif init == 'uniform':
        nn.init.uniform_(p, a=a, b=b)
    elif init == 'normal':
        nn.init.normal_(p, std=std)
    elif init == 'zero':
        nn.init.zeros_(p)
    elif init == 'he_uniform':
        nn.init.kaiming_uniform_(p, a=a)
    elif init == 'he_normal':
        nn.init.kaiming_normal_(p, a=a)
    elif init == 'xavier_uniform':
        nn.init.xavier_uniform_(p, gain=gain)
    elif init == 'xavier_normal':
        nn.init.xavier_normal_(p, gain=gain)
    elif init == 'trunc_normal':
        nn.init.trunc_normal_(p, std=std)
    else:
        raise NotImplementedError


def tensor_prompt(x, y=None, z=None, init='xavier_uniform', gain=1, std=1, a=1, b=1):
    if y is None:
        p = nn.Parameter(torch.FloatTensor(x), requires_grad=True)
    elif z is None:
        p = nn.Parameter(torch.FloatTensor(x, y), requires_grad=True)
    else:
        p = nn.Parameter(torch.FloatTensor(x, y, z), requires_grad=True)
    if p.dim() > 2:
        tensor_init(p[0], init, gain=gain, std=std, a=a, b=b)
        for i in range(1, x):
            p.data[i] = p.data[0]
    else:
        tensor_init(p, init)
    return p


class ConditionStrainer(nn.Module):
    """Bottleneck (down-project -> GELU -> up-project) applied per block.

    One instance serves all blocks of a transformer stage. Parameters:
      d : down-projection weights,  shape [num_blocks, 2, embed_dim, low_dim]
      u : up-projection weights,    shape [num_blocks, 2, low_dim, embed_dim]
      s : learnable per-sublayer scale
    The leading size-2 axis indexes the sub-layer: 0 = after MHSA, 1 = after FFN.

    forward(x, l, f): l = block index within the stage, f = sub-layer (0/1).
    With u initialised to zero, the strainer starts as a no-op (c == phi_enc(x)).
    """

    def __init__(self, block_indices, embed_dim, d_init='he_uniform',
                 u_init='zero', low_dim=64, init_value=0.1, scale_train=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.low_dim = low_dim
        self.block_indices = block_indices
        self.non_linear = nn.GELU()
        self.d = nn.ParameterList(
            [tensor_prompt(2, embed_dim, low_dim, init=d_init) for _ in block_indices])
        self.u = nn.ParameterList(
            [tensor_prompt(2, low_dim, embed_dim, init=u_init) for _ in block_indices])
        self.s = nn.ParameterList(
            [nn.Parameter(init_value * torch.ones(2), requires_grad=scale_train) for _ in block_indices])

    def forward(self, x, l, f):
        if l not in self.block_indices:
            return 0
        x = torch.einsum('bnd,ds->bns', x, self.d[l][f])
        x = self.non_linear(x)
        x = torch.einsum('bns,sd->bnd', x, self.u[l][f])
        x = x.mul_(self.s[l][f])
        return x

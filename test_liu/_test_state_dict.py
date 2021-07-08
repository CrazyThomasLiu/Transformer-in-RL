import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch as th
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self, dim=2,num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.proj(x)
        return x

model=Attention()
print(model.state_dict())
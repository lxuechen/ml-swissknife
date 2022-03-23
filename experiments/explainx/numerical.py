import math
from typing import Union, Sequence

import torch


def logmeanexp(x: Union[Sequence[torch.Tensor], torch.Tensor], keepdim=False, dim=0):
    if isinstance(x, (tuple, list)):
        elem0 = x[0]
        if elem0.dim() == 0:
            x = torch.stack(x)
        elif elem0.dim() == 1:
            x = torch.cat(x, dim=0)
        else:
            raise ValueError
    return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(x.size(dim))

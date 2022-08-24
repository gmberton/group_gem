
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # TODO try this
    # return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), 1).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class GroupGeM(nn.Module):
    def __init__(self, num_channels, p=3, eps=1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.params = Parameter((torch.ones(num_channels)*p))
        self.eps = eps
    def forward(self, x):
        B, C, H, W = x.shape
        out = torch.zeros([B, C, 1, 1], dtype=x.dtype, device=x.device)
        assert C == self.num_channels
        for c in range(C):
            part = x[:, c:c+1, :, :]
            param = self.params[c]
            part = gem(part, p=param, eps=self.eps)
            out[:, c:c+1] = part
        return out


class FastGroupGeM(nn.Module):
    def __init__(self, num_channels, p=3, eps=1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.params = Parameter((torch.ones(num_channels)*p))
        self.eps = eps
    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.num_channels
        x = x.clamp(min=self.eps)
        x = torch.swapaxes(torch.swapaxes(x, 1, 3).pow(self.params), 1, 3)
        x = torch.swapaxes(F.adaptive_avg_pool2d(x, 1), 1, 3)
        return torch.swapaxes(x.pow(1./self.params), 1, 3)


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:,:,0,0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


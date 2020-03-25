import torch.distributions as dist
from torch import nn


class NormalizingFlowWrapper(object):
    def __init__(self, transforms, flow, base_dist):
        self.dist = dist.TransformedDistribution(base_dist, flow)
        self.modules = nn.ModuleList(transforms)

    def cuda(self):
        self.modules.cuda()

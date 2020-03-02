import pyro.distributions as dist
from torch import nn


class ConditionalNormalizingFlowWrapper(object):
    def __init__(self, transforms, flow, base_dist):
        self.dist = dist.ConditionalTransformedDistribution(base_dist, flow)
        self.modules = nn.ModuleList(transforms)

    def cuda(self):
        self.modules.cuda()

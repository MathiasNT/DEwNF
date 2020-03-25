import torch.distributions as dist
from torch import nn
import torch
from pyro.distributions.transforms import affine_coupling, permute
import itertools


class NormalizingFlowWrapper(object):
    def __init__(self, transforms, flow, base_dist):
        self.dist = dist.TransformedDistribution(base_dist, flow)
        self.modules = nn.ModuleList(transforms)

    def cuda(self):
        self.modules.cuda()


def normalizing_flow_factory(flow_depth, problem_dim, c_net_depth, c_net_h_dim, cuda):
    # We define the base distribution
    if cuda:
        base_dist = dist.Normal(torch.zeros(problem_dim).cuda(), torch.ones(problem_dim).cuda())
    else:
        base_dist = dist.Normal(torch.zeros(problem_dim), torch.ones(problem_dim))

    # We define the transformations
    transforms = [affine_coupling(input_dim=problem_dim,
                                  hidden_dims=[c_net_h_dim for i in range(c_net_depth)]) for i in range(flow_depth)]

    # We need to permute dimensions to affect them both THIS NEEDS A FIX
    perms = [permute(2, torch.tensor([1, 0])) for i in range(flow_depth)]

    # We sandwich the AffineCouplings and permutes together. Unelegant hotfix to remove last permute but it works
    flows = list(itertools.chain(*zip(transforms, perms)))[:-1]

    # We define the normalizing flow wrapper
    normalizing_flow = NormalizingFlowWrapper(transforms, flows, base_dist)
    if cuda:
        normalizing_flow.cuda()
    return normalizing_flow

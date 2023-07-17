import csv
import re
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import torch.nn.functional as F

class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()


def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) + \
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt):
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError


def to_etype_name(rating):
    return str(rating).replace('.', '_')


# def common_loss(emb1, emb2):
#     emb1 = emb1 - th.mean(emb1, dim=0, keepdim=True)
#     emb2 = emb2 - th.mean(emb2, dim=0, keepdim=True)
#     emb1 = th.nn.functional.normalize(emb1, p=2, dim=1)
#     emb2 = th.nn.functional.normalize(emb2, p=2, dim=1)
#     cov1 = th.matmul(emb1, emb1.t())
#     cov2 = th.matmul(emb2, emb2.t())
#     cost = th.mean((cov1 - cov2) ** 2)
#     return cost
def projection(args, z: th.Tensor) -> th.Tensor:
    fc1 = th.nn.Linear(args.num_hidden, args.num_proj_hidden1).to(args.device)
    fc2 = th.nn.Linear(args.num_proj_hidden1, args.num_proj_hidden2).to(args.device)
    fc3 = th.nn.Linear(args.num_proj_hidden2, args.num_hidden).to(args.device)
    z1 = F.elu(fc1(z))
    z2 = F.elu(fc2(z1))
    # z = th.sigmoid(fc1(z))
    return fc3(z2)
def sim(z1: th.Tensor, z2: th.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return th.mm(z1, z2.t())

def semi_loss(args, z1: th.Tensor, z2: th.Tensor, flag: int):
    # if flag == 0:
    #     f = lambda x: th.exp(x / args.tau_drug)
    # else:
    #     f = lambda x: th.exp(x / args.tau_disease)
    f = lambda x: th.exp(x / args.tau)
    refl_sim = f(args.intra * sim(z1, z1))  # torch.Size([663, 663])
    between_sim = f(args.inter * sim(z1, z2))  # z1 z2:torch.Size([663, 75])
    # refl_sim = f(sim(z1, z1))  # torch.Size([663, 663])
    # between_sim = f(sim(z1, z2))  # z1 z2:torch.Size([663, 75])
    # refl_sim = (F.cosine_similarity(z1, z1))  # torch.Size([663])
    # between_sim = f(F.cosine_similarity(z1, z2))

    return -th.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def batched_semi_loss(args, z1: th.Tensor, z2: th.Tensor,
                        batch_size: int):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: th.exp(x / args.tau)
    indices = th.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = f(sim(z1[mask], z1))  # [B, N]
        between_sim = f(sim(z1[mask], z2))  # [B, N]

        losses.append(-th.log(
            between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            / (refl_sim.sum(1) + between_sim.sum(1)
                - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

    return th.cat(losses)

def LOSS(args, z1: th.Tensor, z2: th.Tensor,
        mean: bool = True, batch_size: int = 0, flag: int = 0):
    h1 = projection(args, z1)
    h2 = projection(args, z2)

    if batch_size == 0:
        l1 = semi_loss(args, h1, h2, flag)
        l2 = semi_loss(args, h2, h1, flag)
    else:
        l1 = batched_semi_loss(h1, h2, batch_size)
        l2 = batched_semi_loss(h2, h1, batch_size)
    # if batch_size == 0:
    #     l1 = semi_loss(args, z1, z2)
    #     l2 = semi_loss(args, z2, z1)
    # else:
    #     l1 = batched_semi_loss(args, z1, z2, batch_size)
    #     l2 = batched_semi_loss(args, z2, z1, batch_size)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret

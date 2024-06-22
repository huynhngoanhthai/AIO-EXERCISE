import torch
import torch.nn as nn
from torch import Tensor


class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x)
        return exp_x / sum_exp_x


class SoftmaxStable(nn.Module):
    def __init__(self):
        super(SoftmaxStable, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        max_x = torch.max(x)
        exp_x = torch.exp(x - max_x)
        sum_exp_x = torch.sum(exp_x)
        return exp_x / sum_exp_x


x = torch.tensor([1, 2, 3])

softmax = Softmax()
softmax_stable = SoftmaxStable()

print("Softmax:", softmax(x))

from torch import nn
import torch
from setting import *


# 模型构建
def get_net1():
    net1 = nn.Sequential(nn.Linear(dim_in, dim_out))
    return net1


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, num_outputs)
        self.relu = nn.ReLU()


    def forward(self, X):
        H1 = self.relu(self.lin1(X))
        H2 = self.relu(self.lin2(H1))
        out = self.lin3(H2)
        return out


def get_net2():
    return Net(dim_in, dim_out, True)


loss = nn.MSELoss()


# 试一下相对误差
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), min=1, max=float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()
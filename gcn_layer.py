"""
GCN神经网络层
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch import sparse


class Layer(nn.Module):
    """
    GCN神经网络层，继承自nn，优化W参数，输入信息包括：
    1. 矩阵A_hat = D_{-1/2} * adj_hat * D_{-1/2}
    2. 节点特征矩阵H
    """

    def __init__(self, input_dim, hidden_dim, a_hat, is_sparse=True):
        """
        初始化函数
        :param input_dim: 输入的特征矩阵的维数
        :param hidden_dim:输出的特征矩阵的维数
        :param a_hat: 有图决定的定值矩阵
        :param is_sparse: 特征矩阵是否是稀疏的
        :return:
        """
        super(Layer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.a_hat = a_hat
        self.is_sparse = is_sparse
        self.w = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim))

    def forward(self, x):
        # x为输入的特征矩阵
        # 如果特征矩阵以稀疏矩阵形式存储
        # if self.is_sparse:
        #     x = sparse.mm(x, self.w)
        # else:
        # print(type(x))
        x = torch.mm(x, self.w)
        out = torch.mm(self.a_hat, x)
        output = F.relu(out)
        return output

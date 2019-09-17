"""
实现GCN的模型
"""
import torch
from torch import nn
from gcn_layer import Layer


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, a_hat, num_class_label):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            Layer(self.input_dim, 512, a_hat, is_sparse=True),
            Layer(512, output_dim, a_hat, is_sparse=False)
        )
        self.fc = nn.Linear(output_dim, num_class_label)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.layers(x)
        out = self.dropout(out)
        # print(type(out))
        out = self.fc(out)
        return out

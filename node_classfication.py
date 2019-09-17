"""
运用GCN模型进行节点分类
"""
import torch
from torch import nn, optim
from model import Model
from data_processing import Graph


class NodeClassification(object):
    def __init__(self,
                 training_times,
                 lr,
                 ):
        g = Graph()
        g.read_from_file('./data/cora/cora_edgelist.txt')
        label_set, self.all_label = g.read_node_label('./data/cora/cora_labels.txt')
        g.read_node_features('./data/cora/cora.features')
        self.g = g
        self.a = torch.Tensor(self.g.a_hat)
        self.lr = lr
        self.training_times = training_times
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.gcn = Model(self.g.x.shape[1], 512, self.a, len(label_set)).to(self.device)
        self.optimizer = optim.Adam(self.gcn.parameters(), lr=self.lr)
        self.createon = nn.CrossEntropyLoss()

    def train(self):
        # 取出所有的节点还有对应的索引
        all_label = []
        all_label_idx = []
        for i, j in self.all_label:
            all_label.append(j)
            all_label_idx.append(i)
        # label_x是取出图中固定的20个点作为已知节点的标签计算交叉熵
        label_x = all_label[:300]
        label_x_idx = all_label_idx[:300]
        label_x = torch.Tensor(label_x).long().to(self.device)
        # x是图中输入的特征矩阵
        x = torch.Tensor(self.g.x).to(self.device)
        for epoch in range(self.training_times):
            output = self.gcn(x)
            pred_label_matrix = output[label_x_idx]
            loss = self.createon(pred_label_matrix, label_x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('epoch {} || loss is :{}'.format(epoch, loss.item()))
            pred_labels = torch.max(output[300:], dim=1)[1].data.numpy()
            if epoch == self.training_times - 1:
                print('The acc is: ', sum(pred_labels == all_label[300:]) / len(all_label[300:]))


if __name__ == '__main__':
    nc = NodeClassification(500, 0.01)
    nc.train()

"""
对GCN的输入数据进行封装，使用networkx
"""
import networkx as nx
import numpy as np


class Graph(object):
    """
    构建一个无向图，用于输入到GCN中，需要以下信息：
    1. 图的邻接矩阵A，索引按照节点数字从小到大      A = [N * N]
    2. 图的对角矩阵D，索引需要与邻接矩阵一致        D = [N * N]
    3. 图的节点的特征矩阵，索引需要与邻接矩阵一致    F = [N * D]
    """
    def __init__(self):
        """
        初始化图
        """
        self.G = None
        self.node_size = 0

    def read_from_file(self, filename):
        """
        从一个给定格式的文件中构建一个图
        并从这个图中得到A/D
        :param filename: 输入的图结构文件
        :return:包含自身节点的邻接矩阵adj_hat, 从a_hat中得到的d_hat
        """
        # 从文件中读取出一个图
        self.G = nx.read_edgelist(filename)
        # 节点总个数
        self.node_size = self.G.number_of_nodes()
        # 将节点按照值得大小排序，方便后续操作
        soreted_node_list = sorted(self.G.nodes(), key=lambda x: int(x))
        # 邻接矩阵
        adj_matrix = nx.convert_matrix.to_numpy_array(self.G, nodelist=soreted_node_list)
        assert len(self.G.nodes()) == len(soreted_node_list), "转化的节点列表不一致"
        # A_HAT矩阵，加上一个单位矩阵
        adj_hat = adj_matrix + np.eye(adj_matrix.shape[0])
        print(adj_hat)
        # 生成矩阵D_HAT
        d_hat = np.zeros_like(adj_matrix)
        for node in self.G.nodes():
            d_hat[int(node), int(node)] = int(self.G.degree[node]) + 1
        return adj_hat, d_hat

    def read_node_label(self, label_file):
        """
        读取节点的标签，加入到节点属性中
        :param label_file: 标签文件
        :return: None
        """
        assert self.G is not None, '必须先要构建一个图'
        fin = open(label_file, 'r', encoding='utf8')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['label'] = vec[1:]
        fin.close()
        print(list(self.G.node(data=True)))
        # todo：GCN训练是一个半监督学习，不需要整个的标签，只需要少量的标签。

    def read_node_features(self, filename):
        """
        读取节点的特征，特征使用one-hot编码
        :param filename: 节点特征文件，格式如文件所述
        :return: None
        """
        assert self.G is not None, '必须先要构建一个图'
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array(
                [float(x) for x in vec[1:]])
        fin.close()
        print(list(self.G.node(data=True)))


if __name__ == '__main__':
    g = Graph()
    g.read_from_file('./data/cora/cora_edgelist.txt')
    g.read_node_label('./data/cora/cora_labels.txt')
    g.read_node_features('./data/cora/cora.features')

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hid=16, drop=0.5, set_seed=False, rd_seed=1234):
        super(GCN, self).__init__()
        # 网络各层的维度初始化
        if set_seed:
            torch.manual_seed(rd_seed)
        self.conv1 = GCNConv(num_features, num_hid)
        self.conv2 = GCNConv(num_hid, num_classes)
        self.drop = drop
        # self.conv2 = GCNConv(16, 4)
        # self.conv3 = GCNConv(4, 2)
        # self.classifier = Linear(num_hid, num_classes)

    def forward(self, x, edge_index):
        # h = self.conv1(x, edge_index)
        # h = F.relu(self.conv1(x, edge_index))
        # h = self.conv2(h, edge_index)
        # h = h.tanh()
        # h = self.conv3(h, edge_index)
        # h = h.tanh()

        # out = self.classifier(h)
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.drop, training=self.training)
        h = self.conv2(h, edge_index)
        out = F.softmax(h, dim=1)
        return out, h

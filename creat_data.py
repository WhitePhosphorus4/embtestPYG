import torch
from torch_geometric.data import Data
from tools.utils import visualize_graph, visualize_embedding
from torch_geometric.utils import to_networkx

# 边连接
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 0]], dtype=torch.long)

# 节点特征
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
y = torch.tensor([0, 1, 2], dtype=torch.long)

data = Data(x = x, y=y, edge_index = edge_index)
G = to_networkx(data, to_undirected=False)
visualize_graph(G, color = data.y)

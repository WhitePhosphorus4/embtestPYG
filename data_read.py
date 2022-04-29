from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
from tools.utils import visualize_graph, visualize_embedding

# 读取KarateClub数据集
dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]   # 取第一个图
# 查看图的更多细节
print(f'Data: {data}')
print(f'Number of nodes: {data.num_nodes}') # 图中节点数
print(f'Number of edges: {data.num_edges}') # 图中边数
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}') # 平均节点度
print(f'train_mask: {data.train_mask}') # 训练集的mask
print(f'Number of training nodes: {data.train_mask.sum()}') # 训练集中节点数(即已有标签的节点数)
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}') # 训练集中节点标签比例
print(f'Has isolated nodes: {data.has_isolated_nodes()}') # 是否有孤立节点
print(f'Has self-loops: {data.has_self_loops()}') # 是否有自环
print(f'Is undirected: {data.is_undirected()}') # 是否是无向图，另一个相似函数为is_directed()，表示是否为有向图
print(f'data.x: {data.x}') # 节点特征
print(f'data.y: {data.y}') # 节点标签

edge_index = data.edge_index
print(f'Edge index: {edge_index.t()}')  # 边的索引，第一行为起点，第二行为终点。t()表示对矩阵进行转置
print(f'Edge index size: {edge_index.size()}')  # 边的索引的维度，即有多少条边

G = to_networkx(data, to_undirected=True)      # 将数据转换为networkx图，to_undirected()表示是否转换为无向图
visualize_graph(G, color = data.y)     # 可视化图


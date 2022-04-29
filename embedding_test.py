from operator import mod
# from sklearn import datasets
from torch_geometric.datasets import KarateClub
from tools.models import GCN
from tools.utils import visualize_graph, visualize_embedding


datasets = KarateClub()
graph_data = datasets[0]
model = GCN(num_features=datasets.num_features, num_classes=datasets.num_classes)

# print(model)

classify_result, embedding_result = model(graph_data.x, graph_data.edge_index)
print(f"Embedding shape: {list(embedding_result.shape)}")

visualize_embedding(embedding_result, color=graph_data.y)
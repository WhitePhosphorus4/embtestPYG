# from pickletools import optimize
from sys import displayhook
import time
# from numpy import kron
from sklearn import datasets
import torch
# from torch_cluster import graclus_cluster
import torch.optim as optim
from tools.models import GCN
from torch_geometric.datasets import KarateClub, Planetoid
from tools.utils import visualize_graph, visualize_embedding, accuracy
# from IPython.display import Javascript  # Restrict height of output cell.
# displayhook(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 430})'''))

max_epoch = 5001
datasets = KarateClub()
# dataset = Planetoid(root='./dataset/Cora', name='Cora')
graph_data = datasets[0]
model = GCN(num_features=datasets.num_features, num_classes=datasets.num_classes, set_seed=True, rd_seed=1444)

# criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.functional.nll_loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)

def train(data, epoch):
    model.train()
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    # loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss = torch.nn.functional.nll_loss(out[:], data.y[:])
    acc = accuracy(out[:], data.y[:])
    if epoch % 10 == 0:
        print("epoch: {}, loss: {:.4f}, acc: {:.4f}".format(epoch, loss.item(), acc))
    loss.backward()
    optimizer.step()
    return loss, h

for epoch in range(max_epoch):
    loss, h = train(graph_data, epoch)
    if epoch % 1000 == 0 and epoch != 0:
        visualize_embedding(h, color=graph_data.y, epoch=epoch, loss=loss)
        # time.sleep(0.3)

# time.sleep(1)
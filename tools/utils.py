# 图的可视化
# %matplotlib inline
# import torch
import networkx as nxc
import matplotlib.pyplot as plt


def visualize_graph(G, color = "b"):
    '''
    可视化图，输入为networkx图
    若使用graphgeometric建图，则可使用graphgeometric.utils.to_networkx()函数转换为networkx图
    '''
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nxc.draw_networkx(G, pos=nxc.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    '''
    可视化embedding图
    主要可视化完成embedding的图结果，可以输入epoch和loss，用于显示训练过程
    '''
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


def accuracy(output, labels):
    '''
    计算准确率
    '''
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
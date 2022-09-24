# 图的可视化
# %matplotlib inline
# import torch
import networkx as nxc
import matplotlib.pyplot as plt
import numpy as np 
import scipy.sparse as sp


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
    h = h.detach().cpu().numpy()
    h = normalize(h)
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    plt.xticks(np.arange(-1, 1.1, 0.4))
    plt.yticks(np.arange(-1, 1.1, 0.4))
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


def normalize(h):
    '''
    Normalize the matrix h by max-min normalization.
    '''
    for i in range(h.shape[1]):
        t = h[:, i]
        max_t = t.max()
        h[:, i] = t / max_t
    return h


def normalize_maxmin(Mx, axis=2):
    '''
    Normalize the matrix Mx by max-min normalization.
    axis=0: normalize each row
    axis=1: normalize each column
    axis=2: normalize the whole matrix
    '''
    if axis == 1:
        M_min = np.min(Mx, axis=1)
        M_max = np.max(Mx, axis=1)
        for i in range(Mx.shape[1]):
            Mx[:, i] = (Mx[:, i] - M_min) / (M_max - M_min)
    elif axis == 0:
        M_min = np.min(Mx, axis=0)
        M_max = np.max(Mx, axis=0)
        for i in range(Mx.shape[0]):
            Mx[i, :] = (Mx[i, :] - M_min) / (M_max - M_min)
    elif axis == 2:
        M_min = np.amin(Mx)
        M_max = np.amax(Mx)
        Mx = (Mx - M_min) / (M_max - M_min)
    else:
        print('Error')
        return None
    return Mx

def normalize_Zscore(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    mx.dot(r_mat_inv_sqrt)
    mx.transpose()
    mx.dot(r_mat_inv_sqrt)
    return mx
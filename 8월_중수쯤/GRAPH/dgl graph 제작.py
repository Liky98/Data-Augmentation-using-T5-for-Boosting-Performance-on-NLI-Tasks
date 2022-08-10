import ogb
from ogb.lsc import MAG240MDataset
import tqdm
import numpy as np
import torch
import dgl
import dgl.function as fn
import argparse
import os
import torch
import matplotlib.pyplot as plt


ei_writes = [ torch.tensor([0, 1]), torch.tensor([1, 2]) ]
ei_affiliated = [ torch.tensor([0, 1]), torch.tensor([1, 2]) ]
ei_cites = [ torch.tensor([0, 3]), torch.tensor([3, 4]) ]


g = dgl.heterograph({
    ('author', 'write', 'paper'): (ei_writes[0], ei_writes[1]),
    ('paper', 'write-by', 'author'): (ei_writes[1], ei_writes[0]),
    ('author', 'affiliate-with', 'institution'): (ei_affiliated[0], ei_affiliated[1]),
    ('institution', 'affiliate', 'author'): (ei_affiliated[1], ei_affiliated[0]),
    ('paper', 'cite', 'paper'): (np.concatenate([ei_cites[0], ei_cites[1]]), np.concatenate([ei_cites[1], ei_cites[0]]))
})

print(g)


def show(g):
    import networkx as nx

    g = dgl.to_homogeneous(g)

    nx_G = g.to_networkx().to_undirected()

    pos = nx.kamada_kawai_layout(nx_G)
    pos = nx.spring_layout(nx_G)

    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()

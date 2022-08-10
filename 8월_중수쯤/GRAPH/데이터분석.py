import ogb
from ogb.lsc import MAG240MDataset
import tqdm
import numpy as np
import torch
import dgl
import matplotlib.pyplot as plt
import networkx as nx

"""ssl 오류뜨면 쓰는 코드"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
"""-------------------"""

print('Building graph')
dataset = MAG240MDataset()

ei_writes = dataset.edge_index('author', 'writes', 'paper')
ei_cites = dataset.edge_index('paper', 'paper')
ei_affiliated = dataset.edge_index('author', 'institution')

split_dict = dataset.get_idx_split()
valid_idx = split_dict['valid'] # numpy array storing indices of validation paper nodes


ei_writes = ei_writes[0:2, :3]
print(ei_writes)

ei_cites = ei_cites[0:2, :3]
print(ei_cites)

ei_affiliated = ei_affiliated[0:2, :3]
print(ei_affiliated)


g = dgl.heterograph({
    ('author', 'write', 'paper'): (ei_writes[0], ei_writes[1]),
    ('paper', 'write-by', 'author'): (ei_writes[1], ei_writes[0]),
    ('author', 'affiliate-with', 'institution'): (ei_affiliated[0], ei_affiliated[1]),
    ('institution', 'affiliate', 'author'): (ei_affiliated[1], ei_affiliated[0]),
    ('paper', 'cite', 'paper'): (np.concatenate([ei_cites[0], ei_cites[1]]), np.concatenate([ei_cites[1], ei_cites[0]]))
})

print(g)


g = dgl.to_homogeneous(g)
print(1)
nx_G = g.to_networkx().to_undirected()
print(2)
#pos = nx.kamada_kawai_layout(nx_G)
pos = nx.spring_layout(nx_G)
print(3)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
print(4)
plt.show()

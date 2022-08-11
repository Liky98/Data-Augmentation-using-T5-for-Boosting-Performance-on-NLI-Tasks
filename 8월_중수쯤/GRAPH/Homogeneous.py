import dgl
import torch

hg = dgl.heterograph({
    ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    ('developer', 'develops', 'game'): (torch.tensor([0, 1]), torch.tensor([2, 3]))
    })

print(f'heterograph > {hg}\n')
"""
Graph(num_nodes={'developer': 2, 'game': 2, 'user': 3},
      num_edges={('developer', 'develops', 'game'): 2, ('user', 'follows', 'user'): 2},
      metagraph=[('developer', 'game', 'develops'), ('user', 'user', 'follows')])
    노드 수 > 
"""
import networkx as nx
import matplotlib.pyplot as plt

g = dgl.to_homogeneous(hg)
print(f'g > {g}\n')

nx_G = g.to_networkx().to_undirected()
print(f'nx_G > {nx_G}\n')

pos = nx.kamada_kawai_layout(nx_G)
print(f'kawai > {pos}\n')


pos = nx.spring_layout(nx_G)

nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
plt.show()

#%%
hg.nodes['user'].data['h'] = torch.ones(3, 1)
hg.nodes['developer'].data['h'] = torch.zeros(2, 1)
hg.nodes['game'].data['h'] = torch.ones(2, 1)
g = dgl.to_homogeneous(hg)

# The first three nodes are for 'user', the next two are for 'developer',
# and the last two are for 'game'
g.ndata
# The first two edges are for 'follows', and the next two are for 'develops' edges.
g.edata
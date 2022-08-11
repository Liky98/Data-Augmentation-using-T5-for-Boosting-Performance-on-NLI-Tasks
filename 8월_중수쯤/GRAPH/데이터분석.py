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
valid_idx = split_dict['valid']
#%%
print(f'validation_idx > {valid_idx}')  #[     8016      8596      8665 ... 121749594 121751492 121751535]
print(f'validation_idx_length > {valid_idx.shape}') #(138949,)

print(f'paper_label > {dataset.paper_label}')
print(f'paper_year > {dataset.paper_year}')
print(f'paper_feature > {dataset.paper_feat}')

edge_id_writes = ei_writes[0:2, :3]
print(edge_id_writes[0])
print(edge_id_writes[1])

edge_id_cites = ei_cites[0:2, :3]
print(edge_id_cites)

edge_id_affiliated = ei_affiliated[0:2, :3]
print(edge_id_affiliated)

print(np.concatenate([edge_id_cites[0], edge_id_cites[1]]))
#%%
g = dgl.heterograph({
    ('author', 'write', 'paper'): (edge_id_writes[0], edge_id_writes[1]), # 저자-논문 관계
    ('paper', 'write-by', 'author'): (edge_id_writes[1], edge_id_writes[0]), # 논문-저자 관계

    ('author', 'affiliate-with', 'institution'): (edge_id_affiliated[0], edge_id_affiliated[1]), # 저자-학회 관계
    ('institution', 'affiliate', 'author'): (edge_id_affiliated[1], edge_id_affiliated[0]), # 학회-저자 관계

    ('paper', 'cite', 'paper'): (np.concatenate([edge_id_cites[0], edge_id_cites[1]]), np.concatenate([edge_id_cites[1], edge_id_cites[0]])) #논문-논문 관계 (레퍼런스?)
})

print(g)
print(torch.tensor([0, 3]))
#%%
# g = dgl.to_homogeneous(g)
# print(1)
# nx_G = g.to_networkx().to_undirected()
# print(2)
# #pos = nx.kamada_kawai_layout(nx_G)
# pos = nx.spring_layout(nx_G)
# print(3)
# nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
# print(4)
# plt.show()

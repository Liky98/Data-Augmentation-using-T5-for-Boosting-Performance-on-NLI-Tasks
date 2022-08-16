import argparse

import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

dataset = PygNodePropPredDataset(name='ogbn-mag')
print(f"dataset > {dataset}")

split_idx = dataset.get_idx_split()
graph = dataset[0][0]
label = dataset[0][1]
print(f"graph > {graph}\n")
print(f"edge_index_dict > {graph['edge_index_dict']}\n")
print(f"edge_feat_dict > {graph['edge_feat_dict']}\n")
print(f"num_nodes_dict > {graph['num_nodes_dict']}\n")
print(f"edge_reltype > {graph['edge_reltype']}\n")  # relation

#%%
# print(split_idx['train'])
# print(graph['num_nodes_dict']['paper'])
torch.cuda.is_available()

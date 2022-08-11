"""
ogbn-arxiv

2017년까지 발표된 논문에 대해 교육하고, 2018년에 발표된 논문에 대해 검증하고, 2019년 이후에 발표된 논문에 대해 테스트
"""
from ogb.nodeproppred import NodePropPredDataset
dataset = NodePropPredDataset(name='ogbn-arxiv')

print("-----dataset config-------")
print(f'| dataset name = {dataset.name}')
print(f'| edge_index = {dataset.graph["edge_index"].shape}')
print(f'| edge_feat = {dataset.graph["edge_feat"]}')
print(f'| node_feat = {dataset.graph["node_feat"].shape}')
print(f'| node_year = {dataset.graph["node_year"].shape}')
print(f'| num_nodes = {dataset.graph["num_nodes"]}')
print(f'| num_classes = {dataset.num_classes}')
print(f'| labels = {dataset.labels.shape}')
print(f'| train idx = {dataset.get_idx_split()["train"].shape}')
print(f'| validation idx = {dataset.get_idx_split()["valid"].shape}')
print(f'| test idx = {dataset.get_idx_split()["test"].shape}')

# ei_writes = dataset.edge_index('author', 'writes', 'paper')
# ei_cites = dataset.edge_index('paper', 'paper')
# ei_affiliated = dataset.edge_index('author', 'institution')
dataset.graph

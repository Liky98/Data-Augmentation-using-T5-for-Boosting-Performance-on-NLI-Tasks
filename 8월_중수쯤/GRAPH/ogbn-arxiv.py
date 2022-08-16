"""
ogbn-arxiv

2017년까지 발표된 논문에 대해 교육하고, 2018년에 발표된 논문에 대해 검증하고, 2019년 이후에 발표된 논문에 대해 테스트
"""
from ogb.nodeproppred import NodePropPredDataset, Evaluator
import torch
import numpy as np
import argparse
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl
import matplotlib.pyplot as plt
import networkx as nx

n_node_feats, n_classes = 0, 0

def print_config():
    dataset = NodePropPredDataset(name='ogbn-mag')
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

def load_data(dataset, args):
    global n_node_feats, n_classes

    data = DglNodePropPredDataset(name=dataset)

    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    # Replace node features here
    if args.pretrain_path != 'None':
        graph.ndata["feat"] = torch.tensor(np.load(args.pretrain_path)).float()
        print("Pretrained node feature loaded! Path: {}".format(args.pretrain_path))

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return data, graph, labels, train_idx, val_idx, test_idx, evaluator

def preprocess(graph):
    global n_node_feats

    # make bidirected 이중 방향 그래프로 바꾸기
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph


if __name__ == "__main__" :
    argparser = argparse.ArgumentParser("DRGAT implementation on ogbn-arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--pretrain_path", type=str, default='None', help="path for pretrained node features")
    args = argparser.parse_args()

    data, graph, labels, train_idx, val_idx, test_idx, evaluator = load_data('ogbn-arxiv', args)
    print(f'| graph = {graph}')
    print(f'| labels = {labels}')
    print(f'| train_idx = {train_idx}')
    print(f'| val_idx = {val_idx}')
    print(f'| test_idx = {test_idx}')
    print(f'| evaluator = {evaluator}')

    graph = preprocess(graph)
    print(f'| graph = {graph}\n')

    """
    각 노드의 feature 정보를 표현하기 위해서는 ndata라는 속성을 이용하면 됩니다.
    각 엣지의 feature 정보를 표현하기 위해서는 edata라는 속성이 있습니다. 
    
    """

    print(f'temp 출력 > {graph.nodes[2]}')
    # G = dgl.to_networkx(graph)
    # plt.figure(figsize=[15, 7])
    # nx.draw(G)


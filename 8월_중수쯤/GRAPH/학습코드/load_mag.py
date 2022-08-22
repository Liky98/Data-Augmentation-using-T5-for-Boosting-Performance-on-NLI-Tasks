import os
import dgl
import torch
from torch_sparse import SparseTensor
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from ogb.lsc import MAG240MDataset

def get_ogb_evaluator(dataset):
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]

def load_mag(symmetric=True):
    #dataset = MAG240MDataset(root="../dataset/")
    dataset = DglNodePropPredDataset(name="ogbn-mag", root="../dataset/")

    embed_size = 256

    g, init_labels = dataset[0]
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx['train']['paper']
    val_nid = splitted_idx['valid']['paper']
    test_nid = splitted_idx['test']['paper']

    features = g.nodes['paper'].data['feat']
    if len(''): #extra embedding 사용할때
        print(f'Use extra embeddings generated with the {""} method')
        path = os.path.join("../data/'", f'{""}_nars')
        author_emb = torch.load(os.path.join(path, 'author.pt'), map_location=torch.device('cpu')).float()
        topic_emb = torch.load(os.path.join(path, 'field_of_study.pt'), map_location=torch.device('cpu')).float()
        institution_emb = torch.load(os.path.join(path, 'institution.pt'), map_location=torch.device('cpu')).float()

    else:

        author_emb = torch.Tensor(g.num_nodes('author'), embed_size).uniform_(-0.5, 0.5)
        topic_emb = torch.Tensor(g.num_nodes('field_of_study'), embed_size).uniform_(-0.5, 0.5)
        institution_emb = torch.Tensor(g.num_nodes('institution'), embed_size).uniform_(-0.5, 0.5)

    with torch.no_grad():
        if features.size(1) != embed_size:
            rand_weight = torch.Tensor(features.size(1), embed_size).uniform_(-0.5, 0.5)
            features = features @ rand_weight
        if author_emb.size(1) != embed_size:
            rand_weight = torch.Tensor(author_emb.size(1), embed_size).uniform_(-0.5, 0.5)
            author_emb = author_emb @ rand_weight
        if topic_emb.size(1) != embed_size:
            rand_weight = torch.Tensor(topic_emb.size(1), embed_size).uniform_(-0.5, 0.5)
            topic_emb = topic_emb @ rand_weight
        if institution_emb.size(1) != embed_size:
            rand_weight = torch.Tensor(institution_emb.size(1), embed_size).uniform_(-0.5, 0.5)
            institution_emb = institution_emb @ rand_weight

    g.nodes['paper'].data['feat'] = features
    g.nodes['author'].data['feat'] = author_emb
    g.nodes['institution'].data['feat'] = institution_emb
    g.nodes['field_of_study'].data['feat'] = topic_emb

    init_labels = init_labels['paper'].squeeze()
    n_classes = int(init_labels.max()) + 1
    evaluator = get_ogb_evaluator("ogbn-mag")

    # for k in g.ntypes:
    #     print(k, g.ndata['feat'][k].shape)
    for k in g.ntypes:
        print(k, g.nodes[k].data['feat'].shape)

    adjs = []
    for i, etype in enumerate(g.etypes):
        src, dst, eid = g._graph.edges(i)
        adj = SparseTensor(row=dst, col=src)
        adjs.append(adj)
        print(g.to_canonical_etype(etype), adj)

    # F --- *P --- A --- I
    # paper : [736389, 128]
    # author: [1134649, 256]
    # institution [8740, 256]
    # field_of_study [59965, 256]

    new_edges = {}
    ntypes = set()

    etypes = [ # src->tgt
        ('A', 'A-I', 'I'),
        ('A', 'A-P', 'P'),
        ('P', 'P-P', 'P'),
        ('P', 'P-F', 'F'),
    ]

    if symmetric:
        adjs[2] = adjs[2].to_symmetric()
        assert torch.all(adjs[2].get_diag() == 0)

    for etype, adj in zip(etypes, adjs):
        stype, rtype, dtype = etype
        dst, src, _ = adj.coo()
        src = src.numpy()
        dst = dst.numpy()
        if stype == dtype:
            new_edges[(stype, rtype, dtype)] = (np.concatenate((src, dst)), np.concatenate((dst, src)))
        else:
            new_edges[(stype, rtype, dtype)] = (src, dst)
            new_edges[(dtype, rtype[::-1], stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)

    new_g = dgl.heterograph(new_edges)
    new_g.nodes['P'].data['P'] = g.nodes['paper'].data['feat']
    new_g.nodes['A'].data['A'] = g.nodes['author'].data['feat']
    new_g.nodes['I'].data['I'] = g.nodes['institution'].data['feat']
    new_g.nodes['F'].data['F'] = g.nodes['field_of_study'].data['feat']

    IA, PA, PP, FP = adjs
    """
    이부분은 sparse 라이브러리 문제있는 코드.
    """

    return new_g, init_labels, new_g.num_nodes('P'), n_classes, train_nid, val_nid, test_nid, evaluator

new_g, init_labels, new_g_num_nodes, n_classes, train_nid, val_nid, test_nid, evaluator = load_mag()

from ogb.lsc import MAG240MDataset
import numpy as np


dataset = MAG240MDataset()

    # split_dict = dataset.get_idx_split()
    # valid_idx = split_dict['valid']
    #
    # edge_index_writes = dataset.edge_index('author', 'paper') # edge type can be omitted and inferred by their package.
    # edge_index_cites  = dataset.edge_index('paper', 'paper')
    # edge_index_affiliated_with = dataset.edge_index('author', 'institution')
    #
    #
    # print(f'edge_index_writes : {edge_index_writes.shape}')
    # print(f'edge_index_cites : {edge_index_cites.shape}')
    # print(f'edge_index_affiliated_with : {edge_index_affiliated_with.shape}')
    # """
    # edge_index_writes : (2, 386022720)
    # edge_index_cites : (2, 1297748926)
    # edge_index_affiliated_with : (2, 44592586)
    # """
    #
    #
    # g = dgl.heterograph({
    #     ('author', 'write', 'paper'): (edge_index_writes[0], edge_index_writes[1]),
    #     ('paper', 'write-by', 'author'): (edge_index_writes[1], edge_index_writes[0]),
    #     ('author', 'affiliate-with', 'institution'): (edge_index_affiliated_with[0], edge_index_affiliated_with[1]),
    #     ('institution', 'affiliate', 'author'): (edge_index_affiliated_with[1], edge_index_affiliated_with[0]),
    #     ('paper', 'cite', 'paper'): (np.concatenate([edge_index_cites[0], edge_index_cites[1]]), np.concatenate([edge_index_cites[1], edge_index_cites[0]]))
    #     })
    #
    # g = g.formats('csc')
    # dgl.save_graphs('./graph.dgl', g)

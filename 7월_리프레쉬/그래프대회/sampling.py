from ogb.lsc import MAG240MDataset
import ogb

dataset = MAG240MDataset("./val_sampling.pt")


# print(dataset)
# import torch
# data = torch.load("val_sampling.pt")
# print(data.edge_index('paper', 'paper'))
import numpy as np
import torch
data = torch.randn([128, 32, 3]) # n_batch, seq_len , d_k
print(data.shape)
data = data.transpose(-2,-1)
print(data.shape)
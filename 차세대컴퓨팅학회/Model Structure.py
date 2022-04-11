import torch
path = 'output/(최종)training_roberta_STS_SNLI.pth'
model = torch.load(path)

model.forward

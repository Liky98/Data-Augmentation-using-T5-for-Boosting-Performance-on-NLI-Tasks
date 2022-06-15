import argparse
import torch.nn as nn
import torch
from torch.autograd import Variable
import LSTM
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from transformers import BertTokenizer

args = argparse.ArgumentParser() # 인자값 받을 수 있는 인스턴스 생성
args.add_argument("--num_epochs", default=2000, type = int, help='size')
args.add_argument("--learning_rate", default=1e-5, type = int, help='size')
args.add_argument("--num_layers", default=1, type=int)
args.add_argument("--num_classes", default=1, type=int)

args.add_argument("--input_size", default=1, type = int, help='LSTM Model Input size')
args.add_argument("--hidden_size", default=2, type= int, help='LSTM')
args.add_argument("--dropout", default=0.3, type=float, help='LSTM')
args.add_argument("--seq_length", default=1, type=int, help='LSTM')
args = args.parse_args() #내용 저장

raw_data = ["Hello World"]
#label = 1

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(raw_data[0])

label = 1
#%%
print(tokenizer.tokenize(raw_data[0]))
#%%
model = LSTM.LSTM(args)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate)

for epoch in range(args.num_epochs) :
    for data in raw_data :
        data = tokenizer.tokenize(data)

        output = model(data)
        optimizer.zero_grad()

        loss = loss_function(output,label)

        loss.backward()
        optimizer.step()

        if epoch %100 == 0 :
            print("epoch: $d, loss : %1.5f" % (epoch, loss.item()))

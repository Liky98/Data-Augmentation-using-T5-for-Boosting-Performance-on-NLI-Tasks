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
args.add_argument("--batch_size", default=1, type=int)
args = args.parse_args() #내용 저장

#%%
raw_data = ["Hello World", "Hello word"]
label = [0,1]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_inp = tokenizer.encode_plus(raw_data, max_length=32, pad_to_max_length=True,
                                      return_tensors="pt")
tokenized_output = tokenizer.encode_plus(label, max_length=32, pad_to_max_length=True,
                                         return_tensors="pt")

#%%
print(tokenized_inp['input_ids'].shape)
print(tokenized_output)
#%%

model = LSTM.LSTM_RNN기반(args)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate)

for epoch in range(args.num_epochs) :
    #hidden = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True)
    #cell = torch.zeros(1, args.batch_size, args.hidden_size, requires_grad=True)
    output = model(tokenized_inp['input_ids'])
    loss = loss_function(output, tokenized_output['input_ids'])

    optimizer.zero_grad()

    loss = loss_function(output,label)

    loss.backward()
    optimizer.step()

    if epoch %100 == 0 :
        print("epoch: $d, loss : %1.5f" % (epoch, loss.item()))


